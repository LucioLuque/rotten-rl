"""
rotten_rl/evaluation.py

Evaluador reusable para agentes de La Podrida sobre el wrapper de OpenSpiel.

No depende de torch: recibe una funcion `act_fn(state, legal) -> action_id`
para el asiento evaluado, y una `opponent_fn(state, legal) -> action_id` para
el resto. Asi sirve igual para evaluar NFSP, PSRO, una heuristica o random.

Metricas (todas del documento del proyecto), por partida y por ronda:
  - win_rate                : fraccion de partidas ganadas por el asiento (con
                              reparto justo de empates)
  - mean_score_diff         : E[S_i - mean(S_opp)]  sobre puntaje CRUDO de partida
  - mean_return             : retorno del wrapper (segun reward_scheme)
  - norm_round_score        : E[S_ronda / (10 + 2h)]
  - exact_bid_rate          : # rondas con tricks == bid / # rondas
  - mae_bid                 : E[|bid - tricks|]
  - bid_bias                : E[bid - tricks]  (con signo)
  - calibration[b]          : E[tricks | bid = b]
  - win_rate_by_seat[s]     : win_rate cuando el evaluado se sienta en s
Todas las de nivel-ronda se reportan ademas DESGLOSADAS POR h.

Oponentes, de menor a mayor exigencia:
  - random    : piso; solo confirma que el pipeline anda (se le gana rapido).
  - heuristic : la vara real; ganarle exige cantar y jugar con criterio.
  - network   : versiones anteriores del propio agente (snapshots).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Dict, Optional

import numpy as np

from .engine import RANKS, SUITS, Card


ActFn = Callable[[object, list], int]


# ----------------------------------------------------------------------
# Oponentes de referencia
# ----------------------------------------------------------------------

def random_opponent(rng: np.random.RandomState) -> ActFn:
    def act(state, legal):
        return int(rng.choice(legal))
    return act


def _beats(candidate: Card, best: Card, trump) -> bool:
    if trump is not None and candidate.suit == trump and best.suit != trump:
        return True
    if best.suit == trump and candidate.suit != trump:
        return False
    if candidate.suit == best.suit:
        return candidate.value > best.value
    return False


def _current_best(eng) -> Card | None:
    if not eng.current_trick:
        return None
    best = eng.current_trick[0][1]
    for _, card in eng.current_trick[1:]:
        if _beats(card, best, eng.trump_suit):
            best = card
    return best


# Bazas esperadas por carta segun la cantidad de jugadores. Estimadas con una
# regresion lineal (bazas ganadas ~ cartas de cada tipo en la mano) sobre 400
# partidas por n de la heuristica contra si misma, recortadas a [0, 1]. Con mas
# rivales es mas probable que alguien falle o tenga una carta mas alta, asi que
# todo vale menos: con 2 jugadores hasta una J gana media baza; con 7, un K casi
# nada. Para mas de 7 jugadores se usa la fila de 7.
#        trump_high  trump_low    A     K     Q     J   otras
_BID_WEIGHTS = {
    2: (1.00, 0.95, 0.69, 0.58, 0.57, 0.50, 0.23),
    3: (1.00, 0.77, 0.56, 0.45, 0.37, 0.24, 0.06),
    4: (1.00, 0.64, 0.49, 0.31, 0.22, 0.16, 0.00),
    5: (0.99, 0.53, 0.38, 0.22, 0.15, 0.08, 0.00),
    6: (0.95, 0.47, 0.34, 0.14, 0.08, 0.03, 0.00),
    7: (0.92, 0.40, 0.27, 0.12, 0.05, 0.01, 0.00),
}
_HONOR_INDEX = {"A": 2, "K": 3, "Q": 4, "J": 5}


def _card_bid_weight(card: Card, trump, num_players: int) -> float:
    weights = _BID_WEIGHTS[min(max(num_players, 2), 7)]
    if card.suit == trump:
        return weights[0] if card.value >= 11 else weights[1]
    return weights[_HONOR_INDEX.get(card.rank, 6)]


def heuristic_opponent(rng: np.random.RandomState) -> ActFn:
    """Heuristica simple, la que sugiere el documento: cantar segun cartas altas,
    y jugar para ganar la baza si todavia falta cumplir el canto. El valor de
    cada carta al cantar baja con la cantidad de jugadores (ver _BID_WEIGHTS)."""
    def act(state, legal):
        eng = state._engine
        idx = eng.current_player_index
        trump = eng.trump_suit

        if eng.phase == "bidding":
            hand = eng.players[idx].hand
            est = sum(_card_bid_weight(c, trump, eng.num_players) for c in hand)
            target = int(round(est))
            bids = [a - 52 for a in legal]           # legal son ids 52+bid
            best_bid = min(bids, key=lambda b: (abs(b - target), b))
            return 52 + best_bid

        # fase de juego: legal son ids de carta
        cards = [(a, Card(RANKS[a % len(RANKS)], SUITS[a // len(RANKS)])) for a in legal]
        player = eng.players[idx]
        wants = player.tricks_won < (player.bid or 0)

        if not eng.current_trick:
            key = (lambda ac: ac[1].value)
            return (max(cards, key=key) if wants else min(cards, key=key))[0]

        best = _current_best(eng)
        winners = [(a, c) for a, c in cards if best is not None and _beats(c, best, trump)]
        if wants and winners:
            return min(winners, key=lambda ac: (ac[1].suit == trump, ac[1].value))[0]
        losers = [(a, c) for a, c in cards if (a, c) not in winners]
        pool = losers or cards
        return min(pool, key=lambda ac: (ac[1].suit == trump, ac[1].value))[0]

    return act


def network_opponent(networks, rng: np.random.RandomState) -> ActFn:
    """Juega con redes de politica (p.ej. snapshots de las avg-nets de NFSP):
    softmax sobre los logits de las acciones legales y muestrea, igual que
    NFSP en modo AVERAGE_POLICY. Sirve para enfrentar al agente contra
    versiones anteriores de si mismo. `networks` es una lista con una red por
    asiento (como los agentes de NFSP); el asiento j usa networks[j]."""
    import torch

    from .openspiel_game import info_state_tensor

    def act(state, legal):
        cur = state.current_player()
        info = info_state_tensor(state, cur)
        with torch.no_grad():
            logits = networks[cur](torch.as_tensor(info, dtype=torch.float32)[None])[0]
        probs = torch.softmax(logits[legal], dim=0).numpy().astype(np.float64)
        return int(rng.choice(legal, p=probs / probs.sum()))
    return act


# ----------------------------------------------------------------------
# Evaluacion
# ----------------------------------------------------------------------

def evaluate(
    game,
    act_fn: ActFn,
    opponent_fn: ActFn,
    num_players: int,
    num_games: int,
    rng: np.random.RandomState,
    seat: Optional[int] = None,
    parallel: bool = False,
    deal_seed: Optional[int] = None,
) -> Dict:
    """Juega `num_games` partidas con `act_fn` en `seat` y `opponent_fn` en el
    resto. Devuelve un dict anidado con metricas overall, por-h, por asiento y
    calibracion.

    Con `seat=None` (por defecto) el asiento evaluado rota partida a partida
    (0, 1, ..., n-1, 0, ...): el asiento importa (quien canta ultimo tiene la
    restriccion de suma, quien sale primero) y evaluar siempre en el mismo
    sesga las metricas. Ojo: con n jugadores, empatar con el rival da
    win_rate = 1/n, no 0.5.

    Con `parallel=True` juega todas las partidas a la vez: las funciones con un
    atributo `batch(states, legals) -> acciones` (las redes, ver
    rotten_rl/algorithms) deciden en lote; las demas, de a una. Mismas
    metricas, pero otro orden de sorteos que de a una.

    Con `deal_seed` la evaluacion es "duplicada", como en el bridge de torneo:
    se juegan num_games // n repartos y cada uno n veces, una con el evaluado en
    cada asiento. El reparto de cada partida sale de su propio generador
    (semilla `(deal_seed, reparto)`) y no depende de las jugadas, asi que dos
    agentes evaluados con la misma `deal_seed` juegan exactamente las mismas
    cartas. Se cancelan la suerte de las cartas y la del asiento: queda mucho
    menos ruido para comparar agentes, y si todos juegan igual (y sin azar) da
    exactamente el empate. `rng` no se usa para repartir.

    `overall` trae ademas el error estandar de win_rate y mean_score_diff
    (`*_se`); en modo duplicado, entre repartos (cada reparto es una muestra).
    `per_deal` trae win y diff promedio de cada reparto (de cada partida si no
    es duplicada), para comparar dos agentes reparto a reparto (`paired_diff`)."""
    n = num_players

    wins = 0.0
    wins_by_seat = [0.0] * n
    games_by_seat = [0] * n
    score_diff_sum = 0.0
    return_sum = 0.0
    norm_round_sum = 0.0
    norm_round_cnt = 0

    # acumuladores de nivel-ronda (overall y por h)
    def _acc():
        return {"n": 0, "exact": 0, "abs": 0.0, "signed": 0.0}
    overall = _acc()
    by_h: Dict[int, dict] = defaultdict(_acc)
    calib: Dict[int, list] = defaultdict(lambda: [0, 0])   # bid -> [sum_tricks, count]
    calib_by_h: Dict[tuple, list] = defaultdict(lambda: [0, 0])

    ladder = game.new_initial_state()._ladder  # escalera fija de esta config

    def round_closed(state, seat, prev_ptr):
        nonlocal norm_round_sum, norm_round_cnt
        h = ladder[prev_ptr]
        b = state._last_closed_bid[seat]
        t = state._last_closed_tricks[seat]
        e = abs(b - t)
        overall["n"] += 1
        overall["exact"] += int(e == 0)
        overall["abs"] += e
        overall["signed"] += (b - t)
        a = by_h[h]
        a["n"] += 1
        a["exact"] += int(e == 0)
        a["abs"] += e
        a["signed"] += (b - t)
        calib[b][0] += t
        calib[b][1] += 1
        calib_by_h[(h, b)][0] += t
        calib_by_h[(h, b)][1] += 1
        # puntaje de ronda normalizado por (10 + 2h)
        s_round = (10 + 2 * t) if t == b else t
        norm_round_sum += s_round / (10 + 2 * h)
        norm_round_cnt += 1

    def game_over(state, seat, g):
        nonlocal wins, score_diff_sum, return_sum
        # fin de partida: puntajes CRUDOS acumulados en el motor
        scores = [state._engine.players[i].score for i in range(n)]
        my = scores[seat]
        top = max(scores)
        if my == top:
            n_top = sum(1 for s in scores if s == top)
            wins += 1.0 / n_top                      # reparto justo de empates
            wins_by_seat[seat] += 1.0 / n_top
            game_win[g] = 1.0 / n_top
        others = [scores[i] for i in range(n) if i != seat]
        game_diff[g] = my - (sum(others) / len(others))
        score_diff_sum += game_diff[g]
        return_sum += state.returns()[seat]

    if deal_seed is not None:
        num_deals = max(1, num_games // n)
        num_games = num_deals * n
        groups = [g // n for g in range(num_games)]          # reparto de cada partida
        deal_rngs = [np.random.RandomState([deal_seed, d]) for d in groups]
        if seat is None:
            seats = [g % n for g in range(num_games)]
        else:
            seats = [seat] * num_games
    else:
        groups = list(range(num_games))
        deal_rngs = [rng] * num_games
        seats = [seat if seat is not None else g % n for g in range(num_games)]
    for s in seats:
        games_by_seat[s] += 1
    game_win = [0.0] * num_games
    game_diff = [0.0] * num_games

    def chance(state, g):
        outcomes, probs = zip(*state.chance_outcomes())
        state.apply_action(int(deal_rngs[g].choice(outcomes, p=probs)))

    if not parallel:
        for g in range(num_games):
            seat = seats[g]
            state = game.new_initial_state()
            prev_ptr = 0
            while not state.is_terminal():
                if state.is_chance_node():
                    chance(state, g)
                    continue
                cur = state.current_player()
                legal = state.legal_actions(cur)
                action = act_fn(state, legal) if cur == seat else opponent_fn(state, legal)
                state.apply_action(action)
                if state._round_ptr != prev_ptr:
                    round_closed(state, seat, prev_ptr)
                    prev_ptr = state._round_ptr
            game_over(state, seat, g)
    else:
        states = [game.new_initial_state() for _ in range(num_games)]
        prev_ptr = [0] * num_games
        alive = list(range(num_games))
        while alive:
            mine, theirs, still = [], [], []
            for g in alive:
                state = states[g]
                while state.is_chance_node():
                    chance(state, g)
                if state.is_terminal():
                    game_over(state, seats[g], g)
                    continue
                still.append(g)
                (mine if state.current_player() == seats[g] else theirs).append(g)
            alive = still
            for fn, games in ((act_fn, mine), (opponent_fn, theirs)):
                if not games:
                    continue
                legals = [states[g].legal_actions(states[g].current_player()) for g in games]
                if hasattr(fn, "batch"):
                    actions = fn.batch([states[g] for g in games], legals)
                else:
                    actions = [fn(states[g], legal) for g, legal in zip(games, legals)]
                for g, action in zip(games, actions):
                    state = states[g]
                    state.apply_action(int(action))
                    if state._round_ptr != prev_ptr[g]:
                        round_closed(state, seats[g], prev_ptr[g])
                        prev_ptr[g] = state._round_ptr

    def _pack(acc):
        m = max(acc["n"], 1)
        return {
            "exact_bid_rate": acc["exact"] / m,
            "mae_bid": acc["abs"] / m,
            "bid_bias": acc["signed"] / m,
            "rounds": acc["n"],
        }

    def _se(values):
        """Error estandar de la media, tomando como muestra cada grupo
        (cada reparto en modo duplicado, cada partida si no)."""
        sums = defaultdict(float)
        for grp, v in zip(groups, values):
            sums[grp] += v
        return _mean_se(np.array(list(sums.values())) / (num_games / len(sums)))[1]

    def _per_group(values):
        sums = defaultdict(float)
        for grp, v in zip(groups, values):
            sums[grp] += v
        return [sums[grp] / (num_games / len(sums)) for grp in sorted(sums)]

    result = {
        "overall": {
            "win_rate": wins / num_games,
            "win_rate_se": _se(game_win),
            "mean_score_diff": score_diff_sum / num_games,
            "mean_score_diff_se": _se(game_diff),
            "mean_return": return_sum / num_games,
            "norm_round_score": norm_round_sum / max(norm_round_cnt, 1),
            **_pack(overall),
        },
        "by_h": {h: _pack(acc) for h, acc in sorted(by_h.items())},
        "win_rate_by_seat": {
            s: wins_by_seat[s] / games_by_seat[s] for s in range(n) if games_by_seat[s] > 0
        },
        "calibration": {
            b: (s / c) for b, (s, c) in sorted(calib.items()) if c > 0
        },
        "calibration_by_h": {
            f"h{h}_b{b}": (s / c) for (h, b), (s, c) in sorted(calib_by_h.items()) if c > 0
        },
        "per_deal": {"win": _per_group(game_win), "diff": _per_group(game_diff)},
    }
    return result


def _mean_se(x):
    x = np.asarray(x, dtype=float)
    se = float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")
    return float(x.mean()), se


def paired_diff(a: Dict, b: Dict, key: str = "diff"):
    """(media, error estandar) de a - b reparto a reparto, para dos
    evaluaciones duplicadas con la misma deal_seed y cantidad de partidas (por
    ejemplo dos agentes contra la heuristica). La suerte de las cartas es la
    misma para los dos y se cancela en la resta: el error es mucho menor que
    comparar las dos medias por separado."""
    x, y = a["per_deal"][key], b["per_deal"][key]
    if len(x) != len(y):
        raise ValueError("Las evaluaciones tienen distinta cantidad de repartos.")
    return _mean_se(np.array(x) - np.array(y))
