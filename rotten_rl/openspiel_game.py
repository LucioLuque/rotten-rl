"""
rotten_rl/openspiel_game.py

Wrapper de OpenSpiel (pyspiel) para el motor `rotten_rl.engine.PodridaGame`.

Idea general
------------
- Un *episodio* de OpenSpiel = una PARTIDA COMPLETA (la escalera 1..h..1).
  El motor `PodridaGame` sabe jugar UNA ronda; acá le agregamos la
  orquestación de partida (escalera, rotación de dealer, acumulación de
  puntaje, fin de partida) dentro del `State`.
- El reparto se modela con NODOS DE AZAR EXPLÍCITOS (EXPLICIT_STOCHASTIC):
  se reparte UNA carta por transición de azar, igual que `oh_hell`. El RNG
  lo maneja OpenSpiel, no el motor.
- Es un juego de información imperfecta: cada jugador ve su mano + lo público.
  El tensor de info-state (808) se arma en el Observer, desde la perspectiva
  del jugador, y NUNCA filtra las manos rivales.

Estado de este archivo
----------------------
Es un ESQUELETO FUNCIONAL y COMPLETO: carga, pasa `random_sim_test`, genera
playthroughs y produce el tensor de 808. Todos los bloques de observación de
la spec V1 están implementados. Los lugares pensados para que ajustes a gusto
están marcados con `# AJUSTE:`.

Uso
---
    import pyspiel
    import rotten_rl.openspiel_game                # registra el juego
    game = pyspiel.load_game("podrida", {"players": 4})
    state = game.new_initial_state()
    # ... loop estándar de OpenSpiel (ver __main__ abajo)
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pyspiel

from .engine import (
    RANKS,
    SUITS,
    BID_OFFSET,
    CARD_ACTIONS,
    MAX_HAND_SIZE,
    Card,
    PodridaGame,
    build_round_ladder,
)


# ----------------------------------------------------------------------
# Metadatos del juego
# ----------------------------------------------------------------------

_MAX_PLAYERS = 7
_NUM_DISTINCT_ACTIONS = CARD_ACTIONS + MAX_HAND_SIZE + 1   # 52 cartas + cantos 0..7 = 60
_ROUND_SCORE_NORM = 24.0                                   # 10 + 2*7, tu normalizador de reward

_GAME_TYPE = pyspiel.GameType(
    short_name="podrida",
    long_name="La Podrida (Oh Hell, variante con obligacion de fallar)",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    # Es general-sum de verdad (con reward "raw"). El surrogate "relative" lo
    # vuelve suma-cero, pero dejamos GENERAL_SUM porque es el superconjunto
    # seguro y PSRO en OpenSpiel lo maneja con meta-solvers (alpha-rank/prd).
    utility=pyspiel.GameType.Utility.GENERAL_SUM,
    # Damos reward por ronda (no solo al final): modelo REWARDS.
    reward_model=pyspiel.GameType.RewardModel.REWARDS,
    max_num_players=_MAX_PLAYERS,
    min_num_players=2,
    provides_information_state_string=True,
    provides_information_state_tensor=True,
    provides_observation_string=True,
    provides_observation_tensor=True,
    parameter_specification={
        "players": 4,
        "must_trump_when_void": True,
        "reward_scheme": "raw",     # "raw" -> score/24 ; "relative" -> (score - mean_opp)/24
        "max_hand": MAX_HAND_SIZE,
    },
)


# ----------------------------------------------------------------------
# Layout del tensor de info-state (spec V1 = 808)
# ----------------------------------------------------------------------
# (nombre, tamanio). El orden define la posicion en el tensor plano.
_INFO_PIECES = [
    ("hand_size", _MAX_PLAYERS),                 # one-hot de h (1..7)
    ("round_index", 13),                         # one-hot de la posicion en la escalera
    ("cards_remaining", _MAX_PLAYERS),           # cartas en mano de c/jugador, /h
    ("phase", 2),                                # one-hot bidding/playing
    ("trump", 5),                                # 4 palos + sin-triunfo
    ("lead_suit", 5),                            # 4 palos + nadie-jugo-aun
    ("active_players", _MAX_PLAYERS),            # mascara de posiciones reales
    ("dealer_relative", _MAX_PLAYERS),           # one-hot posicion relativa del dealer
    ("is_last_bidder", 1),                       # binario
    ("bids", _MAX_PLAYERS * 9),                  # por jugador: one-hot NOT_BID,0..7
    ("tricks_won", _MAX_PLAYERS),                # bazas ganadas /h
    ("void_suits", _MAX_PLAYERS * 4),            # por jugador y palo: 1 si sabemos que es void
    ("scores", _MAX_PLAYERS),                    # score acumulado /max teorico
    ("bid_minus_tricks", _MAX_PLAYERS),          # (bid - tricks)/h con signo
    ("card_info", 52 * 10),                      # por carta: one-hot de ubicacion
    ("trick_cards", 52),                         # flag: la carta esta en la baza en curso
    ("winning_relative", _MAX_PLAYERS),          # one-hot: quien va ganando la baza
    ("suits_played", 4),                         # cartas jugadas por palo /13
    ("seen_per_suit", 4 * 13),                   # one-hot: cuantas vistas por palo (0..12)
    ("players_left", _MAX_PLAYERS),              # one-hot: cuantos faltan tras este agente
]
_INFO_SIZE = sum(size for _, size in _INFO_PIECES)   # == 808
assert _INFO_SIZE == 808, f"El layout suma {_INFO_SIZE}, deberia ser 808"


def _card_id(card: Card) -> int:
    return SUITS.index(card.suit) * len(RANKS) + RANKS.index(card.rank)


# ----------------------------------------------------------------------
# Game
# ----------------------------------------------------------------------

class PodridaGame_OS(pyspiel.Game):
    """Version pyspiel de La Podrida. El GameInfo se arma por instancia
    porque depende de num_players (la escalera y las cotas de utilidad)."""

    def __init__(self, params=None):
        params = params or {}
        self._num_players = int(params.get("players", 4))
        self._must_trump = bool(params.get("must_trump_when_void", True))
        self._reward_scheme = str(params.get("reward_scheme", "raw"))
        self._max_hand = int(params.get("max_hand", MAX_HAND_SIZE))

        self._ladder = build_round_ladder(self._num_players, self._max_hand)

        # Cotas de utilidad (return de toda la partida) para reward "raw":
        #   por ronda, reward in [0, (10+2h)/24]; el minimo realista es 0.
        max_util = sum((10 + 2 * h) / _ROUND_SCORE_NORM for h in self._ladder)
        # Longitud maxima = nodos de DECISION (no de azar): n cantos + n*h jugadas por ronda.
        max_len = sum(self._num_players + self._num_players * h for h in self._ladder)

        game_info = pyspiel.GameInfo(
            num_distinct_actions=_NUM_DISTINCT_ACTIONS,
            max_chance_outcomes=52,
            num_players=self._num_players,
            min_utility=0.0,
            max_utility=float(max_util),
            utility_sum=None,          # general-sum
            max_game_length=int(max_len),
        )
        super().__init__(_GAME_TYPE, game_info, params)

    def new_initial_state(self):
        return PodridaState_OS(self)

    def make_py_observer(self, iig_obs_type=None, params=None):
        return PodridaObserver_OS(
            iig_obs_type or pyspiel.IIGObservationType(perfect_recall=True),
            params,
        )

    # accesores usados por State/Observer
    @property
    def ladder(self) -> List[int]:
        return self._ladder

    @property
    def n_players(self) -> int:
        return self._num_players


# ----------------------------------------------------------------------
# State
# ----------------------------------------------------------------------

class PodridaState_OS(pyspiel.State):
    """Estado de una PARTIDA completa. Internamente delega las reglas de la
    ronda al motor `PodridaGame` y agrega: reparto por azar, escalera y reward."""

    def __init__(self, game: PodridaGame_OS):
        super().__init__(game)
        self._n = game.n_players
        self._ladder = game.ladder
        self._reward_scheme = game._reward_scheme
        self._max_hand = game._max_hand

        # Motor de una ronda. El seed no se usa para repartir (lo hace el azar
        # de OpenSpiel); queda por compatibilidad del constructor.
        self._engine = PodridaGame(
            num_players=self._n,
            seed=0,
            must_trump_when_void=game._must_trump,
            max_cards_per_player=self._max_hand,
        )
        self._engine.dealer_index = 0

        self._round_ptr = 0
        self._macro = "dealing"            # "dealing" | "engine" | "terminal"
        self._returns = [0.0] * self._n
        self._last_rewards = [0.0] * self._n
        # bid/bazas del jugador en la ULTIMA ronda cerrada (para metricas de canto)
        self._last_closed_bid = [0] * self._n
        self._last_closed_tricks = [0] * self._n

        # cota teorica de puntaje de toda la partida (para normalizar 'scores')
        self._score_ceiling = float(sum(10 + 2 * h for h in self._ladder)) or 1.0

        self._begin_dealing(self._ladder[self._round_ptr])

    # ---------------- reparto por azar ----------------

    def _begin_dealing(self, hand_size: int) -> None:
        self._macro = "dealing"
        self._deal_h = hand_size
        self._undealt = list(range(52))
        # Slots a llenar: h cartas por jugador (0..n-1) y al final el triunfo (-1).
        slots: List[int] = []
        for p in range(self._n):
            slots.extend([p] * hand_size)
        slots.append(-1)                    # -1 marca la carta de triunfo
        self._deal_slots = slots
        self._deal_i = 0
        self._dealt_hands: List[List[Card]] = [[] for _ in range(self._n)]
        self._pending_trump: Optional[Card] = None

    def _finish_dealing(self) -> None:
        """Instala el reparto explicito en el motor (equivale a start_round pero
        sin barajar: las cartas ya las eligio el azar de OpenSpiel)."""
        eng = self._engine
        h = self._deal_h
        for p in eng.players:
            p.reset_for_round()
        eng.cards_per_player = h
        eng.played_cards = []
        eng.current_trick = []
        eng.completed_tricks = []
        eng.trick_number = 0
        eng.discard_pile = []
        eng.deck = []
        for p in range(self._n):
            hand = list(self._dealt_hands[p])
            hand.sort(key=lambda c: (SUITS.index(c.suit), RANKS.index(c.rank)))
            eng.players[p].hand = hand
        eng.trump_card = self._pending_trump
        eng.trump_suit = self._pending_trump.suit
        eng.discard_pile.append(self._pending_trump)
        eng.first_player_index = (eng.dealer_index + 1) % self._n
        eng.current_player_index = eng.first_player_index
        eng.phase = "bidding"
        self._macro = "engine"

    # ---------------- API pyspiel ----------------

    def current_player(self):
        if self._macro == "terminal":
            return pyspiel.PlayerId.TERMINAL
        if self._macro == "dealing":
            return pyspiel.PlayerId.CHANCE
        return self._engine.current_player_index

    def _legal_actions(self, player):
        # Solo se invoca en nodos de jugador.
        return sorted(self._engine.legal_action_ids(player))

    def chance_outcomes(self):
        assert self.is_chance_node()
        k = len(self._undealt)
        p = 1.0 / k
        return [(c, p) for c in sorted(self._undealt)]

    def _apply_action(self, action):
        if self._macro == "dealing":
            self._apply_deal(action)
            return
        self._apply_move(action)

    def _apply_deal(self, card_id: int) -> None:
        self._undealt.remove(card_id)
        slot = self._deal_slots[self._deal_i]
        card = Card(RANKS[card_id % len(RANKS)], SUITS[card_id // len(RANKS)])
        if slot == -1:
            self._pending_trump = card
        else:
            self._dealt_hands[slot].append(card)
        self._deal_i += 1
        if self._deal_i == len(self._deal_slots):
            self._finish_dealing()

    def _apply_move(self, action: int) -> None:
        eng = self._engine
        idx = eng.current_player_index
        eng.apply_action(action, idx)

        if eng.phase == "done":
            self._last_closed_bid = [p.bid or 0 for p in eng.players]
            self._last_closed_tricks = [p.tricks_won for p in eng.players]
            self._score_round_reward()
            # el motor ya rota el dealer al cerrar la ronda (dentro de play_card)
            self._round_ptr += 1
            if self._round_ptr >= len(self._ladder):
                self._macro = "terminal"
            else:
                self._begin_dealing(self._ladder[self._round_ptr])
        else:
            self._last_rewards = [0.0] * self._n

    def _score_round_reward(self) -> None:
        eng = self._engine
        raw = [p.last_round_score for p in eng.players]
        if self._reward_scheme == "relative":
            # surrogate suma-cero: cada uno contra la media del resto
            n = self._n
            total = sum(raw)
            rewards = [
                (raw[i] - (total - raw[i]) / (n - 1)) / _ROUND_SCORE_NORM
                if n > 1 else raw[i] / _ROUND_SCORE_NORM
                for i in range(n)
            ]
        else:  # "raw"
            rewards = [s / _ROUND_SCORE_NORM for s in raw]
        self._last_rewards = rewards
        for i in range(self._n):
            self._returns[i] += rewards[i]

    def _action_to_string(self, player, action):
        if player == pyspiel.PlayerId.CHANCE:
            c = Card(RANKS[action % len(RANKS)], SUITS[action // len(RANKS)])
            return f"Deal:{c.rank}{c.suit[0]}"
        if action >= BID_OFFSET:
            return f"Bid:{action - BID_OFFSET}"
        c = Card(RANKS[action % len(RANKS)], SUITS[action // len(RANKS)])
        return f"Play:{c.rank}{c.suit[0]}"

    def is_terminal(self):
        return self._macro == "terminal"

    def rewards(self):
        return list(self._last_rewards)

    def returns(self):
        return list(self._returns)

    def __str__(self):
        eng = self._engine
        if self._macro == "dealing":
            return (f"[Ronda {self._round_ptr + 1}/{len(self._ladder)} "
                    f"repartiendo {self._deal_i}/{len(self._deal_slots)}]")
        if self._macro == "terminal":
            return f"[FIN] returns={['%.2f' % r for r in self._returns]}"
        bids = [p.bid for p in eng.players]
        tricks = [p.tricks_won for p in eng.players]
        return (f"[R{self._round_ptr + 1} {eng.phase}] triunfo={eng.trump_suit} "
                f"turno=P{eng.current_player_index} bids={bids} bazas={tricks} "
                f"baza={[(i, f'{c.rank}{c.suit[0]}') for i, c in eng.current_trick]}")


# ----------------------------------------------------------------------
# Observer (produce el tensor de info-state de 808)
# ----------------------------------------------------------------------

class PodridaObserver_OS:
    """Arma el tensor de 808 desde la perspectiva de un jugador.
    Usa el mismo tensor para info-state y observation (para NFSP alcanza el
    info-state; separar la observation Markoviana queda como refinamiento)."""

    def __init__(self, iig_obs_type, params):
        if params:
            raise ValueError(f"Parametros de observacion no soportados: {params}")
        self.tensor = np.zeros(_INFO_SIZE, np.float32)
        self.dict = {}
        idx = 0
        for name, size in _INFO_PIECES:
            self.dict[name] = self.tensor[idx:idx + size]
            idx += size

    # -- helpers de perspectiva --
    @staticmethod
    def _rel(j: int, pl: int, n: int) -> int:
        """Posicion de j relativa a pl (0 = uno mismo)."""
        return (j - pl) % n

    def set_from(self, state: PodridaState_OS, player: int):
        self.tensor.fill(0.0)
        # El info-state solo esta bien definido en fase de juego del motor.
        if state._macro != "engine":
            return

        eng = state._engine
        n = state._n
        h = eng.cards_per_player or 1
        d = self.dict

        # 1) hand_size (one-hot de h)
        if 1 <= h <= _MAX_PLAYERS:
            d["hand_size"][h - 1] = 1.0

        # 2) round_index (one-hot en la escalera)
        if 0 <= state._round_ptr < 13:
            d["round_index"][state._round_ptr] = 1.0

        # 3) cards_remaining (por jugador relativo, /h)
        for j in range(n):
            d["cards_remaining"][self._rel(j, player, n)] = len(eng.players[j].hand) / h

        # 4) phase
        d["phase"][0 if eng.phase == "bidding" else 1] = 1.0

        # 5) trump (4 palos + sin triunfo)
        if eng.trump_suit is not None:
            d["trump"][SUITS.index(eng.trump_suit)] = 1.0
        else:
            d["trump"][4] = 1.0

        # 6) lead_suit (palo de salida de la baza en curso)
        lead = eng.lead_suit
        if lead is not None:
            d["lead_suit"][SUITS.index(lead)] = 1.0
        else:
            d["lead_suit"][4] = 1.0

        # 7) active_players (mascara: posiciones relativas reales)
        for j in range(n):
            d["active_players"][self._rel(j, player, n)] = 1.0

        # 8) dealer_relative
        d["dealer_relative"][self._rel(eng.dealer_index, player, n)] = 1.0

        # 9) is_last_bidder (el agente es el ultimo en cantar esta ronda)
        if eng.phase == "bidding":
            bids_made = sum(p.bid is not None for p in eng.players)
            if eng.current_player_index == player and bids_made == n - 1:
                d["is_last_bidder"][0] = 1.0

        # 10) bids (por jugador relativo: one-hot NOT_BID(0), 0..7 -> idx 1..8)
        for j in range(n):
            r = self._rel(j, player, n)
            b = eng.players[j].bid
            slot = 0 if b is None else (b + 1)
            if 0 <= slot < 9:
                d["bids"][r * 9 + slot] = 1.0

        # 11) tricks_won (/h)
        for j in range(n):
            d["tricks_won"][self._rel(j, player, n)] = eng.players[j].tricks_won / h

        # 12) void_suits (inferido del historial: si en una baza se salio con
        #     palo L y j no lo siguio, j es void en L)
        voids = self._infer_voids(eng, n)
        for j in range(n):
            r = self._rel(j, player, n)
            for s in range(4):
                if voids[j][s]:
                    d["void_suits"][r * 4 + s] = 1.0

        # 13) scores (/ techo teorico de la partida)
        for j in range(n):
            d["scores"][self._rel(j, player, n)] = eng.players[j].score / state._score_ceiling

        # 14) bid_minus_tricks ((bid - tricks)/h con signo)
        for j in range(n):
            b = eng.players[j].bid or 0
            d["bid_minus_tricks"][self._rel(j, player, n)] = (b - eng.players[j].tricks_won) / h

        # 15) card_info (52 x 10): no vista / en mi mano / es triunfo / la jugo Pj(rel 0..6)
        played_by = {}  # card_id -> jugador que la jugo (esta ronda)
        for trick in eng.completed_tricks:
            for j, c in trick:
                played_by[_card_id(c)] = j
        for j, c in eng.current_trick:
            played_by[_card_id(c)] = j
        my_hand_ids = {_card_id(c) for c in eng.players[player].hand}
        trump_id = _card_id(eng.trump_card) if eng.trump_card is not None else -1
        for cid in range(52):
            base = cid * 10
            if cid in my_hand_ids:
                d["card_info"][base + 1] = 1.0
            elif cid == trump_id:
                d["card_info"][base + 2] = 1.0
            elif cid in played_by:
                d["card_info"][base + 3 + self._rel(played_by[cid], player, n)] = 1.0
            else:
                d["card_info"][base + 0] = 1.0   # no la vi (incluye manos rivales)

        # 16) trick_cards (flag: carta pertenece a la baza en curso)
        for _, c in eng.current_trick:
            d["trick_cards"][_card_id(c)] = 1.0

        # 17) winning_relative (quien va ganando la baza en curso)
        if eng.current_trick:
            winner = eng.resolve_trick(list(eng.current_trick))
            d["winning_relative"][self._rel(winner, player, n)] = 1.0

        # 18) suits_played (jugadas por palo / 13) y 19) seen_per_suit
        played_per_suit = [0] * 4
        for cid in played_by:
            played_per_suit[cid // len(RANKS)] += 1
        for s in range(4):
            d["suits_played"][s] = played_per_suit[s] / 13.0

        # seen = jugadas + en mi mano + triunfo (cartas cuya identidad conozco)
        seen_per_suit = list(played_per_suit)
        for cid in my_hand_ids:
            seen_per_suit[cid // len(RANKS)] += 1
        if trump_id >= 0:
            seen_per_suit[trump_id // len(RANKS)] += 1
        for s in range(4):
            cnt = min(seen_per_suit[s], 12)
            d["seen_per_suit"][s * 13 + cnt] = 1.0

        # 20) players_left (cuantos faltan jugar en la baza despues de este agente)
        if eng.phase == "playing":
            already = len(eng.current_trick)
            left = n - 1 - already      # los que faltan tras el que mueve ahora
            left = max(0, min(left, _MAX_PLAYERS - 1))
            d["players_left"][left] = 1.0

    @staticmethod
    def _infer_voids(eng: PodridaGame, n: int):
        voids = [[False] * 4 for _ in range(n)]
        all_tricks = list(eng.completed_tricks)
        if eng.current_trick:
            all_tricks = all_tricks + [list(eng.current_trick)]
        for trick in all_tricks:
            if not trick:
                continue
            lead_suit = trick[0][1].suit
            lead_idx = SUITS.index(lead_suit)
            for j, c in trick:
                if c.suit != lead_suit:
                    voids[j][lead_idx] = True    # no siguio el palo => void en el palo
        return voids

    def string_from(self, state: PodridaState_OS, player: int) -> str:
        eng = state._engine
        if state._macro != "engine":
            return f"macro={state._macro} r={state._round_ptr}"
        hand = ",".join(f"{c.rank}{c.suit[0]}" for c in eng.players[player].hand)
        bids = [p.bid for p in eng.players]
        tricks = [p.tricks_won for p in eng.players]
        trick = [(i, f"{c.rank}{c.suit[0]}") for i, c in eng.current_trick]
        return (f"p{player}|r{state._round_ptr}|{eng.phase}|T{eng.trump_suit}|"
                f"hand[{hand}]|bids{bids}|won{tricks}|trick{trick}")


# ----------------------------------------------------------------------
# Registro
# ----------------------------------------------------------------------

pyspiel.register_game(_GAME_TYPE, PodridaGame_OS)


# ----------------------------------------------------------------------
# Smoke test manual
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import numpy as np

    game = pyspiel.load_game("podrida", {"players": 4})
    print("num_players:", game.num_players())
    print("num_distinct_actions:", game.num_distinct_actions())
    print("max_chance_outcomes:", game.max_chance_outcomes())
    print("info_state_tensor_size:", game.information_state_tensor_size())
    print("max_game_length:", game.max_game_length())

    rng = np.random.RandomState(0)
    state = game.new_initial_state()
    while not state.is_terminal():
        if state.is_chance_node():
            outcomes, probs = zip(*state.chance_outcomes())
            state.apply_action(rng.choice(outcomes, p=probs))
        else:
            pl = state.current_player()
            _ = state.information_state_tensor(pl)   # fuerza el armado del tensor
            state.apply_action(rng.choice(state.legal_actions()))
    print("returns:", ["%.3f" % r for r in state.returns()])
