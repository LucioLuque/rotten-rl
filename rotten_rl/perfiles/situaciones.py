"""
rotten_rl/perfiles/situaciones.py  —  Aprender de una persona con jugadas elegidas, no al azar.

Cada decisión se reduce a un *tipo de jugada*, en tres niveles de detalle (`spot_keys`): lo básico, lo
que importa de la mano, y además cómo va la partida y qué cartas salieron. Y lo que se hace también se
guarda en general (cantar uno más que la estimación, ganar con la carta más baja que gana). Así una
respuesta vale para todas las manos parecidas.

`pick_spots` simula partidas y elige qué preguntar: tipos frecuentes en los que la decisión cambia
mucho el resultado de la ronda, con varias manos para los más importantes, sin repetir las ya
respondidas. `LearnedProfile` juega con lo aprendido: la respuesta más repetida en el tipo más
detallado que tenga respuestas, y la heurística si no sabe nada.
"""

import json
import os
from collections import Counter, defaultdict

import numpy as np

from ..evaluation import _beats, _card_bid_weight, _current_best, heuristic_opponent
from .partida import replay, sample_chance

_BID = 52
_MAX_OFFSET = 2
_CLEAR_GAP = 15       # puntos de diferencia para ir claramente ganando o perdiendo
_FINAL_ROUNDS = 3     # últimas rondas de la partida


def _bucket(x, top):
    return f"{top}+" if x >= top else str(x)


def _position(index, n):
    return "primero" if index == 0 else ("ultimo" if index == n - 1 else "medio")


def bid_estimate(eng, idx):
    return sum(_card_bid_weight(c, eng.trump_suit, eng.num_players) for c in eng.players[idx].hand)


def _match_context(state):
    """Cómo va la partida para quien decide: puntos contra el mejor rival y si ya es el final."""
    eng = state._engine
    me = eng.players[eng.current_player_index]
    best = max(p.score for p in eng.players if p is not me)
    gap = me.score - best
    standing = "ganando" if gap >= _CLEAR_GAP else ("perdiendo" if gap <= -_CLEAR_GAP else "parejo")
    return standing, state._round_ptr >= len(state._ladder) - _FINAL_ROUNDS


def _cards_context(state):
    """Qué dicen las cartas vistas: triunfos que faltan aparecer y si tengo una carta que ya no
    se puede superar en su palo."""
    eng = state._engine
    hand = eng.players[eng.current_player_index].hand
    seen = {eng.id_to_card(a) for a in state._played_by} | set(hand) | {eng.trump_card}
    unseen = [eng.id_to_card(a) for a in range(52) if eng.id_to_card(a) not in seen]
    trumps_out = sum(c.suit == eng.trump_suit for c in unseen)
    master = any(all(c.suit != u.suit or c.value > u.value for u in unseen) for c in hand)
    return _bucket(trumps_out, 3), master


def bid_key(state):
    """Los tres niveles del canto."""
    eng = state._engine
    idx, n, h = eng.current_player_index, eng.num_players, eng.cards_per_player
    hand = eng.players[idx].hand
    trumps = [c for c in hand if c.suit == eng.trump_suit]
    highs = sum(c.value >= 13 for c in hand if c.suit != eng.trump_suit)
    pos = (idx - eng.first_player_index) % n
    said = sum(p.bid or 0 for p in eng.players)
    expected = h * pos / n
    pressure = "poco" if said < expected else ("mucho" if said > expected else "justo")
    size = "corta" if h <= 2 else ("media" if h <= 4 else "larga")
    general = ("canto", size, _position(pos, n), _bucket(len(trumps), 3))
    detailed = general + (_bucket(highs, 2), any(c.value >= 11 for c in trumps), pressure)
    return general, detailed, detailed + _match_context(state)


def play_key(state):
    """Los tres niveles de la jugada."""
    eng = state._engine
    idx, n = eng.current_player_index, eng.num_players
    me = eng.players[idx]
    trick = eng.current_trick
    missing = (me.bid or 0) - me.tricks_won
    goal = "le_faltan" if missing > 0 else ("cumplio" if missing == 0 else "se_paso")
    rivals = sum(max(0, (p.bid or 0) - p.tricks_won) for i, p in enumerate(eng.players) if i != idx)
    has_trump = any(c.suit == eng.trump_suit for c in me.hand)
    if not trick:
        general = ("sale", goal)
        detailed = general + (rivals > 0, has_trump, _bucket(len(me.hand), 3))
        return general, detailed, detailed + _match_context(state) + _cards_context(state)
    lead = trick[0][1].suit
    follows = any(c.suit == lead for c in me.hand)
    best = _current_best(eng)
    can_win = any(_beats(c, best, eng.trump_suit) for c in eng.legal_cards(idx))
    general = ("sigue", goal, can_win)
    detailed = general + (rivals > 0, follows, _position(len(trick), n), _bucket(len(me.hand), 3))
    return general, detailed, detailed + _match_context(state) + _cards_context(state)


def play_options(state, legal):
    """{nombre general de la jugada: carta}. Una carta puede tener más de un nombre."""
    eng = state._engine
    trump = eng.trump_suit
    cards = [(a, eng.id_to_card(a)) for a in legal]
    power = lambda ac: (ac[1].suit == trump, ac[1].value)     # noqa: E731
    out = {}
    if not eng.current_trick:
        plain = sorted((ac for ac in cards if ac[1].suit != trump), key=power)
        trumps = sorted((ac for ac in cards if ac[1].suit == trump), key=power)
        if plain:
            out["sale_alta"], out["sale_baja"] = plain[-1][0], plain[0][0]
        if trumps:
            out["sale_triunfo_alto"], out["sale_triunfo_bajo"] = trumps[-1][0], trumps[0][0]
        return out
    best = _current_best(eng)
    winners = sorted((ac for ac in cards if _beats(ac[1], best, trump)), key=power)
    losers = sorted((ac for ac in cards if not _beats(ac[1], best, trump)), key=power)
    if winners:
        out["gana_baja"], out["gana_alta"] = winners[0][0], winners[-1][0]
    if losers:
        out["pierde_baja"], out["pierde_alta"] = losers[0][0], losers[-1][0]
    return out


def answer_labels(state, action):
    """Qué significa una respuesta en general: [+1] en un canto, nombres de jugada en una carta."""
    eng = state._engine
    if action >= _BID:
        est = int(round(bid_estimate(eng, eng.current_player_index)))
        return [f"{int(np.clip(action - _BID - est, -_MAX_OFFSET, _MAX_OFFSET)):+d}"]
    labels = [k for k, a in play_options(state, state.legal_actions()).items() if a == action]
    return labels or ["otra"]                 # una carta intermedia


def spot_keys(state):
    return bid_key(state) if state._engine.phase == "bidding" else play_key(state)


def _round_score(state, player, heuristic):
    """Puntos de `player` si la ronda sigue con la heurística."""
    round_ptr = state._round_ptr
    while state._round_ptr == round_ptr and not state.is_terminal() and not state.is_chance_node():
        state.apply_action(heuristic(state, state.legal_actions()))
    return state._engine.players[player].last_round_score


def _spread(state, heuristic):
    """Cuánto cambia la ronda según la decisión: diferencia entre la mejor y la peor opción."""
    player = state.current_player()
    legal = state.legal_actions()
    if state._engine.phase == "bidding":
        options = legal
    else:
        options = sorted(set(play_options(state, legal).values()))
    scores = []
    for action in options:
        s = state.clone()
        s.apply_action(action)
        scores.append(_round_score(s, player, heuristic))
    return max(scores) - min(scores) if scores else 0


def pick_spots(game, count, seed=0, games=300, decisions_per_game=12, skip=()):
    """Elige `count` decisiones: los tipos de jugada que más importan (frecuencia y cuánto cambia
    el resultado), repasando la lista para sumar manos distintas de los más importantes. `skip`:
    historias ya respondidas (ver `answered`), para pedir más sin repetir."""
    skip = {tuple(h) for h in skip}
    rng = np.random.RandomState(seed)
    heuristic = heuristic_opponent(rng)
    found = defaultdict(list)
    for _ in range(games):
        state = game.new_initial_state()
        while not state.is_terminal():
            if state.is_chance_node():
                sample_chance(state, rng)
                continue
            legal = state.legal_actions()
            if (len(legal) > 1 and rng.uniform() < decisions_per_game / 62
                    and tuple(state.history()) not in skip):
                key = spot_keys(state)[2]
                spread = _spread(state, heuristic)
                if spread > 0:
                    found[key].append((spread, [int(a) for a in state.history()]))
            state.apply_action(heuristic(state, legal))
    def importance(key):
        return np.sqrt(len(found[key])) * np.mean([s for s, _ in found[key]])
    for key in found:
        found[key].sort(key=lambda sh: -sh[0])

    def choose(kind, quota):
        ranked = sorted((k for k in found if k[0] == kind), key=importance, reverse=True)
        out, turn = [], 0
        while len(out) < quota and any(len(found[k]) > turn for k in ranked):
            # Cada pasada solo sigue con la mitad más importante de la anterior.
            for key in ranked[:max(1, len(ranked) >> turn)]:
                if len(found[key]) > turn and len(out) < quota:
                    out.append(found[key][turn][1])
            turn += 1
        return out

    # Los puntos de un canto pesan más, pero la jugada tiene más tipos: cuotas fijas.
    bids = choose("canto", int(count * 0.4))
    leads = choose("sale", int(count * 0.25))
    spots = bids + leads + choose("sigue", count - len(bids) - len(leads))
    rng.shuffle(spots)
    return spots


def _saved_rows(folder):
    for fname in sorted(os.listdir(folder)) if os.path.isdir(folder) else []:
        with open(os.path.join(folder, fname)) as f:
            for line in f:
                row = json.loads(line)
                if row["tipo"] in ("decision", "situacion"):
                    yield row


def answered(folder):
    """Historias de las decisiones ya guardadas de una persona."""
    return [row["historia"] for row in _saved_rows(folder)]


def load_answers(game, folder):
    """Todas las decisiones de una persona: [(general, detallada, completa, etiquetas)]."""
    answers = []
    for row in _saved_rows(folder):
        if len(row["legales"]) < 2:
            continue
        state = replay(game, row["historia"])
        answers.append((*spot_keys(state), answer_labels(state, row["accion"])))
    return answers


class LearnedProfile:
    """`act(state, legal)` a partir de respuestas."""

    def __init__(self, answers, rng=None):
        self.heuristic = heuristic_opponent(rng or np.random.RandomState(0))
        self.counts = defaultdict(Counter)
        for *keys, labels in answers:
            for key in keys:
                self.counts[key].update(labels)

    def predict(self, state, legal):
        """La acción aprendida, o None si no sabe nada de este tipo de jugada."""
        eng = state._engine
        for key in reversed(spot_keys(state)):
            if not self.counts[key]:
                continue
            if eng.phase == "bidding":
                offset = int(self.counts[key].most_common(1)[0][0])
                target = int(round(bid_estimate(eng, eng.current_player_index))) + offset
                return min(legal, key=lambda a: (abs(a - _BID - target), a))
            options = play_options(state, legal)
            for label, _ in self.counts[key].most_common():
                if label in options:
                    return options[label]
        return None

    def __call__(self, state, legal):
        action = self.predict(state, legal)
        return self.heuristic(state, legal) if action is None else action


def agreement(answers_rows, make_profile):
    """Coincidencia con las respuestas guardadas, dejando afuera un quinto por vez."""
    hits = total = 0
    for fold in range(5):
        train = [a for i, a in enumerate(answers_rows) if i % 5 != fold]
        profile = make_profile(train)
        for i, (*keys, labels) in enumerate(answers_rows):
            if i % 5 != fold:
                continue
            guess = next((profile.counts[k] for k in reversed(keys) if profile.counts[k]), None)
            total += 1
            hits += bool(guess) and guess.most_common(1)[0][0] in labels
    return hits / max(1, total)
