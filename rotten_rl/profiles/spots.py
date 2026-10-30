"""
rotten_rl/profiles/spots.py  —  Learning from a person with chosen plays, not random ones.

Each decision is reduced to a *spot type*, at three levels of detail (`spot_keys`): the basics, what
matters about the hand, and also how the match is going and which cards came out. What the person
does is also stored in general terms (bid one more than the estimate, win with the lowest winning
card). So one answer applies to every similar hand.

`pick_spots` simulates matches and chooses what to ask: frequent spot types where the decision
changes the round result a lot, with several hands for the most important ones, without repeating
what was already answered. `LearnedProfile` plays with what it learned: the most repeated answer in
the most detailed spot type that has answers, and the heuristic if it knows nothing.
"""

import json
import os
from collections import Counter, defaultdict

import numpy as np

from ..evaluation import _beats, _card_bid_weight, _current_best, heuristic_opponent
from .match import replay, sample_chance

_BID = 52
_MAX_OFFSET = 2
_CLEAR_GAP = 15       # score gap to be clearly winning or losing
_FINAL_ROUNDS = 3     # last rounds of the match


def _bucket(x, top):
    return f"{top}+" if x >= top else str(x)


def _position(index, n):
    return "first" if index == 0 else ("last" if index == n - 1 else "middle")


def bid_estimate(eng, idx):
    return sum(_card_bid_weight(c, eng.trump_suit, eng.num_players) for c in eng.players[idx].hand)


def _match_context(state):
    """How the match is going for the player deciding: score against the best rival, and whether
    it is the end of the match."""
    eng = state._engine
    me = eng.players[eng.current_player_index]
    best = max(p.score for p in eng.players if p is not me)
    gap = me.score - best
    standing = "winning" if gap >= _CLEAR_GAP else ("losing" if gap <= -_CLEAR_GAP else "even")
    return standing, state._round_ptr >= len(state._ladder) - _FINAL_ROUNDS


def _cards_context(state):
    """What the seen cards say: trumps yet to appear, and whether I hold a card that can no longer
    be beaten in its suit."""
    eng = state._engine
    hand = eng.players[eng.current_player_index].hand
    seen = {eng.id_to_card(a) for a in state._played_by} | set(hand) | {eng.trump_card}
    unseen = [eng.id_to_card(a) for a in range(52) if eng.id_to_card(a) not in seen]
    trumps_out = sum(c.suit == eng.trump_suit for c in unseen)
    master = any(all(c.suit != u.suit or c.value > u.value for u in unseen) for c in hand)
    return _bucket(trumps_out, 3), master


def bid_key(state):
    """The three levels of a bid."""
    eng = state._engine
    idx, n, h = eng.current_player_index, eng.num_players, eng.cards_per_player
    hand = eng.players[idx].hand
    trumps = [c for c in hand if c.suit == eng.trump_suit]
    highs = sum(c.value >= 13 for c in hand if c.suit != eng.trump_suit)
    pos = (idx - eng.first_player_index) % n
    said = sum(p.bid or 0 for p in eng.players)
    expected = h * pos / n
    pressure = "low" if said < expected else ("high" if said > expected else "even")
    size = "short" if h <= 2 else ("medium" if h <= 4 else "long")
    general = ("bid", size, _position(pos, n), _bucket(len(trumps), 3))
    detailed = general + (_bucket(highs, 2), any(c.value >= 11 for c in trumps), pressure)
    return general, detailed, detailed + _match_context(state)


def play_key(state):
    """The three levels of a card play."""
    eng = state._engine
    idx, n = eng.current_player_index, eng.num_players
    me = eng.players[idx]
    trick = eng.current_trick
    missing = (me.bid or 0) - me.tricks_won
    goal = "needs_tricks" if missing > 0 else ("made_bid" if missing == 0 else "over_bid")
    rivals = sum(max(0, (p.bid or 0) - p.tricks_won) for i, p in enumerate(eng.players) if i != idx)
    has_trump = any(c.suit == eng.trump_suit for c in me.hand)
    if not trick:
        general = ("lead", goal)
        detailed = general + (rivals > 0, has_trump, _bucket(len(me.hand), 3))
        return general, detailed, detailed + _match_context(state) + _cards_context(state)
    lead = trick[0][1].suit
    follows = any(c.suit == lead for c in me.hand)
    best = _current_best(eng)
    can_win = any(_beats(c, best, eng.trump_suit) for c in eng.legal_cards(idx))
    general = ("follow", goal, can_win)
    detailed = general + (rivals > 0, follows, _position(len(trick), n), _bucket(len(me.hand), 3))
    return general, detailed, detailed + _match_context(state) + _cards_context(state)


def play_options(state, legal):
    """{general name of the play: card}. A card can have more than one name."""
    eng = state._engine
    trump = eng.trump_suit
    cards = [(a, eng.id_to_card(a)) for a in legal]
    power = lambda ac: (ac[1].suit == trump, ac[1].value)     # noqa: E731
    out = {}
    if not eng.current_trick:
        plain = sorted((ac for ac in cards if ac[1].suit != trump), key=power)
        trumps = sorted((ac for ac in cards if ac[1].suit == trump), key=power)
        if plain:
            out["lead_high"], out["lead_low"] = plain[-1][0], plain[0][0]
        if trumps:
            out["lead_trump_high"], out["lead_trump_low"] = trumps[-1][0], trumps[0][0]
        return out
    best = _current_best(eng)
    winners = sorted((ac for ac in cards if _beats(ac[1], best, trump)), key=power)
    losers = sorted((ac for ac in cards if not _beats(ac[1], best, trump)), key=power)
    if winners:
        out["win_low"], out["win_high"] = winners[0][0], winners[-1][0]
    if losers:
        out["lose_low"], out["lose_high"] = losers[0][0], losers[-1][0]
    return out


def answer_labels(state, action):
    """What an answer means in general: ["+1"] for a bid, play names for a card."""
    eng = state._engine
    if action >= _BID:
        est = int(round(bid_estimate(eng, eng.current_player_index)))
        return [f"{int(np.clip(action - _BID - est, -_MAX_OFFSET, _MAX_OFFSET)):+d}"]
    labels = [k for k, a in play_options(state, state.legal_actions()).items() if a == action]
    return labels or ["other"]                # a middle card


def spot_keys(state):
    return bid_key(state) if state._engine.phase == "bidding" else play_key(state)


def _round_score(state, player, heuristic):
    """Points of `player` if the round goes on with the heuristic."""
    round_ptr = state._round_ptr
    while state._round_ptr == round_ptr and not state.is_terminal() and not state.is_chance_node():
        state.apply_action(heuristic(state, state.legal_actions()))
    return state._engine.players[player].last_round_score


def _spread(state, heuristic):
    """How much the decision changes the round: gap between the best and the worst option."""
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
    """Chooses `count` decisions: the spot types that matter most (frequency and how much they
    change the result), going over the list again to add other hands of the most important ones.
    `skip`: histories already answered (see `answered`), to ask for more without repeating."""
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
            # Each pass only continues with the most important half of the previous one.
            for key in ranked[:max(1, len(ranked) >> turn)]:
                if len(found[key]) > turn and len(out) < quota:
                    out.append(found[key][turn][1])
            turn += 1
        return out

    # Points of a bid weigh more, but play has more types: fixed quotas.
    bids = choose("bid", int(count * 0.4))
    leads = choose("lead", int(count * 0.25))
    spots = bids + leads + choose("follow", count - len(bids) - len(leads))
    rng.shuffle(spots)
    return spots


def _saved_rows(folder):
    for fname in sorted(os.listdir(folder)) if os.path.isdir(folder) else []:
        with open(os.path.join(folder, fname)) as f:
            for line in f:
                row = json.loads(line)
                if row["type"] in ("decision", "spot"):
                    yield row


def answered(folder):
    """Histories of the decisions already saved for a person."""
    return [row["history"] for row in _saved_rows(folder)]


def load_answers(game, folder):
    """All the decisions of a person: [(general, detailed, full, labels)]."""
    answers = []
    for row in _saved_rows(folder):
        if len(row["legal"]) < 2:
            continue
        state = replay(game, row["history"])
        answers.append((*spot_keys(state), answer_labels(state, row["action"])))
    return answers


class LearnedProfile:
    """`act(state, legal)` from answers."""

    def __init__(self, answers, rng=None):
        self.heuristic = heuristic_opponent(rng or np.random.RandomState(0))
        self.counts = defaultdict(Counter)
        for *keys, labels in answers:
            for key in keys:
                self.counts[key].update(labels)

    def predict(self, state, legal):
        """The learned action, or None if it knows nothing about this spot type."""
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


def agreement(answers, make_profile):
    """How well saved answers are predicted, leaving out one fifth at a time."""
    hits = total = 0
    for fold in range(5):
        train = [a for i, a in enumerate(answers) if i % 5 != fold]
        profile = make_profile(train)
        for i, (*keys, labels) in enumerate(answers):
            if i % 5 != fold:
                continue
            guess = next((profile.counts[k] for k in reversed(keys) if profile.counts[k]), None)
            total += 1
            hits += bool(guess) and guess.most_common(1)[0][0] in labels
    return hits / max(1, total)
