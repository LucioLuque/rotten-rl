"""
rotten_rl/profiles/match.py  —  A match with a human player, with nothing on screen.

Runs on the OpenSpiel state, so the rival can be anything shaped like `act(state, legal)`: a network,
the heuristic or a profile. Deals come from `deal_seed` and the deal number, as in the duplicate
evaluation (`evaluation.evaluate`) and regardless of the plays: the same deal can later be given to a
model in the same seat and compared.

Each human decision is saved to a .jsonl with the full match history (`state.history()`), which is
enough to rebuild the state and get whatever observation is needed.
"""

import json
import os
import time

import numpy as np


def sample_chance(state, rng):
    outcomes, probs = zip(*state.chance_outcomes())
    state.apply_action(int(rng.choice(outcomes, p=probs)))


def next_deal(folder, deal_seed):
    """First deal of `deal_seed` not yet played in the sessions saved in `folder`."""
    if not os.path.isdir(folder):
        return 0
    played = [-1]
    for fname in os.listdir(folder):
        with open(os.path.join(folder, fname)) as f:
            for line in f:
                row = json.loads(line)
                if row["type"] == "match" and row["deal_seed"] == deal_seed:
                    played.append(row["deal"])
    return max(played) + 1


class SessionLog:
    """One file per session, written line by line: closing the game loses nothing."""

    def __init__(self, folder, profile, rival):
        os.makedirs(folder, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.path = os.path.join(folder, f"{stamp}.jsonl")
        self.profile, self.rival = profile, rival
        self.file = open(self.path, "a")

    def write(self, kind, **row):
        self.file.write(json.dumps({"type": kind, **row}, ensure_ascii=False) + "\n")
        self.file.flush()


class HumanMatch:
    """A whole match. The human moves with `apply`, the rival with `step_rival`. When a round
    closes it waits for `next_round`, so the result can be shown."""

    def __init__(self, game, rival, seat, deal_seed, deal, log=None):
        self.game, self.rival, self.seat = game, rival, seat
        self.deal_seed, self.deal = deal_seed, deal
        self.rng = np.random.RandomState([deal_seed, deal])
        self.log = log
        self.state = game.new_initial_state()
        if log:
            log.write("match", profile=log.profile, rival=log.rival, players=game.num_players(),
                      seat=seat, deal_seed=deal_seed, deal=deal)
        self.next_round()

    @property
    def engine(self):
        return self.state._engine

    @property
    def round_index(self):
        return self.state._round_ptr

    @property
    def finished(self):
        return self.state.is_terminal()

    @property
    def round_over(self):
        return self.finished or self.state.is_chance_node()

    @property
    def human_turn(self):
        return not self.round_over and self.state.current_player() == self.seat

    def next_round(self):
        while self.state.is_chance_node():
            sample_chance(self.state, self.rng)
        self.turn_started = time.time()

    def apply(self, action):
        """Applies the action of whoever has the turn. Returns False if it is not legal."""
        state = self.state
        legal = state.legal_actions()
        if action not in legal:
            return False
        if self.log and state.current_player() == self.seat:
            self.log.write("decision", deal=self.deal, round=state._round_ptr,
                           cards=self.engine.cards_per_player, phase=self.engine.phase,
                           history=[int(a) for a in state.history()],
                           legal=[int(a) for a in legal], action=int(action),
                           seconds=round(time.time() - self.turn_started, 2))
        state.apply_action(action)
        self.turn_started = time.time()
        if self.log and state.is_terminal():
            self.log.write("end", deal=self.deal, scores=[p.score for p in self.engine.players])
        return True

    def step_rival(self):
        state = self.state
        return self.apply(self.rival(state, state.legal_actions()))


class SpotSession:
    """A series of single chosen decisions, shaped like `HumanMatch`."""

    def __init__(self, game, spots, log=None):
        self.game, self.spots, self.log = game, spots, log
        self.index = -1
        self.finished = False
        self.round_over = False
        self.next_spot()

    def next_spot(self):
        self.index += 1
        if self.index >= len(self.spots):
            self.finished = True
            return
        self.state = replay(self.game, self.spots[self.index])
        self.seat = self.state.current_player()
        self.turn_started = time.time()

    @property
    def engine(self):
        return self.state._engine

    @property
    def human_turn(self):
        return not self.finished

    def apply(self, action):
        legal = self.state.legal_actions()
        if action not in legal:
            return False
        if self.log:
            self.log.write("spot", cards=self.engine.cards_per_player, phase=self.engine.phase,
                           history=self.spots[self.index], legal=[int(a) for a in legal],
                           action=int(action), seconds=round(time.time() - self.turn_started, 2))
        self.next_spot()
        return True


def replay(game, history):
    """Rebuilds the state saved with a decision."""
    state = game.new_initial_state()
    for action in history:
        state.apply_action(action)
    return state
