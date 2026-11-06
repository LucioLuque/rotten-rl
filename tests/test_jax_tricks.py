"""The JAX trick phase (rotten_rl/jax_engine/tricks.py) must match engine.py move by move."""

import random

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from rotten_rl.engine import SUITS, PodridaGame  # noqa: E402
from rotten_rl.jax_engine import tricks  # noqa: E402

step = jax.jit(tricks.step)
legal_mask = jax.jit(tricks.legal_mask)


def _jax_state(game):
    hands = np.zeros((game.num_players, 52), bool)
    for i, player in enumerate(game.players):
        for card in player.hand:
            hands[i, game.card_to_id(card)] = True
    return tricks.new_round(jnp.asarray(hands), SUITS.index(game.trump_suit),
                            game.current_player_index, game.cards_per_player)


@pytest.mark.parametrize("num_players", [2, 3, 7])
def test_matches_engine_move_by_move(num_players):
    rng = random.Random(num_players)
    for deal in range(40):
        game = PodridaGame(num_players=num_players, seed=1000 * num_players + deal)
        hand_size = rng.randint(1, 7)
        game.start_round(hand_size, dealer_index=rng.randrange(num_players))
        while game.phase == "bidding":
            game.apply_action(rng.choice(game.legal_action_ids()))
        state = _jax_state(game)
        while game.phase == "playing":
            assert int(state.current) == game.current_player_index
            engine_legal = sorted(game.legal_action_ids())
            jax_legal = sorted(np.flatnonzero(np.asarray(legal_mask(state))).tolist())
            assert jax_legal == engine_legal
            card = rng.choice(engine_legal)
            game.apply_action(card)
            state = step(state, jnp.int32(card))
        assert bool(tricks.is_done(state))
        assert np.asarray(state.tricks_won).tolist() == [p.tricks_won for p in game.players]
        assert not np.asarray(legal_mask(state)).any()


@pytest.mark.parametrize("num_players, hand_size", [(2, 7), (3, 4), (7, 7)])
def test_deal(num_players, hand_size):
    state = tricks.deal(jax.random.PRNGKey(num_players), num_players, hand_size, first_player=1)
    hands = np.asarray(state.hands)
    assert (hands.sum(axis=1) == hand_size).all()
    assert (hands.sum(axis=0) <= 1).all()                      # nobody shares a card
    assert int(state.current) == 1 and 0 <= int(state.trump) < 4


def test_random_rounds_deal_and_finish():
    won = tricks.play_random_rounds(jax.random.PRNGKey(0), 256, 2, 7)
    assert won.shape == (256, 2)
    assert (np.asarray(won).sum(axis=1) == 7).all()
