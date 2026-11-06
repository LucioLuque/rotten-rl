"""The JAX evaluation (rotten_rl/jax_engine/evaluate.py): its heuristic must choose what
evaluation.heuristic_opponent chooses, and its metrics must agree with evaluation.evaluate."""

import random

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pyspiel = pytest.importorskip("pyspiel")
torch = pytest.importorskip("torch")
import jax.numpy as jnp  # noqa: E402

import rotten_rl.openspiel_game  # noqa: E402,F401
from rotten_rl import evaluation  # noqa: E402
from rotten_rl.jax_engine import evaluate as je  # noqa: E402
from rotten_rl.jax_engine import game as jg  # noqa: E402
from rotten_rl.jax_engine import ppo  # noqa: E402


def _play_pyspiel(num_players, seed, explore=0.3):
    """A match where every player follows the Python heuristic, except a random legal move with
    probability `explore` (to reach varied states). Returns the deal orders and, for every
    decision, the heuristic's choice and the move played."""
    rng = random.Random(seed)
    heuristic = evaluation.heuristic_opponent(np.random.RandomState(seed))
    game = pyspiel.load_game("podrida", {"players": num_players})
    state = game.new_initial_state()
    orders, dealt, moves = [], [], []
    while not state.is_terminal():
        if state.is_chance_node():
            card = rng.choice([c for c, _ in state.chance_outcomes()])
            dealt.append(card)
            state.apply_action(card)
            if not state.is_chance_node():
                orders.append(dealt + [c for c in range(52) if c not in dealt])
                dealt = []
            continue
        legal = state.legal_actions()
        choice = heuristic(state, legal)
        action = rng.choice(legal) if rng.random() < explore else choice
        moves.append((choice, action))
        state.apply_action(action)
    return orders, moves


@pytest.mark.parametrize("num_players", [2, 3, 5, 7])
def test_heuristic_matches_python(num_players):
    step = jax.jit(jg.step)
    act = jax.jit(je.heuristic_action)
    for seed in range(4):
        orders, moves = _play_pyspiel(num_players, seed)
        state = jg.new_game(np.array(orders), num_players)
        for i, (choice, action) in enumerate(moves):
            assert int(act(state)) == choice, f"seed {seed}, move {i}"
            state, _ = step(state, jnp.int32(action))


def test_metrics_agree_with_pyspiel_evaluation():
    """A small random policy against the heuristic: both evaluations within their errors."""
    n, games = 2, 600
    params = ppo.init_mlp(jax.random.PRNGKey(3), [jg.OBS_SIZE, 64, jg.NUM_ACTIONS])
    got = je.evaluate(params, "heuristic", n, games, deal_seed=7)

    from rotten_rl.algorithms.dqn import greedy_batch, mlp
    net = mlp(jg.OBS_SIZE, [64], jg.NUM_ACTIONS)
    net.load_state_dict(ppo.to_torch_state_dict(params))

    def act(state, legal):
        return int(act.batch([state], [legal])[0])
    act.batch = lambda states, legals: greedy_batch(net, states, legals, 60, torch.device("cpu"))
    game = pyspiel.load_game("podrida", {"players": n})
    rng = np.random.RandomState(0)
    ref = evaluation.evaluate(game, act, evaluation.heuristic_opponent(rng), n, games, rng,
                              parallel=True, deal_seed=7)
    assert set(got) == set(ref) and set(got["overall"]) == set(ref["overall"])
    for k in ("win_rate", "mean_score_diff"):
        se = np.hypot(got["overall"][f"{k}_se"], ref["overall"][f"{k}_se"])
        assert abs(got["overall"][k] - ref["overall"][k]) < 4 * se, k
    assert abs(got["overall"]["exact_bid_rate"] - ref["overall"]["exact_bid_rate"]) < 0.05
    assert got["overall"]["rounds"] == ref["overall"]["rounds"] == games * 13


def test_duplicate_deals_cancel_out():
    """The heuristic against itself... as the evaluated policy must be a network, use "past" with
    the same network: every seat plays the same, so each deal is an exact tie."""
    n = 3
    params = ppo.init_mlp(jax.random.PRNGKey(5), [jg.OBS_SIZE, 32, jg.NUM_ACTIONS])
    m = je.evaluate(params, "past", n, 90, opponent_params=params)
    assert np.allclose(m["per_deal"]["diff"], 0.0)
    assert m["overall"]["win_rate"] == pytest.approx(1 / n)
