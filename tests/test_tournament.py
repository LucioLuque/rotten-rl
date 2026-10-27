"""scripts/tournament.py: carga politicas guardadas y, en evaluacion duplicada,
una red contra si misma da exactamente el empate."""
import importlib.util
import json
import os

import numpy as np
import pytest

pyspiel = pytest.importorskip("pyspiel")
torch = pytest.importorskip("torch")

from rotten_rl.algorithms.dqn import mlp  # noqa: E402
from rotten_rl.evaluation import evaluate  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "tournament", os.path.join(os.path.dirname(__file__), "..", "scripts", "tournament.py"))
tournament = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tournament)


def _corrida(tmp_path, game, nombre, eps):
    run = tmp_path / nombre
    (run / "checkpoints").mkdir(parents=True)
    (run / "config.json").write_text(json.dumps(
        {"algo": "ppo", "players": game.num_players(), "hidden_layers_sizes": ["16"]}))
    for ep in eps:
        torch.manual_seed(ep)
        net = mlp(game.information_state_tensor_size(), [16], game.num_distinct_actions())
        torch.save(net.state_dict(), run / "checkpoints" / f"policy_ep{ep}.pt")
    return str(run)


def test_agentes_y_empate_exacto(tmp_path):
    game = pyspiel.load_game("podrida", {"players": 3})
    run = _corrida(tmp_path, game, "x99_ppo_prueba", [1000, 2000])
    rng = np.random.RandomState(0)
    assert tournament.make_agent(run, game, "cpu", rng)[0] == "x99@2k"     # la ultima
    assert tournament.make_agent(run + "@1000", game, "cpu", rng)[0] == "x99@1k"
    assert tournament.make_agent(run + ":sample", game, "cpu", rng)[0] == "x99@2k:sample"
    with pytest.raises(ValueError):
        tournament.make_agent(run, pyspiel.load_game("podrida", {"players": 4}), "cpu", rng)

    _, a = tournament.make_agent(run, game, "cpu", rng)
    _, b = tournament.make_agent(run, game, "cpu", rng)
    _, viejo = tournament.make_agent(run + "@1000", game, "cpu", rng)
    o = evaluate(game, a, b, 3, 12, rng, parallel=True, deal_seed=1)["overall"]
    assert o["mean_score_diff"] == pytest.approx(0.0, abs=1e-9)
    assert o["win_rate"] == pytest.approx(1 / 3)
    o = evaluate(game, a, viejo, 3, 12, rng, parallel=True, deal_seed=1)["overall"]
    assert o["mean_score_diff"] != 0.0                       # redes distintas, juegan distinto
