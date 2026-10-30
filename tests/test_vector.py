"""Muchas partidas a la vez (VectorRunner): ve lo mismo que EpisodeRunner, y
PPO y DQN aprenden y se retoman igual en lote."""
import numpy as np
import pytest

pyspiel = pytest.importorskip("pyspiel")
torch = pytest.importorskip("torch")

from open_spiel.python import rl_agent  # noqa: E402

import rotten_rl.openspiel_game  # noqa: E402,F401
from rotten_rl.algorithms import SharedDQN, SharedPPO  # noqa: E402
from rotten_rl.checkpoint import latest_checkpoint, load_checkpoint, save_checkpoint  # noqa: E402
from rotten_rl.training import EpisodeRunner  # noqa: E402
from rotten_rl.vector import VectorRunner  # noqa: E402


def _elegir(k, legal):
    return legal[(k * 7) % len(legal)]


class _Grabador(rl_agent.AbstractAgent):
    """Agente determinista que anota (es_LAST, info, legales, reward) de cada paso."""

    def __init__(self, player_id):
        self.player_id = player_id
        self.visto = []
        self._k = 0

    def step(self, time_step, is_evaluation=False):
        p = self.player_id
        last = time_step.last()
        self.visto.append((
            last,
            None if last else np.asarray(time_step.observations["info_state"][p]).copy(),
            [] if last else list(time_step.observations["legal_actions"][p]),
            time_step.rewards[p]))
        if last:
            return None
        self._k += 1
        return rl_agent.StepOutput(
            action=_elegir(self._k, time_step.observations["legal_actions"][p]), probs=None)


class _GrabadorLote:
    """Lo mismo, con la interfaz de VectorRunner: un grabador por stream."""

    def __init__(self):
        self.visto = {}
        self._k = {}

    def batch_step(self, streams, infos, masks, legals, rewards):
        acciones = []
        for s, info, mask, legal, r in zip(streams, infos, masks, legals, rewards):
            assert list(np.flatnonzero(mask)) == sorted(legal)
            self.visto.setdefault(s, []).append((False, info.copy(), list(legal), r))
            self._k[s] = self._k.get(s, 0) + 1
            acciones.append(_elegir(self._k[s], legal))
        return np.array(acciones)

    def batch_end(self, streams, rewards):
        for s, r in zip(streams, rewards):
            self.visto.setdefault(s, []).append((True, None, [], r))


@pytest.mark.parametrize("players,rondas", [(2, False), (3, True), (4, False)])
def test_una_partida_igual_que_episode_runner(players, rondas):
    game = pyspiel.load_game("podrida", {"players": players})
    agentes = [_Grabador(i) for i in range(players)]
    EpisodeRunner(game, np.random.RandomState(5)).play(agentes, round_episodes=rondas)

    lote = _GrabadorLote()
    VectorRunner(game, np.random.RandomState(5)).play(lote, 1, round_episodes=rondas)

    for p, a in enumerate(agentes):
        assert len(lote.visto[p]) == len(a.visto)
        for (last1, info1, legal1, r1), (last2, info2, legal2, r2) in zip(a.visto, lote.visto[p]):
            assert last1 == last2 and legal1 == legal2 and r1 == pytest.approx(r2)
            if not last1:
                np.testing.assert_array_equal(info1, info2)


@pytest.mark.parametrize("rondas", [False, True])
def test_muchas_partidas_entregan_todo_el_reward(rondas):
    players, partidas = 3, 5
    game = pyspiel.load_game("podrida", {"players": players})
    lote = _GrabadorLote()
    seat0 = VectorRunner(game, np.random.RandomState(0)).play(lote, partidas,
                                                              round_episodes=rondas)
    assert len(seat0) == partidas
    for g in range(partidas):
        streams = [lote.visto[g * players + p] for p in range(players)]
        # A cada asiento le llega todo su reward, y el del asiento 0 es el que se devuelve.
        assert sum(r for _, _, _, r in streams[0]) == pytest.approx(seat0[g])
        for st in streams:
            assert st[-1][0]                       # cada stream termina con un LAST
            lasts = sum(last for last, *_ in st)
            assert lasts == (len(game.ladder) if rondas else 1)


def _entrenar_lote(game, algo, rng, lotes, k, start=0):
    runner = VectorRunner(game, rng)
    out, ep = [], start
    for _ in range(lotes):
        algo.begin_batch(ep + 1, k)
        out.append(runner.play(algo, k, round_episodes=True))
        algo.end_batch(k)
        ep += k
    return out


def _ppo(game, **kw):
    torch.manual_seed(0)
    args = dict(hidden_layers_sizes=[32], learning_rate=1e-3, discount_factor=1.0,
                rollout_episodes=4, epochs=2, minibatch=64, vector_games=4)
    args.update(kw)
    return SharedPPO(game, **args)


def _dqn(game, **kw):
    torch.manual_seed(0)
    args = dict(hidden_layers_sizes=[32], replay_buffer_capacity=2000,
                min_buffer_size_to_learn=50, batch_size=16, learn_every=4, learning_rate=1e-3,
                discount_factor=1.0, epsilon_start=0.5, epsilon_end=0.05,
                epsilon_decay_episodes=20, target_update_every=50, vector_games=4)
    args.update(kw)
    return SharedDQN(game, **args)


def _params(net):
    return torch.cat([p.detach().flatten() for p in net.parameters()])


def test_ppo_en_lote_aprende():
    game = pyspiel.load_game("podrida", {"players": 3})
    np.random.seed(0)
    algo = _ppo(game)
    antes = _params(algo.policy).clone()
    _entrenar_lote(game, algo, np.random.RandomState(0), 3, 4)
    assert algo.updates == 3                       # un update por lote de 4 partidas
    assert np.isfinite(algo.last_stats["loss_policy"])
    assert not torch.equal(antes, _params(algo.policy))


def test_dqn_en_lote_aprende():
    game = pyspiel.load_game("podrida", {"players": 3})
    np.random.seed(0)
    algo = _dqn(game)
    antes = _params(algo.q).clone()
    _entrenar_lote(game, algo, np.random.RandomState(0), 2, 4)
    # Cada decision deja una transicion: tantas como pasos de los 3 asientos en 8 partidas.
    assert len(algo.buffer) == algo.steps
    assert np.isfinite(algo.losses()["loss_dqn"])
    assert not torch.equal(antes, _params(algo.q))


@pytest.mark.parametrize("hacer", [_ppo, _dqn])
def test_en_lote_retomar_es_igual_que_de_corrido(tmp_path, hacer):
    game = pyspiel.load_game("podrida", {"players": 2})
    np.random.seed(1)
    torch.manual_seed(1)
    rng = np.random.RandomState(1)
    corrido = hacer(game, rollout_episodes=8) if hacer is _ppo else hacer(game)
    _entrenar_lote(game, corrido, rng, 1, 4)       # PPO: corta con un lote a medio juntar
    save_checkpoint(str(tmp_path), 4, corrido, rng)
    ret_corrido = _entrenar_lote(game, corrido, rng, 2, 4, start=4)

    np.random.seed(99)
    torch.manual_seed(99)
    rng2 = np.random.RandomState(99)
    retomado = hacer(game, rollout_episodes=8) if hacer is _ppo else hacer(game)
    info = load_checkpoint(latest_checkpoint(str(tmp_path)), retomado, rng2)
    ret_retomado = _entrenar_lote(game, retomado, rng2, 2, 4, start=info["episode"])

    assert ret_retomado == ret_corrido
    net_a, net_b = (corrido.policy, retomado.policy) if hacer is _ppo else (corrido.q, retomado.q)
    assert torch.equal(_params(net_a), _params(net_b))


def test_buffer_en_lote_igual_que_de_a_una():
    """append_batch deja el buffer igual que k appends, y la descompresion en
    torch da el mismo info-state que la de numpy."""
    from open_spiel.python.pytorch import dqn as os_dqn
    from rotten_rl.buffers import (CompactReplayBuffer, decode_info_states,
                                   decode_info_states_torch, pack_rows, unpack_rows)
    game = pyspiel.load_game("podrida", {"players": 3})
    lote = _GrabadorLote()
    VectorRunner(game, np.random.RandomState(0)).play(lote, 2)
    infos = np.stack([info for st in lote.visto.values() for last, info, *_ in st if not last])
    k = len(infos) - 1
    rng = np.random.RandomState(0)
    acciones, rewards = rng.randint(60, size=k), rng.randn(k).astype(np.float32)
    finales, mascaras = rng.rand(k) < 0.2, rng.rand(k, 60) < 0.3

    def trans(i):
        return os_dqn.Transition(infos[i], np.asarray(acciones[i]), np.asarray(rewards[i]),
                                 infos[i + 1], np.asarray(finales[i]), mascaras[i])
    uno = CompactReplayBuffer.init(50, trans(0))
    varios = CompactReplayBuffer.init(50, trans(0))
    for i in range(k):
        uno.append(trans(i))
    for start in range(0, k, 7):
        sl = slice(start, min(k, start + 7))
        varios.append_batch(infos[sl], acciones[sl], rewards[sl], infos[1:][sl], finales[sl],
                            mascaras[sl])
    assert int(uno.entry_index) == int(varios.entry_index) == k
    for a, b in zip(uno.experience, varios.experience):
        np.testing.assert_array_equal(a, b)
    # El buffer da la vuelta (capacidad 50): lo de numpy y lo de torch coinciden.
    e = varios.experience
    dec = decode_info_states_torch(torch.as_tensor(e.info_state))
    np.testing.assert_array_equal(dec.numpy(), decode_info_states(e.info_state))
    # Las filas del espejo en GPU del DQN devuelven exactamente la transicion.
    s, a, r, s2, fin, mask = unpack_rows(torch.as_tensor(pack_rows(e)))
    np.testing.assert_array_equal(s.numpy(), decode_info_states(e.info_state))
    np.testing.assert_array_equal(a.numpy(), e.action)
    np.testing.assert_array_equal(r.numpy(), e.reward)
    np.testing.assert_array_equal(s2.numpy(), decode_info_states(e.next_info_state))
    np.testing.assert_array_equal(fin.numpy(), e.is_final_step)
    np.testing.assert_array_equal(mask.numpy(), e.legal_actions_mask)


@pytest.mark.parametrize("hacer", [_ppo, _dqn])
def test_evaluar_en_paralelo(hacer):
    """La decision en lote de la red es la misma que de a una, y evaluar en
    paralelo una partida da las mismas metricas que de a una."""
    from rotten_rl.evaluation import evaluate, heuristic_opponent
    game = pyspiel.load_game("podrida", {"players": 3})
    algo = hacer(game)
    act = algo.eval_act()

    estados = []
    state = game.new_initial_state()
    rng = np.random.RandomState(0)
    while not state.is_terminal():
        if state.is_chance_node():
            outcomes, probs = zip(*state.chance_outcomes())
            state.apply_action(int(rng.choice(outcomes, p=probs)))
            continue
        estados.append(state.clone())
        state.apply_action(state.legal_actions()[0])
    legals = [s.legal_actions(s.current_player()) for s in estados]
    assert list(act.batch(estados, legals)) == [act(s, l) for s, l in zip(estados, legals)]

    for seat in range(3):
        de_a_una = evaluate(game, act, heuristic_opponent(np.random.RandomState(1)), 3, 1,
                            np.random.RandomState(2), seat=seat)
        paralelo = evaluate(game, act, heuristic_opponent(np.random.RandomState(1)), 3, 1,
                            np.random.RandomState(2), seat=seat, parallel=True)
        for m in (de_a_una, paralelo):             # con una sola partida no hay error estandar
            assert np.isnan(m["overall"].pop("win_rate_se"))
            assert np.isnan(m["overall"].pop("mean_score_diff_se"))
        assert paralelo == de_a_una


def test_ppo_baja_la_tasa_de_aprendizaje():
    """--ppo_anneal_lr: la tasa baja linealmente con las partidas jugadas, en
    lote y de a una, y llega a 0 al final."""
    game = pyspiel.load_game("podrida", {"players": 3})
    np.random.seed(0)
    algo = _ppo(game, anneal_episodes=12)
    lrs = []
    runner = VectorRunner(game, np.random.RandomState(0))
    for ep in (0, 4, 8):
        algo.begin_batch(ep + 1, 4)
        runner.play(algo, 4, round_episodes=True)
        algo.end_batch(4)
        lrs.append(algo.optimizer.param_groups[0]["lr"])
    assert lrs == pytest.approx([1e-3 * 8 / 12, 1e-3 * 4 / 12, 0.0])

    de_a_una = _ppo(game, anneal_episodes=12, vector_games=1)
    seats = de_a_una.seats()
    runner1 = EpisodeRunner(game, np.random.RandomState(0))
    for ep in range(1, 5):
        de_a_una.begin_episode(ep)
        runner1.play(seats, round_episodes=True)
    de_a_una.begin_episode(5)                      # actualiza con las 4 partidas jugadas
    assert de_a_una.updates == 1
    assert de_a_una.optimizer.param_groups[0]["lr"] == pytest.approx(1e-3 * 8 / 12)


def test_ppo_inicializacion_ortogonal():
    """--ppo_init orthogonal: arranca con una politica casi uniforme."""
    game = pyspiel.load_game("podrida", {"players": 3})
    algo = _ppo(game, init="orthogonal")
    x = torch.rand(64, game.information_state_tensor_size())
    with torch.no_grad():
        probs = torch.softmax(algo.policy(x), dim=1)
        assert torch.allclose(probs, torch.full_like(probs, 1 / algo.num_actions), atol=1e-3)
        normal = torch.softmax(_ppo(game).policy(x), dim=1)
    assert not torch.allclose(normal, torch.full_like(normal, 1 / algo.num_actions), atol=1e-3)
    with pytest.raises(ValueError):
        _ppo(game, init="otra")


def test_manos_ocultas_son_las_de_los_demas():
    """hidden_hands_tensor: por posicion relativa, exactamente la mano de ese
    jugador (y nada de la propia)."""
    from rotten_rl.openspiel_game import HIDDEN_HANDS_SIZE, _card_id, hidden_hands_tensor
    game = pyspiel.load_game("podrida", {"players": 4})
    state = game.new_initial_state()
    rng = np.random.RandomState(3)
    vistos = 0
    while not state.is_terminal():
        if state.is_chance_node():
            outcomes, probs = zip(*state.chance_outcomes())
            state.apply_action(int(rng.choice(outcomes, p=probs)))
            continue
        p = state.current_player()
        t = hidden_hands_tensor(state, p)
        assert t.shape == (HIDDEN_HANDS_SIZE,)
        hands = [p_.hand for p_ in state._engine.players]
        for r in range(1, 4):
            esperado = sorted(_card_id(c) for c in hands[(p + r) % 4])
            assert list(np.flatnonzero(t[(r - 1) * 52:r * 52])) == esperado
        assert not t[3 * 52:].any()                # posiciones que no existen con 4
        vistos += 1
        state.apply_action(int(rng.choice(state.legal_actions())))
    assert vistos > 100


def test_ppo_critico_central_aprende_y_se_retoma(tmp_path):
    game = pyspiel.load_game("podrida", {"players": 3})
    np.random.seed(0)
    algo = _ppo(game, critic="central")
    assert algo.value[0].in_features == game.information_state_tensor_size() + 6 * 52
    antes = _params(algo.value).clone()
    _entrenar_lote(game, algo, np.random.RandomState(0), 2, 4)
    assert algo.updates == 2
    assert np.isfinite(algo.last_stats["loss_value"])
    assert not torch.equal(antes, _params(algo.value))
    with pytest.raises(ValueError):
        _ppo(game, critic="central", vector_games=1)

    np.random.seed(1)
    torch.manual_seed(1)
    rng = np.random.RandomState(1)
    corrido = _ppo(game, critic="central", rollout_episodes=8)
    _entrenar_lote(game, corrido, rng, 1, 4)
    save_checkpoint(str(tmp_path), 4, corrido, rng)
    ret_corrido = _entrenar_lote(game, corrido, rng, 2, 4, start=4)
    np.random.seed(99)
    torch.manual_seed(99)
    rng2 = np.random.RandomState(99)
    retomado = _ppo(game, critic="central", rollout_episodes=8)
    info = load_checkpoint(latest_checkpoint(str(tmp_path)), retomado, rng2)
    assert _entrenar_lote(game, retomado, rng2, 2, 4, start=info["episode"]) == ret_corrido
    assert torch.equal(_params(corrido.value), _params(retomado.value))
