# rotten-rl

Agentes de aprendizaje por refuerzo para **La Podrida** (*Oh Hell*), el juego de bazas con canto,
sobre [OpenSpiel](https://github.com/google-deepmind/open_spiel).

## Estructura

```
rotten_rl/                    paquete
  engine.py                   motor del juego en Python puro: reglas, cantos, bazas y puntaje
  openspiel_game.py           wrapper pyspiel.Game/State y observación (2 a 7 jugadores)
  evaluation.py               rivales de referencia (azar, heurística, redes) y métricas
  training.py                 loop de partidas de a una (opción: cada ronda es un episodio)
  vector.py                   muchas partidas a la vez, con la red decidiendo en lote
  algorithms/                 algoritmos con una interfaz común (base.py):
    nfsp.py                   NFSP de OpenSpiel, un agente por asiento
    dqn.py                    DQN con una red compartida por todos los asientos
    ppo.py                    PPO con política y crítico compartidos (self-play)
  exact/                      partes del juego resueltas sin RL
    round1.py                 la ronda de 1 carta: equilibrio (CFR) y comparaciones
  checkpoint.py               checkpoints completos para cortar y retomar
  buffers.py                  buffers de NFSP/DQN con el info-state comprimido (sin pérdida)
  run_logger.py               logging a TensorBoard, CSV y config.json en runs/
  renderer.py                 render con Pygame
scripts/
  train.py                    entrenamiento: --algo nfsp|dqn|ppo, --vector_games, --resume, ...
  run_queue.py                cola de experimentos para el laboratorio
  play.py                     partida jugable contra bots
  eval/                       evaluar agentes ya entrenados
    tournament.py             torneo entre agentes (políticas guardadas, heurística, azar)
    summarize_runs.py         tabla comparativa de corridas
  exact/
    solve_round1.py           resuelve la ronda de 1 carta de 2 a 7 jugadores
  dev/                        herramientas de desarrollo
    bench_parallel.py         cuántas corridas conviene correr a la vez
    diag_dqn.py               diagnóstico: DQN solo contra rivales fijos
    make_obs_fixture.py       regenera tests/data/info_states.npz
tests/                        tests con pytest (tests/data: observaciones de referencia)
docs/                         documentación (índice en docs/README.md)
```

## Instalación

```bash
pip install -e ".[play,dev]"
```

Hace falta `open_spiel` 2.0 o más nuevo (Python 3.11+): las versiones 1.x de pip no incluyen
`open_spiel/python/pytorch/nfsp.py`.

## Uso

```bash
python scripts/train.py --algo ppo --round_episodes --reward_scheme relative --players 2 --run_name ppo2p
# 32 partidas a la vez, decidiendo en lote en la GPU (DQN y PPO; mucho más rápido):
python scripts/train.py --algo ppo --round_episodes --reward_scheme relative --players 3 \
    --vector_games 32 --device cuda --run_name ppo3p
tensorboard --logdir runs
python scripts/play.py      # jugar contra bots
pytest
```
