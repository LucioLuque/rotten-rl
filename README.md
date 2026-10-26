# rotten-rl

Agentes de aprendizaje por refuerzo para **La Podrida** (*Oh Hell*), el juego de bazas con canto,
sobre [OpenSpiel](https://github.com/google-deepmind/open_spiel).

## Estructura

```
rotten_rl/              paquete
  engine.py             motor del juego en Python puro: reglas, cantos, bazas y puntaje
  openspiel_game.py     wrapper pyspiel.Game/State y observer (2 a 7 jugadores, con padding)
  evaluation.py         rivales de referencia (random, heurística, redes) y métricas
  training.py           loop de partidas para agentes de OpenSpiel (sin rl_environment);
                        opción de que cada ronda sea un episodio
  algorithms/           algoritmos con una interfaz común (base.py):
    nfsp.py             NFSP de OpenSpiel, un agente por asiento
    dqn.py              DQN con una red compartida por todos los asientos
    ppo.py              PPO con política y crítico compartidos (self-play)
  checkpoint.py         checkpoints completos (cualquier algoritmo) para cortar y retomar
  buffers.py            buffers de NFSP/DQN con el info-state comprimido (sin pérdida)
  run_logger.py         logging a TensorBoard, CSV y config.json en runs/
  renderer.py           render con Pygame
scripts/
  train.py              entrenamiento con self-play: --algo nfsp|dqn|ppo, --round_episodes,
                        --resume, --stop_at
  train_nfsp.py         alias de train.py (para retomar corridas viejas)
  run_queue.py          cola de experimentos para el laboratorio
  bench_parallel.py     cuántas corridas conviene correr a la vez
  summarize_runs.py     tabla comparativa de corridas
  diag_dqn.py           diagnóstico: DQN solo contra rivales fijos
  play.py               partida jugable contra bots
tests/                  tests con pytest
docs/plan.md            plan del proyecto, diseño de la observación, recompensa y métricas
docs/lab-s004.md        cómo correr en las máquinas del laboratorio S004 (VPN, SSH, horarios, velocidad)
docs/estado.md          estado actual: corridas, resultados y próximos pasos
docs/openspiel-upstream.md  bugs de OpenSpiel que encontramos y qué contribuir
docs/optimizacion.md    dónde se va el tiempo y la memoria, y qué conviene optimizar
docs/bibliografia.md    trabajos sobre juegos parecidos y qué tomar de cada uno
docs/reporte.md         reporte de la última noche de corridas
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
tensorboard --logdir runs
python scripts/play.py      # jugar contra bots
pytest
```
