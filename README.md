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
  exact/                      partes del juego resueltas sin RL (ver su README.md)
    round1.py                 la ronda de 1 carta: equilibrio (CFR) y comparaciones
    round2/                   la ronda de 2 cartas, en C++ (MCCFR)
  checkpoint.py               checkpoints completos para cortar y retomar
  buffers.py                  buffers de NFSP/DQN con el info-state comprimido (sin pérdida)
  run_logger.py               logging a TensorBoard, CSV y config.json en runs/
  renderer.py                 render con Pygame
scripts/
  train.py                    entrenamiento: --algo nfsp|dqn|ppo, --vector_games, --resume, ...
  run_queue.py                cola de experimentos para el laboratorio
  play.py                     partida jugable contra la heurística, un perfil o un modelo
  eval/                       evaluar agentes ya entrenados
    tournament.py             torneo entre agentes (políticas guardadas, heurística, azar)
    summarize_runs.py         tabla comparativa de corridas
  exact/
    solve_round1.py           resuelve la ronda de 1 carta de 2 a 7 jugadores
    analyze_round2.py         preguntas sobre la estrategia de la ronda de 2 cartas
  dev/                        herramientas de desarrollo
    bench_parallel.py         cuántas corridas conviene correr a la vez
    diag_dqn.py               diagnóstico: DQN solo contra rivales fijos
    make_obs_fixture.py       regenera tests/data/info_states.npz
tests/                        tests con pytest (tests/data: observaciones de referencia)
docs/                         documentación (índice en docs/README.md)
```

## Qué hay

- **Entrenamiento con RL** (`scripts/train.py`): agentes que aprenden jugando contra sí mismos
  (self-play). Hoy el mejor es PPO con una red compartida por todos los asientos; el estado y los
  resultados están en `docs/estado.md` y en los reportes de cada noche (`docs/reportes/`).
- **Evaluación** (`rotten_rl/evaluation.py`, `scripts/eval/`): contra el azar, una heurística y
  versiones anteriores, con repartos duplicados y error estándar, y un torneo entre agentes.
- **Partes resueltas sin RL** (`rotten_rl/exact/`): las partes del juego lo bastante chicas para
  calcular cómo conviene jugar de forma exacta: la ronda de 1 carta y la de 2 (en C++). Sirven para entender el
  juego y para controlar si el agente aprendió bien en un caso con respuesta conocida. Ver
  `rotten_rl/exact/README.md` y `docs/primera-ronda/`.

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
python scripts/play.py      # jugar; con --perfil guarda tus decisiones (docs/perfiles/)
# torneo entre agentes guardados (ver el docstring para el formato de --agents)
python scripts/eval/tournament.py --players 3 --deals 1000 --agents heuristic,runs/<corrida>
python scripts/exact/solve_round1.py      # la ronda de 1 carta resuelta (CPU, ~30–40 min)
pytest
```
