# rotten-rl

Agentes de aprendizaje por refuerzo para **La Podrida** (*Oh Hell*), el juego de bazas con canto,
sobre [OpenSpiel](https://github.com/google-deepmind/open_spiel).

## Estructura

```
rotten_rl/                    paquete
  engine.py                   motor del juego en Python puro: reglas, cantos, bazas y puntaje
  openspiel_game.py           wrapper pyspiel.Game/State y observación (2 a 7 jugadores)
  jax_engine/                 el juego y PPO en JAX, todo en la GPU (lo que se usa para entrenar;
                              ver su README.md): game.py, tricks.py, ppo.py (self-play,
                              explotador contra una red congelada, liga), evaluate.py
  evaluation.py               rivales de referencia (azar, heurística, redes) y métricas
  tournament_report.py        tablas y ranking de los torneos
  match_value/                probabilidad de ganar la partida según los puntajes (docs/match_value/)
  profiles/                   perfiles de juego (reglas, partidas guardadas) para medir contra estilos
  exact/                      partes del juego resueltas sin RL (ver su README.md)
    round1.py                 la ronda de 1 carta: equilibrio (CFR) y comparaciones
    round2/                   la ronda de 2 cartas, en C++ (MCCFR)
  algorithms/                 el entrenamiento viejo en PyTorch (base.py, nfsp.py, dqn.py, ppo.py)
  training.py, vector.py      loops del entrenamiento en PyTorch
  checkpoint.py, buffers.py   checkpoints y buffers del entrenamiento en PyTorch
  lab_hours.py                ventanas del laboratorio según el calendario del aula
  run_logger.py               logging a TensorBoard, CSV y config.json en runs/
  renderer.py                 render con Pygame
scripts/
  train.py                    entrenamiento en PyTorch: --algo nfsp|dqn|ppo (anterior al de JAX)
  run_queue.py                cola de experimentos para el laboratorio (con --script, cualquier script)
  play.py                     partida jugable contra la heurística, un perfil o un modelo
  jax_engine/
    train_ppo.py              entrenamiento en JAX: PPO self-play, --opponent (explotador), --league
    tournament.py             torneo entre agentes en la GPU (segundos)
    tournament_groups.py      el torneo resumido por corrida (promedio de semillas)
    bench_*.py, profile_ppo.py, lab_bench.sh   mediciones de velocidad
  lab/                        el laboratorio: watch_queue.sh (vigilante), backup_loop.sh y
                              pull_from_lab.sh (respaldo), start_queue.sh, setup_machine.sh,
                              thin_policies.sh (LAB_SSH=labs004b para la segunda máquina)
  eval/                       torneo y resumen de corridas en PyTorch
  exact/                      resolver la ronda de 1 carta, analizar la de 2 y comparar el agente
  match_value/fit_win_prob.py ajuste de la probabilidad de ganar
  profiles/screenshots.py     capturas de la interfaz sin ventana
  dev/                        herramientas de desarrollo
tests/                        tests con pytest (tests/data: observaciones de referencia)
docs/                         documentación (índice en docs/README.md)
```

## Qué hay

- **Entrenamiento en JAX** (`scripts/jax_engine/train_ppo.py`): PPO con una red compartida por todos
  los asientos, el juego entero en la GPU (~30 veces más rápido que el de PyTorch). Self-play, o
  contra una red congelada (`--opponent`) o una liga de redes (`--league`). Estado y resultados en
  `docs/estado.md`, plan en `docs/plan.md` y un reporte por noche en `docs/reportes/`.
- **Evaluación** (`rotten_rl/jax_engine/evaluate.py`, `scripts/jax_engine/tournament.py`): contra el
  azar, una heurística y otras redes, con repartos duplicados (cada reparto con el agente en cada
  asiento) y error estándar.
- **Partes resueltas sin RL** (`rotten_rl/exact/`): la ronda de 1 carta y la de 2 (en C++), para
  entender el juego y controlar si el agente aprendió bien en casos con respuesta conocida. Ver
  `rotten_rl/exact/README.md`, `docs/primera-ronda/` y `docs/segunda-ronda/`.

## Instalación

```bash
pip install -e ".[play,dev]"
```

Hace falta `open_spiel` 2.0 o más nuevo (Python 3.11+): las versiones 1.x de pip no incluyen
`open_spiel/python/pytorch/nfsp.py`.

## Uso

```bash
# entrenar en JAX (en el lab, con su entorno: ver docs/tecnico/lab-s004.md)
python scripts/jax_engine/train_ppo.py --players 2 --seeds 0,1,2,3 --games 12000000 \
    --rollout_games 512 --minibatch 1024 --epochs 2 --ent_coef 0.06 --run_name s20
# torneo entre corridas (ver el docstring para el formato de --agents)
python scripts/jax_engine/tournament.py --players 2 --agents heuristic,runs/s20_* --out torneo.md
python scripts/play.py      # jugar; con --profile guarda tus decisiones (docs/profiles/)
python scripts/exact/solve_round1.py      # la ronda de 1 carta resuelta (CPU, ~30–40 min)
pytest                      # los tests de JAX necesitan un procesador con AVX2 (el servidor no tiene)
```
