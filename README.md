# rotten-rl

Agentes de aprendizaje por refuerzo para **La Podrida** (*Oh Hell*), el juego de bazas con canto,
sobre [OpenSpiel](https://github.com/google-deepmind/open_spiel).

## Estructura

```
rotten_rl/              paquete
  engine.py             motor del juego en Python puro: reglas, cantos, bazas y puntaje
  openspiel_game.py     wrapper pyspiel.Game/State y observer (2 a 7 jugadores, con padding)
  evaluation.py         rivales de referencia (random, heurística, redes) y métricas
  run_logger.py         logging a TensorBoard, CSV y config.json en runs/
  renderer.py           render con Pygame
scripts/
  train_nfsp.py         entrenamiento NFSP con self-play
  play.py               partida jugable contra bots
tests/                  tests con pytest
docs/plan.md            plan del proyecto, diseño de la observación, recompensa y métricas
docs/lab-s004.md        cómo correr en las máquinas del laboratorio S004 (VPN, SSH, horarios, velocidad)
```

## Instalación

```bash
pip install -e ".[play,dev]"
```

Los paquetes de `open_spiel` en pip no incluyen `open_spiel/python/pytorch/nfsp.py`. Para entrenar
hace falta OpenSpiel compilado desde el código fuente, o copiar ese archivo del
[repo de OpenSpiel](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/pytorch/nfsp.py)
dentro de la instalación (la versión de `master` necesita Python 3.11 o más nuevo).

## Uso

```bash
python scripts/train_nfsp.py --players 2 --run_name baseline2p
tensorboard --logdir runs
python scripts/play.py      # jugar contra bots
pytest
```
