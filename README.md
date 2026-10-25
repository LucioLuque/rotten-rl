# rotten-rl

Agentes de aprendizaje por refuerzo para **La Podrida** (*Oh Hell*), el juego de bazas con canto,
sobre [OpenSpiel](https://github.com/google-deepmind/open_spiel).

## Estructura (`podrida/`)

| Archivo | Qué hace |
|---|---|
| `podrida.py` | Motor del juego en Python puro: reglas, cantos, bazas y puntaje |
| `podrida_openspiel.py` | Wrapper `pyspiel.Game`/`State` y observer (información de 2 a 7 jugadores con padding) |
| `evaluation.py` | Oponentes de referencia (random, heurística) y métricas de evaluación |
| `train_nfsp_podrida.py` | Entrenamiento NFSP con self-play |
| `run_logger.py` | Logging a TensorBoard, CSV y `config.json` en `runs/` |
| `demo_visual.py`, `visual_renderer.py` | Partida jugable con Pygame |
| `test_reglas.py` | Chequeo de la regla de seguir el palo |
| `RL podrida.pdf` | Plan del proyecto y diseño de la observación |

## Uso

```bash
pip install open_spiel torch dm-tree tensorboard pygame
cd podrida
python train_nfsp_podrida.py --players 2 --run_name baseline2p
tensorboard --logdir runs
python demo_visual.py      # jugar contra bots
```
