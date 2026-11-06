# rotten_rl/jax_engine

La Podrida escrita en JAX: el estado de miles de partidas en arreglos, sin código de Python por
jugada, compilado con `jax.jit` y corriendo en paralelo con `jax.vmap`. El objetivo es correr el
juego y PPO enteros en la GPU. Plan y motivo en `docs/tecnico/optimizacion.md` ("Elegido para hacer")
y avance en `docs/jax_engine/README.md`.

- `game.py`: la partida completa (etapa 2): escalera, canto, bazas, puntaje, rewards y la
  observación de 808, igual al juego "podrida" de pyspiel (`tests/test_jax_game.py`).
- `ppo.py`: PPO entero en JAX (etapa 3), misma receta que las corridas actuales; entrena con
  `scripts/jax_engine/train_ppo.py` y guarda redes que cargan los torneos y `play.py`.
- `tricks.py`: fase de bazas sola (etapa 1). Mismas reglas que `rotten_rl/engine.py`: seguir el palo, si
  no se tiene fallar con triunfo, si tampoco cualquiera; gana el triunfo más alto o, si no hay, la
  más alta del palo de salida.

Se valida contra `engine.py` jugada por jugada (`tests/test_jax_tricks.py`) y se mide con
`scripts/jax_engine/bench_tricks.py`.

En la laptop JAX corre en el procesador (`jax[cpu]` en `.venv`). En el lab va en un entorno aparte,
`~/scratch/work/venvs/jax` (`jax[cuda12]`), para no mezclar sus librerías de CUDA con las del PyTorch
que usan las corridas.

**Cuidado en el lab:** JAX reserva al arrancar el 75% de la memoria de la GPU. Con corridas de
PyTorch andando, usar siempre `XLA_PYTHON_CLIENT_PREALLOCATE=false` (o `XLA_PYTHON_CLIENT_MEM_FRACTION`).
