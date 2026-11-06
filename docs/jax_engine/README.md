# El juego en JAX

Qué es y por qué: `docs/tecnico/optimizacion.md`, "Elegido para hacer" (decidido el 6/11/2026).
Código en `rotten_rl/jax_engine/` (ver su README). Etapas: 1) bazas con 2 jugadores y medir;
2) motor completo con observación; 3) PPO en JAX; 4) integrarlo con checkpoints, evaluación y
torneos.

## Etapa 1: fase de bazas (6/11/2026)

`rotten_rl/jax_engine/tricks.py`: reparto y bazas (seguir palo, obligación de fallar, triunfo), sin
canto ni puntaje ni observación. Para 2 a 7 jugadores.

**Validación:** `tests/test_jax_tricks.py` juega 40 rondas al azar con 2, 3 y 7 jugadores (1 a 7
cartas) en `engine.py` y en JAX con las mismas jugadas; en cada paso coinciden el turno y las cartas
legales, y al final las bazas ganadas. Pasa.

**Velocidad** (`scripts/jax_engine/bench_tricks.py`, jugadas por segundo, 2 jugadores, 7 cartas, al
azar; solo simulación, sin observación ni red):

| Dónde | Motor actual | JAX |
|---|---|---|
| Laptop, procesador (8 núcleos) | ~50.000 (`engine.py` y el wrapper de pyspiel, igual) | ~260.000 con 1.024 partidas a la vez; ~350.000 con 65.536 (6–7 veces) |
| Lab, GPU (RTX 4090) | pendiente | pendiente: medir con la GPU libre, antes de relanzar a las 18:20 |

El wrapper de pyspiel juega la escalera completa (rondas de 1 a 7 cartas), no solo de 7: la
comparación es aproximada.

En el lab, JAX va en un entorno aparte (`~/scratch/work/venvs/jax`, `jax[cuda12]`) para no mezclar
sus librerías de CUDA con las del PyTorch de las corridas.
