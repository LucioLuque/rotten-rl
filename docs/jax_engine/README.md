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

Con una red del tamaño de la de PPO (808 → 512 → 512 → 60) decidiendo cada jugada sobre una
observación de relleno: en el procesador de la laptop ~53.000 jugadas por segundo, igual que el motor
actual sin red: ahí la red es todo el costo. Lo que importa es la GPU.

**Contra qué comparar en la GPU:** el entrenamiento real hoy, con 8 corridas compartiendo la 4090,
hace ~220 partidas de 3 jugadores por segundo (26/10), unas 186 decisiones cada una (147 cartas y
39 cantos): **~40.000 decisiones por segundo** en total, con red y observación. Para ganar 10 veces,
JAX con red en la GPU tiene que pasar las ~400.000 (todavía sin observación real ni actualización de
PPO, que suman costo).

El wrapper de pyspiel juega la escalera completa (rondas de 1 a 7 cartas), no solo de 7: la
comparación es aproximada.

En el lab, JAX va en un entorno aparte (`~/scratch/work/venvs/jax`, `jax[cuda12]`) para no mezclar
sus librerías de CUDA con las del PyTorch de las corridas.
