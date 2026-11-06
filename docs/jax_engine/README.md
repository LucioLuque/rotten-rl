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

## Etapa 2: partida completa con observación (6/11/2026)

`rotten_rl/jax_engine/game.py`: la partida entera como el juego "podrida" de pyspiel: escalera de 13
rondas, rotación del que reparte, canto con la restricción del último, bazas, puntaje, reward por
ronda ("raw" o "relative", /24) y la observación de 808 valores de cada jugador. Acciones iguales a
las de pyspiel (0..51 cartas, 52 + b cantos). El reparto de cada ronda sale de un orden de las 52
cartas (`new_game(orders)`), así una partida de pyspiel se puede reproducir exacta. Falta
`infer_trump_void` (apagado por defecto) y las manos ocultas del crítico central.

**Validación:** `tests/test_jax_game.py` juega partidas al azar en pyspiel con 2, 3, 4 y 7 jugadores
y las reproduce en JAX: en cada decisión coinciden el turno, las acciones legales, la observación de
todos los jugadores (tolerancia 1e-6, por el redondeo de float32) y los rewards; al final, los
returns. Se comprobó que el test detecta un error metido a propósito en la observación.

**Velocidad en la laptop (procesador)**, partidas completas de 3 jugadores, decisiones por segundo
(`scripts/jax_engine/bench_game.py`):

| | Decisiones por segundo |
|---|---|
| pyspiel + observación (como entrena hoy, sin red), de a una partida | ~9.500 |
| JAX, 1.024 partidas a la vez, con observación | ~198.000 (21 veces) |
| JAX + red del tamaño de PPO eligiendo la jugada | ~46.000 |

Con la red, la laptop sola ya iguala a las ~40.000 decisiones por segundo del lab entero con 8
corridas (sin la actualización de PPO, que falta). Falta medir en la GPU.

## Etapa 3: PPO en JAX (6/11/2026)

`rotten_rl/jax_engine/ppo.py` y `scripts/jax_engine/train_ppo.py`. La misma receta que las corridas
actuales (`--algo ppo --round_episodes --ppo_anneal_lr`, reward "relative"): una red de política y
una de valor compartidas por todos los asientos, self-play, acciones ilegales enmascaradas, GAE(0.95)
por asiento con cada ronda como episodio (el reward de la ronda en la última decisión de cada uno),
ventajas normalizadas, clipping 0.2, entropía 0.01, crítico con peso 0.5 sin recortar, gradiente
recortado a 0.5, Adam con la tasa bajando linealmente a 0 y la inicialización por defecto de PyTorch.

Lo que cambia es cómo corre: el lote de `--rollout_games` partidas se juega entero a la vez (todas
las partidas de n jugadores tienen la misma cantidad de decisiones, así terminan juntas), y jugar,
calcular ventajas y todas las pasadas por minibatches son una sola función compilada. Diferencia
menor: los minibatches que sobran de cada pasada (el resto de dividir por 512) no se usan.

Las redes se guardan como `state_dict` de PyTorch en `<corrida>/checkpoints/policy_ep<N>.pt` con un
`config.json` (`"engine": "jax"`): la evaluación, los torneos y `play.py` las cargan como a cualquier
corrida. Cada `--eval_every` partidas evalúa contra la heurística con la evaluación de siempre
(duplicada, `--eval_deal_seed 1234`). Guarda el estado completo en `<corrida>/state/ckpt.pkl` cada
`--checkpoint_every` partidas y al cortar (`--stop_at lab`, SIGTERM o Ctrl+C); `--resume <corrida>`
sigue (con `--games`, extiende el horizonte). Falta: el crítico central y evaluar contra `past`.

**Validación:** `tests/test_jax_ppo.py`: las ventajas coinciden con una reimplementación literal de
las del PPO de PyTorch (trayectoria por asiento, reward en la última decisión de la ronda), y la red
exportada da en PyTorch lo mismo que en JAX. Prueba en la laptop: con 2 jugadores, de 0.19 a 0.45 de
win rate contra la heurística en 2.048 partidas.

**En el lab:** el entorno `~/scratch/work/venvs/jax` tiene además `optax`, `open_spiel` y torch de
CPU (para evaluar; la GPU la usa solo JAX).

## Medición en la GPU del lab

`scripts/jax_engine/lab_bench.sh` (con la GPU libre): bazas, partidas completas de 2, 3 y 7
jugadores y velocidad de entrenamiento de PPO (2 y 7 jugadores, lotes de 512 y 2.048 partidas). Deja
todo en `~/scratch/logs/jax_bench.log`. Planeado para el viernes 6/11 a las 15:00, cortando las
corridas antes del evento de la tarde.

## Validación en el lab: `j23` contra `f23` (6/11/2026)

`j23_jaxppo2p_b512_anneal_1M_s1`: PPO en JAX con la receta de `f23` (2 jugadores, lotes de 512, 1M,
semilla 1), compartiendo la GPU con 7 corridas de PyTorch. **1M partidas en 12 minutos** (~1.600
partidas por segundo; `f23` tardó más de 5 horas, ~50 por segundo con 8 corridas en la GPU).

| Partidas | `j23` (JAX) vs heurística | `f23` (PyTorch) |
|---|---|---|
| 25k | 0.578 (+5.4) | 0.558 (+6.1) |
| 250k | 0.646 (+14.1) | 0.658 (+13.9) |
| 500k | 0.696 (+18.0) | 0.693 (+17.4) |
| 750k | 0.720 (+21.0) | 0.716 (+21.3) |
| 1M | 0.738 (+23.9) | 0.738 (+23.7) |

Torneo (2.000 repartos): `j23` le saca +1.1 ± 0.4 puntos por partida a `f23` y a `f25` (las dos de
PyTorch a 1M; entre ellas +0.2). **Aprende igual o algo mejor, unas 30 veces más rápido por corrida.**

Siguiente: `j30` (3 jugadores, 3M, receta de `f30`) para validar con más jugadores; las mediciones
con la GPU libre a las 15:00.

## Etapa 4: usarlo como `train.py` (6/11/2026)

- `scripts/jax_engine/train_ppo.py` loguea y evalúa igual que `train.py`: `RunLogger` (mismos CSV
  `eval_<rival>_overall/by_h/by_seat`, `train.csv`, TensorBoard) y evaluación contra `random`,
  `heuristic` y `past` (la versión de la evaluación anterior, jugando su mejor jugada, como
  `frozen_act` del PPO de PyTorch). Acepta `--runs_dir`, `--run_name`, `--stop_at` y `--resume`.
- `scripts/run_queue.py --script scripts/jax_engine/train_ppo.py --python ~/scratch/work/venvs/jax/bin/python`
  lanza corridas de JAX desde la cola; a esas les deja la GPU visible y les pone
  `XLA_PYTHON_CLIENT_PREALLOCATE=false`.
- `scripts/jax_engine/profile_ppo.py`: cuánto tarda cada parte de una actualización (juego,
  observación, redes, GAE, pasadas por minibatches). En el procesador de la laptop (2 jugadores, 128
  partidas): las pasadas de actualización son el 89%, jugar con la red el 11% y el juego más la
  observación ~2%. En la GPU puede repartirse distinto: se mide a las 15:00.
