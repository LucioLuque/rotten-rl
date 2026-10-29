# Estado del proyecto

Última actualización: 28/10/2026 (noche).

## Hecho

- Motor del juego, wrapper de OpenSpiel (observación de 808, 2 a 7 jugadores) y entrenamiento NFSP.
- Evaluación contra random, heurística y la versión anterior del agente, rotando asiento.
- Heurística con pesos de canto calibrados según la cantidad de jugadores.
- Entorno funcionando en el laboratorio S004 (`labo-s004-04`), unas 6 veces más rápido que la laptop.

## Corridas

| Corrida | Dónde | Config | Estado | Resultado |
|---|---|---|---|---|
| `baseline2p_20260726-230529` | otra máquina | 2 jugadores, 1M episodios | Solo llegó a la evaluación del episodio 0 | — |
| `nfsp2p_50k_20261025-175802` | lab | NFSP, 2 jugadores, 50k episodios, **con los dos bugs** | Terminada (49 min) | Plana: vs random 0.51, vs heurística 0.22, acierto de canto 0.25, igual que sin entrenar |
| `diag_dqn2p` | lab | DQN solo vs random, 2 jugadores, 20k episodios, reward arreglado | Terminada (20 min) | vs random 0.71, vs heurística 0.44 (de 0.14), acierto de canto 0.37 (de 0.26); canta de más (sesgo +0.8 vs heurística) |
| `nfsp2p_50k_fix_20261025-184740` | lab, `~/rotten-rl/runs/` | NFSP, 2 jugadores, 50k episodios, con los dos arreglos, lr 0.01 | Terminada | Estancada: vs random 0.57, vs heurística 0.27 (diff −19.7), acierto de canto 0.24, sesgo +0.64 |
| `nfsp2p_50k_lr1e-3_20261025-191935` | lab, `~/scratch/runs/` | Igual, con lr 0.001 (RL y SL), loop propio | Terminada (~35 min) | vs random 0.70, vs heurística **0.46 (diff −4.6)**, acierto de canto 0.33, sesgo +0.24 (vs random) / +0.40 (vs heurística) |
| `nfsp2p_50k_lr3e-4_20261025-191935` | lab, `~/scratch/runs/` | Igual, con lr 0.0003 (RL y SL), loop propio | Terminada (~35 min) | vs random 0.69, vs heurística 0.41 (diff −7.7), acierto de canto 0.34, sesgo +0.28 / +0.47 |

### En curso desde el 25/10/2026 20:00 (cortan solas a las 06:55)

- `nfsp2p_1M_lr1e-3` (tmux `long`): NFSP 2 jugadores, 1M episodios, lr 0.001, eval cada 25k con
  500 partidas, checkpoint cada 25k. No termina en una noche: seguir con `--resume`.
- Cola de experimentos (tmux `queue`, `scripts/run_queue.py`, cola en `~/scratch/queue.txt`, 5 a la
  vez): barrido 1, todos 50k episodios con lr 0.001 y eval de 500 partidas, cambiando una cosa por
  vez: `e00_base`, `e01_seed1`, `e06_seed2` (para medir el ruido entre semillas),
  `e02_antic0.2`, `e03_antic0.3`, `e04_relative` (reward relativo), `e05_h512` (capas de 512),
  `e07_rl3e-4` (lr del DQN 0.0003), `e08_batch256`, `e09_learn32`.
- Comparar con `python scripts/eval/summarize_runs.py --runs_dir ~/scratch/runs --filter e0`.

Resultados del barrido 1 (promedio de las últimas 3 evaluaciones de 500 partidas, vs heurística):

| Experimento | win | diff | acierto de canto | sesgo |
|---|---|---|---|---|
| `e00_base` (semilla 0) | 0.389 | −9.0 | 0.313 | +0.36 |
| `e01_seed1` | 0.399 | −8.6 | 0.306 | +0.45 |
| `e02_antic0.2` | 0.391 | −8.7 | 0.312 | +0.37 |
| `e03_antic0.3` | 0.403 | −7.9 | 0.313 | +0.48 |
| `e04_relative` | **0.428** | **−6.3** | 0.316 | +0.36 |

- Ruido entre semillas: ~±0.01 de win y ±0.5 de diff.
- `anticipatory_param` casi no cambia nada. El reward relativo parece ayudar (a confirmar con
  otras semillas). Todas cantan de más y rondan 0.40 vs heurística a los 50k.
- Barrido 2, agregado a la cola: `e10_relative_seed1`, `e11_relative_seed2`,
  `e12_relative_antic0.3`, `e13_eps0.2` (ε inicial 0.2) y `e14_relative_200k` (200k episodios).
- Desde las 21:36 hay una segunda cola (tmux `queue2`, `~/scratch/queue2.txt`) con
  `--max_total 11`: cuenta todos los entrenamientos de la máquina y solo lanza si hay menos de 11
  (el límite se cambia en caliente en `~/scratch/queue2.txt.max_total`). La primera cola sigue con
  sus 5 hasta terminar su lista. Barrido 3 en `queue2`: `e15_relative_lr3e-4`, `e16_sl1e-4`,
  `e17_antic0.05`, `e18_relative_eps0.2`, `e19_3p_base` y `e20_3p_relative` (primeras pruebas con
  3 jugadores), `e21_relative_h512`.

Resultados del barrido 1 completo y parte del 2 y 3 (vs heurística, promedio de las últimas 3
evaluaciones):

| Configuración | Corridas | win | diff |
|---|---|---|---|
| Base | `e00`, `e01`, `e06` | 0.39 / 0.40 / 0.42 | −9.0 / −8.6 / −8.7 |
| Reward relativo | `e04`, `e10`, `e11` | 0.43 / 0.42 / 0.42 | −6.3 / −6.8 / −6.1 |
| Capas de 512 | `e05` | 0.42 | −6.6 |
| lr del DQN 0.0003 | `e07` | 0.43 | −6.3 |
| Batch 256 | `e08` | 0.43 | −7.1 |
| Aprender cada 32 pasos | `e09` | 0.41 | −7.7 |
| Corrida larga a los 100k | `nfsp2p_1M_lr1e-3` | 0.43 | −6.1 |

- **Reward relativo confirmado** con 3 semillas: ~2.4 puntos mejor (el ruido entre semillas de la
  base es de ±0.2 puntos de diff).
- Capas de 512, lr del DQN 0.0003 y batch 256 mejoran 1.5–2.5 puntos (una semilla cada uno).
- La corrida larga sigue mejorando con más episodios.
- 3 jugadores (`e19`, a los 20k): vs random 0.64 (empate 0.33), vs heurística solo 0.12, y canta de
  menos (sesgo −0.15 a −0.24).
- Barrido 4, agregado a `queue2`: combinar reward relativo + lr del DQN 0.0003 + capas de 512 +
  batch 256, con 3 semillas (`e22`–`e24`, 50k) y una de 200k (`e25_combo_200k`).

### Cuántas corridas en paralelo (25/10/2026 ~23:00)

Con 11 entrenamientos a la vez, cada uno va a ~106 ms/episodio; con 6, a ~50. **11 es peor incluso
en total** (11/106 = 0.104 episodios/ms contra 6/50 = 0.120). Fijar la corrida larga a un núcleo
exclusivo no la aceleró (~96 → ~109 ms/ep), así que el freno no es compartir núcleo sino recursos
comunes (caché, memoria, frecuencia). Se bajó `~/scratch/queue2.txt.max_total` a 6 (las que ya
corren terminan; no se lanzan reemplazos hasta bajar de 6).

**Medido el 26/10/2026 con la máquina libre** (`scripts/dev/bench_parallel.py --ns
1,2,4,6,8,12,16,20,24 --episodes 2000`, NFSP de 2 jugadores, buffers chicos):

| A la vez | ms/episodio (mín / media / máx) | Episodios/s en total | Cuánto se frena cada una |
|---|---|---|---|
| 1 | 35.3 / 35.3 / 35.3 | 28.3 | 1.00× |
| 2 | 36.6 / 36.7 / 36.8 | 54.5 | 1.04× |
| 4 | 38.6 / 38.9 / 39.1 | 102.7 | 1.10× |
| 6 | 41.9 / 42.3 / 42.8 | 141.8 | 1.20× |
| 8 | 46.6 / 47.4 / 48.2 | 168.8 | 1.34× |
| 12 | 52.6 / 57.2 / 65.1 | 211.7 | 1.62× |
| 16 | 58.0 / 68.1 / 80.4 | 237.3 | 1.93× |
| 20 | 64.2 / 80.8 / 96.9 | 251.2 | 2.29× |
| 24 | 72.8 / 107.1 / 134.4 | 233.4 | 3.03× |

El total sube hasta ~20 a la vez y con 24 baja. Hasta 8 cada corrida se frena poco; con 12 o más
el total crece cada vez menos y cada corrida tarda mucho más. Con buffers grandes (NFSP con
reservoir de millones) se frena antes: la noche del 25, con 11 a la vez, iban a ~106 ms/episodio.
Criterio: **hasta 8 a la vez** si importa que terminen rápido (comparaciones para la mañana);
**12–16** si son muchos experimentos independientes y no importa cuándo termina cada uno.

### Plan a partir de la bibliografía (ver `docs/bibliografia.md`)

Probar, en orden y con 3 semillas cada uno:
1. Cada ronda como episodio (Wizard; Suphx).
2. Una red compartida por todos los asientos (Wizard).
3. DQN compartido y PPO en self-play contra la política actual, como alternativas a NFSP (Wizard,
   Big 2 2026, DouZero; NFSP no tiene garantías con más de 2 jugadores).
4. Canto y juego separados (Wizard, Judgement, AlphaDou; el canto es la decisión que más pesa).
5. Evaluación con repartos espejo (menos varianza).

**Hecho (25/10/2026 ~23:30):** `rotten_rl/algorithms/` con una interfaz común (`base.py`), NFSP
movido ahí (`nfsp.py`) y DQN con red compartida por todos los asientos (`dqn.py`, implementación
propia), y un solo `scripts/train.py --algo nfsp|dqn` con `--round_episodes` (1 y 2 del plan).
`train_nfsp.py` quedó como alias. Validado en el lab: `train.py --algo nfsp` da CSVs idénticos byte
a byte al código anterior; retoma checkpoints del formato viejo; DQN compartido con rondas como
episodios pasa de 0.49 a 0.71 vs random en 1.500 episodios.

**PPO (~23:40):** `rotten_rl/algorithms/ppo.py`, política y crítico compartidos, self-play contra
la política actual, entropía 0.01, 5 épocas por lote de 32 partidas (Big 2 2026; MAPPO). Se
retoma idéntico desde un checkpoint aunque corte con un lote a medio juntar. 29 tests. Prueba
corta en el lab (2 jugadores, relativo, rondas como episodios): a los 2.000 episodios, 0.69 vs
random y 0.42 vs heurística (diff −6.5), lo mismo que NFSP a los 50k. Barrido 6 en `queue2`:
`p10`–`p12` (PPO + rondas como episodios, 3 semillas) y `p20_ppo_game_s0` (sin rondas).

Barrido 5 en `queue2` (arranca a medida que se liberan lugares, límite 6), todo 2 jugadores, 50k,
reward relativo:
- `r00`–`r02`: NFSP con la combinación (lr DQN 0.0003, capas 512, batch 256) + `--round_episodes`.
  Comparar con `e22`–`e24` (lo mismo sin rondas como episodios).
- `r10`–`r12`: DQN compartido + `--round_episodes` (lr 0.0005, capas 512, batch 256, replay 300k).
- `r20_dqn_game_s0`: DQN compartido sin rondas como episodios (para separar los dos efectos).

`e14_relative_200k` se cortó a los 52.5k (redundante con `e25_combo_200k` y la corrida larga;
tiene checkpoint si hiciera falta).

PSRO no se descarta, pero baja de prioridad: con más de 2 jugadores no tiene garantías, y en Big 2
el self-play contra la política actual le ganó a entrenar contra un conjunto de versiones
anteriores. Entra después, para medir y mejorar la robustez de un agente ya fuerte.

### Lunes 26/10/2026

Resultados finales de la noche del 25 en `docs/reportes/2026-10-25.md` ("Resultados finales"). En resumen:
DQN y PPO compartidos con rondas como episodios le ganan a la heurística con 2 jugadores (3
semillas cada uno, diferencia entre ellos dentro del ruido) y empatan con ella con 3 jugadores;
NFSP queda atrás.

Decidido:
- NFSP queda como referencia; la corrida larga `nfsp2p_1M_lr1e-3` no se retoma (375k episodios:
  0.53 vs heurística, diff +3.0).
- Se sacó la compatibilidad con el código viejo: el alias `train_nfsp.py`, la lectura de
  checkpoints del formato anterior a `rotten_rl/algorithms/` y `past_nets`. Las corridas lanzadas
  antes de ese cambio ya no se retoman (sus métricas y redes siguen en `~/scratch/runs/`).
- Se pasa a 3 jugadores, con DQN y PPO compartidos en paralelo, para elegir algoritmo. 2
  jugadores queda como banco de pruebas rápido para los cambios de código.

**Partidas en lote y GPU** (`--vector_games 32 --device cuda`, `rotten_rl/vector.py`): DQN y PPO
juegan 32 partidas a la vez y la red decide en lote en la RTX 4090 del lab; la evaluación también
juega sus partidas a la vez y DQN tiene el replay buffer espejado en la GPU. PPO de 3 jugadores
pasa de ~210 a ~16–37 ms por partida, DQN de ~400 a ~30. Validado contra `q10`/`q20`: aprende
igual (ver `docs/reportes/2026-10-26.md`). En la GPU conviene **8 corridas a la vez** (~220 partidas/s en
total). `run_queue.py` le deja la GPU visible a las corridas con `--device cuda`. Lo que queda por
optimizar: `docs/tecnico/optimizacion.md`.

El barrido 7 (`t10`–`t22`, `u10`/`u20`, de a una en CPU) se cortó a las 23:19 para pasar al modo
en lote (tienen checkpoint). Lo reemplaza el barrido 8 (`w*`, en lote, `~/scratch/queue4.txt`).

### Resultados de la noche del 26 al 27/10 (detalle en `docs/reportes/2026-10-26.md`)

- **PPO le gana a DQN** con 3 y 4 jugadores (3 semillas, 200k) y tarda la mitad: se sigue con
  PPO compartido (`--algo ppo --round_episodes --reward_scheme relative --vector_games 32
  --device cuda`, capas 512,512). DQN y NFSP quedan como referencia.
- PPO contra la heurística, 200k, 3 semillas: 3 jugadores 0.426–0.431 (empate 0.333), 4
  jugadores 0.364–0.387 (empate 0.25), 5 jugadores 0.332–0.346 (empate 0.20). 7 jugadores
  (`y70`, una semilla) ya le gana a los 50k; quedó con checkpoint.
- Con lotes de 32 partidas, PPO con 3 jugadores se estanca en ~0.43. 128 partidas por
  actualización dio 0.458 (una semilla, sin confirmar); 64, igual que 32. Capas de 1024 y
  entropía 0.005 no ayudan.

Corridas con checkpoint para retomar (`--resume`): `y70_ppo_7p_s0` (7 jugadores), `x11` (= `w10`
seguida hacia 1M; con lotes de 32 ya no mejora, no hace falta), `x26`, `x27` (lotes de 64). Están
en `/scratch` de la labo-s004-04; desde el 27/10 07:32 el lab nos asigna la labo-s004-17 (ver
`docs/reportes/2026-10-26.md`), así que solo se pueden retomar si volvemos a la -04.

### Noche del 27 al 28/10 (detalle en `docs/reportes/2026-10-27.md`)

En la labo-s004-17 (entorno armado de cero; lo de la -04 no se puede retomar). Cola `queue5`,
8 a la vez en la GPU: PPO de 3 jugadores (`z10` base, `z11`–`z13` lotes de 128, `z14` tasa que
baja, `z15` inicialización ortogonal), `z40` (4 jugadores, lotes de 128) y `z70`–`z72` (7
jugadores). En CPU, la ronda de 2 cartas: 3 jugadores exacta y 4 agrupada. Perfil de 7
jugadores: el juego en Python es ~25% del tiempo, así que el C++ no es urgente.

Resultados hasta la 01:15: **PPO con lotes de 128 partidas por actualización** confirmado con 3
semillas (3 jugadores, 0.441–0.447 vs heurística contra 0.421 de la base; en el torneo le gana a
la base por 1.4–1.9 puntos por partida) y con 4 jugadores (0.393 contra 0.364–0.387). La tasa que
baja ayuda casi igual (una semilla); la inicialización ortogonal, no. Jugar la acción más probable
le gana a muestrear. En la ronda de 1 carta, todos los agentes cantan casi como el equilibrio
(pierden 0.01–0.08 puntos por ronda, la heurística 0.37). Recomendación: usar en PPO
`--ppo_rollout_episodes 128`.

Cierre de la noche (06:30): **lotes de 128 + tasa que baja** (`--ppo_anneal_lr`) da 0.463–0.467
con 3 jugadores (3 semillas) y 0.423 con 4 (antes 0.393); lotes de 128 también mejoran 5 (0.359)
y 7 jugadores (0.289). Con lotes de 256 o 512, 3 jugadores llega a ~0.50 (600k / 400k) y sigue
subiendo. Para retomar (en la -17, o desde el respaldo en `runs/lab/`): `z16r2`, `z18r2` (hacia
1M), `z21` (tasa que baja con horizonte 400k), `z51`, `z74`, `z75`. Próximo paso sugerido:
lotes de 256–512 + tasa que baja con horizonte largo, y llevarlo a 4–7 jugadores.

### Noche del 28 al 29/10 (detalle en `docs/reportes/2026-10-28.md`)

En la labo-s004-19 (otra vez entorno de cero; se subieron `z16`, `z18`, `z21` y `n3_exact` desde
el respaldo). Cola `queue6`, 8 en la GPU: `z21r` (hasta 400k), lotes de 512 + tasa que baja con 3
jugadores y 3 semillas (`a10`–`a12`, 400k), 4 jugadores (`a40`) y 7 (`a70`) con lotes de 256 +
tasa que baja, y `z16r2`/`z18r2` hacia 1M. En CPU, la ronda de 2 cartas exacta de 3 jugadores.
Probado: cortar la VPN no afecta a las corridas.

## Bugs encontrados (25/10/2026)

1. **Reward perdido entre turnos.** El juego da reward al cerrar cada ronda, pero cada agente solo
   ve los TimeSteps de sus turnos y `rl_environment` no acumula: el reward le llegaba solo al que
   jugaba justo después del cierre. Se perdía el 46% del reward con 2 jugadores, 69% con 4 y 74% con
   7. Arreglado con `rotten_rl.training.with_pending_reward`; lo cubre un test.
2. **Exploración del DQN de NFSP.** OpenSpiel 2.x usa por defecto ε de 1.0 a 0.1 en 1M pasos: la
   best-response jugaba casi al azar y la avg-policy lo imitaba. Ahora el script usa 0.06 → 0.001
   (valores del paper).

## Próximos pasos

1. ~~Que NFSP aprenda con 2 jugadores~~ Hecho (arreglos de reward y exploración, barridos 1–4).
2. ~~Loop de entrenamiento propio y checkpoints~~ Hecho: `rotten_rl/training.py`
   (`EpisodeRunner`, 1.65× más rápido) y `rotten_rl/checkpoint.py`; `--resume`, `--stop_at` y
   corte prolijo por señal. Cortar y retomar da resultados idénticos.
3. ~~Varios algoritmos~~ Hecho: NFSP, DQN y PPO compartidos con `--round_episodes`.
4. ~~Bajar la memoria de los buffers~~ Hecho: `--compact_buffers` (sin pérdida, resultados
   idénticos, ~7 veces menos RAM por agente). Ver `docs/tecnico/optimizacion.md`.
5. ~~DQN vs PPO compartidos~~ Hecho: gana PPO (26–27/10).
5b. ~~Observación más rápida~~ Hecho (27/10): idéntica bit a bit (`tests/test_observation.py`
   contra `tests/data/info_states.npz`), 31% más rápida con 3 jugadores y 47% con 7; la partida
   completa, ~8–10%. El historial de la ronda (quién jugó cada carta, voids, quién va ganando la
   baza) lo lleva el estado jugada a jugada. Ver `docs/tecnico/optimizacion.md`.
5c. ~~Revisar `ppo.py` contra la receta estándar~~ Hecho (27/10, contra los detalles de PPO de
   CleanRL). Sin errores: GAE por asiento, rewards entre turnos, máscara de acciones, recortes y
   ventajas normalizadas están bien; no recortar la pérdida del valor es a propósito (Andrychowicz
   et al. 2021: no ayuda). Diferencias que vale probar: (1) la tasa de aprendizaje fija, cuando la
   receta la baja hasta 0 (podría explicar el estancamiento con lotes de 32) → `--ppo_anneal_lr`;
   (2) la inicialización por defecto, cuando la receta es ortogonal con la última capa de la
   política ×0.01 → `--ppo_init orthogonal`; (3) el crítico ve solo lo del jugador (MAPPO usa uno
   que ve todas las manos al entrenar): después. Menores: `eps` de Adam, sin KL objetivo. Aparte:
   se evalúa con la acción más probable aunque se entrena muestreando; medir también la política
   muestreada.
5d. ~~Validar la evaluación~~ Hecho (27/10). Sin sesgo: con jugadores idénticos da el empate
   (azar 4 jugadores, 6.000 partidas: 0.247 contra 0.250 ± 0.006; heurística 7 jugadores, 2.800:
   0.142 contra 0.143 ± 0.007). Pero **500 partidas son pocas**: el win rate tiene ±0.022 y la
   diferencia entre dos corridas (promediando 3 evaluaciones) ±0.018, así que 0.458 (lotes de 128)
   contra 0.43 no es significativo. El asiento pesa mucho (heurística, 3 jugadores: 0.27 en el
   asiento 0, 0.42 en el 2) por la rotación del reparto y la restricción del último en cantar; se
   rota, así que no sesga. Cambios: evaluación duplicada (`--eval_deal_seed`, los mismos repartos
   con el agente en cada asiento, en todas las evaluaciones y corridas), error estándar en cada
   evaluación (`*_se`), `--eval_games` 1500 por defecto, `paired_diff` para comparar dos agentes
   reparto a reparto, y `scripts/eval/tournament.py` (todos contra todos: variantes, versiones
   anteriores, heurística). Medido: el duplicado baja poco el error en este juego (~15% al comparar
   dos agentes; lo que más pesa es clavar o no el canto, no las cartas), así que la palanca es jugar
   más partidas.
5e. ~~Resolver la ronda de 1 carta sin RL~~ Hecho (27/10): equilibrio con CFR+ de 2 a 7
   jugadores (`rotten_rl/exact/`, `docs/primera-ronda/`). Cantar último cuesta ~4 a ~6 puntos
   por ronda; con 7 jugadores nadie canta 1 sin triunfo; escuchar los cantos ajenos vale 0.07 a
   0.49 puntos por ronda; la heurística canta casi como el equilibrio salvo con 2 jugadores.
   Comparado con el agente (27/10, `scripts/exact/compare_round1_agent.py`): canta casi como el
   equilibrio.
5f. ~~Ronda de 2 cartas~~ Hecho para 3 jugadores (27/10): solver en C++ con MCCFR y agrupación de
   las cartas jugadas (`rotten_rl/exact/round2/`, `docs/segunda-ronda/`), verificado contra el
   motor. Converge en 8 minutos; la estrategia le saca ~1 punto por ronda a la heurística.
   Pendiente: comparar con la versión exacta, 4 a 7 jugadores (lab, por sesiones), el agente.
6. **Siguiente (noche del 27/10):** PPO, 8 a la vez: 3 jugadores con lotes de 128 (semillas 1 y 2,
   para confirmar el 0.458 de `x21`); 3 jugadores con lotes de 32 y `--ppo_anneal_lr`; 3 jugadores
   con lotes de 32 y `--ppo_init orthogonal`; 4 jugadores con lotes de 128; 7 jugadores con 3
   semillas (retomando `y70` si volvemos a la -04, si no de cero). Medir también cómo se reparte el
   tiempo con 7 jugadores (si la simulación pesa mucho, optimizar el motor antes de las largas).
   Al terminar las de 3 jugadores: torneo (`scripts/eval/tournament.py`, 2.000 repartos) entre la base,
   lotes de 128, tasa que baja, inicialización ortogonal, la heurística y la versión muestreada de
   la mejor, para decidir con números confiables; y comparar cómo canta el agente de 3 jugadores
   en la ronda de 1 carta con el equilibrio (`round1.policy_tables`).
   En paralelo, en los procesadores libres (si el lab tiene `g++`): ronda de 2 cartas con 3
   jugadores **exacta** (sin agrupar, para medir cuánto pierde la agrupación B) y con 4 jugadores
   agrupada, guardando cada 10 millones de iteraciones y cortando antes de las 06:55.
7. PSRO simétrico con 3 jugadores, para medir y mejorar la robustez de un agente ya fuerte.
8. Un modelo para n de 2 a 4; después escalar a 7.
