# Estado del proyecto

Última actualización: 25/10/2026.

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
- Comparar con `python scripts/summarize_runs.py --runs_dir ~/scratch/runs --filter e0`.

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

### Para retomar el lunes 26/10 (acceso desde las 22:00)

1. Levantar la VPN y verificar con `ssh labs004 hostname` que sea `labo-s004-04` (la máquina de
   hoy: los checkpoints están en su disco local, `~/scratch/runs/<corrida>/state/`).
2. Retomar la corrida larga:
   `python scripts/train_nfsp.py --resume ~/scratch/runs/nfsp2p_1M_lr1e-3_20261025-200043 --stop_at 06:55`.
3. Retomar con `--resume` los experimentos de la cola que hayan quedado cortados a las 06:55
   (`grep -l Cortado ~/scratch/logs/e*.log`) y volver a lanzar `run_queue.py` para los pendientes
   (los ya lanzados están en `~/scratch/queue.txt.started` y no se repiten).

**Si no aparecen los checkpoints** (otra máquina asignada, máquina reinstalada o `~/scratch`
borrado): se pierden las corridas en curso. Las de 50k se relanzan desde cero (~40 min cada una);
la larga también, desde cero. Para que no vuelva a pasar, antes de relanzar implementar:

- `--backup_dir` en `train_nfsp.py`: después de cada checkpoint, copiar solo `meta.pt` (redes,
  optimizadores, contadores y RNGs: ~20 MB con 2 jugadores, ~40 MB con 4) a una carpeta del home
  (NFS: compartido por las 22 máquinas, persistente, cuota de 2 GB) y borrar la copia anterior. Sin
  los buffers (pesan GB y no entran en la cuota): al retomar desde esa copia se vuelven a llenar.
- Que `--resume` use esa copia si no encuentra checkpoints en `~/scratch`.
- Un comando para traer a la laptop, al conectarse, las métricas y las redes de cada evaluación
  (`rsync` sin `state/`), para tener los resultados a salvo y graficarlos.

Descartado: copiar cada checkpoint completo a la laptop. El lab no puede conectarse a la laptop,
así que la laptop tendría que quedar prendida toda la noche con la VPN, bajando ~2.3 GB cada ~20
minutos; si se corta algo, deja de copiar sin avisar.

El home tiene 51 MB usados de 1.9 GB (las dos primeras corridas, en `~/rotten-rl/runs`); las
corridas nuevas van a `~/scratch/runs`, que no tiene cuota (605 GB libres).

Conclusión del barrido de learning rate: 0.01 es demasiado alto; 0.001 y 0.0003 andan parecido (la
diferencia está dentro del ruido de 300 partidas) y siguen mejorando de a poco a los 50k. Las dos
cantan de más.

Los resultados de cada corrida están en `~/rotten-rl/runs/<corrida>/` y el log en
`~/scratch/logs/`.

## Bugs encontrados (25/10/2026)

1. **Reward perdido entre turnos.** El juego da reward al cerrar cada ronda, pero cada agente solo
   ve los TimeSteps de sus turnos y `rl_environment` no acumula: el reward le llegaba solo al que
   jugaba justo después del cierre. Se perdía el 46% del reward con 2 jugadores, 69% con 4 y 74% con
   7. Arreglado con `rotten_rl.training.with_pending_reward`; lo cubre un test.
2. **Exploración del DQN de NFSP.** OpenSpiel 2.x usa por defecto ε de 1.0 a 0.1 en 1M pasos: la
   best-response jugaba casi al azar y la avg-policy lo imitaba. Ahora el script usa 0.06 → 0.001
   (valores del paper).

## Próximos pasos

1. Ver si NFSP aprende con los dos arreglos (`nfsp2p_50k_fix`).
2. ~~Loop de entrenamiento propio y checkpoints~~ Hecho: `rotten_rl/training.py`
   (`EpisodeRunner`, 1.65× más rápido) y `rotten_rl/checkpoint.py`; `train_nfsp.py` tiene
   `--resume`, `--stop_at` y corte prolijo por señal. Cortar y retomar da resultados idénticos.
3. Ajustar hiperparámetros de NFSP (learning rate en curso) hasta que aprenda bien con 2 jugadores.
4. Corridas largas de 2 jugadores.
5. ~~Bajar la memoria de los buffers~~ Hecho: `--compact_buffers` (sin pérdida, resultados
   idénticos, ~7 veces menos RAM por agente). Ver `docs/optimizacion.md`.
6. Corridas de 3 y 4 jugadores.
7. PSRO simétrico con 3 jugadores.
8. Un modelo para n de 2 a 4; después escalar a 7.
