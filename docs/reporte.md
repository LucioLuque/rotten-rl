# Reporte de la noche del 25 al 26/10/2026

Lo que se hizo mientras Lucio dormía, y lo que hay para decidir. Se va completando durante la noche;
la versión final se escribe antes del corte de las 07:00. El detalle de cada corrida está en
`docs/estado.md`.

## Resumen

- **Código**: varios algoritmos con una interfaz común (`scripts/train.py --algo nfsp|dqn|ppo`),
  DQN y PPO con red compartida por todos los asientos, y rondas como episodios. Validado con 29
  tests y en el lab (NFSP da resultados idénticos byte a byte al código anterior).
- **Mejor NFSP**: la combinación (reward relativo + lr del DQN 0.0003 + capas 512 + batch 256), 3
  semillas a los 50k: **0.448 de win rate vs heurística, diff −4.3**. Con rondas como episodios
  (`r00`–`r02`) da prácticamente lo mismo.
- **Primer agente que le gana a la heurística: DQN compartido** (`r10`), a los 20k episodios: 0.503
  y diff +1.1. Una sola semilla, a mitad de la corrida (último dato antes del corte).
- **PPO** fue el más rápido por episodio y en la prueba corta llegó a lo de NFSP en 2.000
  episodios; su barrido (`p10`–`p20`) se lanzaba después del corte.
- **3 jugadores**: NFSP pierde claramente contra la heurística (0.16–0.17, empate 0.33).
- **La VPN se cortó a la 01:33** (límite de 8 h de la sesión): desde ahí no hubo más seguimiento.
  Las corridas siguieron solas en el lab; los resultados del resto de la noche se ven el lunes.

## Qué se hizo

### Código

- **Varios algoritmos con una interfaz común** (`rotten_rl/algorithms/`): NFSP (un agente por
  asiento, el de siempre), **DQN con red compartida** por todos los asientos y **PPO con política y
  crítico compartidos** (self-play contra la política actual, entropía moderada, pocas épocas por
  lote). Un solo `scripts/train.py --algo nfsp|dqn|ppo`; evaluación, logging, checkpoints,
  `--resume`, `--stop_at` y la cola son los mismos para todos.
- **`--round_episodes`**: para los agentes cada ronda es un episodio (idea del paper de Wizard).
- Validado: `train.py --algo nfsp` da CSVs idénticos byte a byte al código anterior; los tres
  algoritmos se retoman desde un checkpoint con resultados idénticos a entrenar de corrido
  (incluido PPO cortado con un lote a medio juntar). 29 tests.
- Pruebas cortas en el lab (2 jugadores, reward relativo, rondas como episodios):
  - DQN compartido: 0.49 → 0.71 vs random en 1.500 episodios.
  - **PPO compartido: en 2.000 episodios, 0.69 vs random y 0.42 vs heurística (diff −6.5)**, lo
    mismo que NFSP a los 50.000. Una sola corrida corta con 200 partidas de evaluación: falta
    confirmarlo (barrido 6).

### Corridas de la noche

En la cola (`queue2`, límite de 6 entrenamientos a la vez), todas 2 jugadores, 50k episodios,
reward relativo, 3 semillas donde dice s0–s2:

| Barrido | Corridas | Qué prueba |
|---|---|---|
| 5 | `r00`–`r02` | NFSP (mejor combinación de hoy) + rondas como episodios; comparar con `e22`–`e24` |
| 5 | `r10`–`r12`, `r20` | DQN compartido con y sin rondas como episodios |
| 6 | `p10`–`p12`, `p20` | PPO compartido con y sin rondas como episodios |

Además siguen las de antes: la corrida larga de NFSP (`nfsp2p_1M_lr1e-3`, baseline, corta sola a
las 06:55), `e20`–`e24` y `e25_combo_200k`.

## Resultados

### ¿Qué algoritmo es más rápido? (pregunta de Lucio, ~23:50)

Hay dos velocidades distintas:

**Tiempo por episodio** (medido en el lab con las tres corriendo a la vez, más ~7 corridas de la
noche; 2 jugadores, capas de 512, reward relativo, rondas como episodios):

| Algoritmo | ms por episodio | Por qué |
|---|---|---|
| PPO compartido | **113–130** | Aprende en lotes cada 32 partidas; entre medio solo juega |
| NFSP (un agente por asiento) | 138 | Dos redes por agente, aprende cada 64 pasos |
| DQN compartido | 176 | Aprende cada 16 pasos con batch 256 (como en Wizard): más cuentas por episodio |

Los números absolutos están inflados por la carga de la máquina; lo que vale es la comparación. DQN
se puede acelerar aprendiendo menos seguido (`--learn_every`), a costa de aprender menos por
episodio.

**Episodios necesarios para aprender** (lo que más importa): en las pruebas cortas, PPO llegó en
2.000 episodios a 0.42 vs heurística (diff −6.5), lo mismo que NFSP a los 50.000; DQN pasó de 0.49
a 0.71 vs random en 1.500. Si se confirma con el barrido 6, **PPO sería el más rápido en los dos
sentidos**. Falta confirmarlo con 3 semillas y 50k episodios.

### NFSP: barridos 2 y 3 (terminados ~23:50)

Vs heurística, promedio de las últimas 3 evaluaciones de 500 partidas (la base con 3 semillas daba
0.39–0.42 y diff −8.6 a −9.0; reward relativo con 3 semillas, diff −6.1 a −6.8):

| Corrida | Qué cambia | win | diff |
|---|---|---|---|
| `e12_relative_antic0.3` | relativo + anticipatory 0.3 | **0.461** | **−4.6** |
| `e15_relative_lr3e-4` | relativo + lr 0.0003 en las dos redes | 0.441 | −5.3 |
| `e22`–`e24` (a los 35k) | combinación: relativo + lr DQN 0.0003 + capas 512 + batch 256 | 0.437–0.440 | −4.5 a −5.0 |
| `e21_relative_h512` (a los 40k) | relativo + capas 512 | 0.424 | −6.3 |
| `e16_sl1e-4` | lr de la red promedio 0.0001 | 0.424 | −6.8 |
| `e17_antic0.05` | anticipatory 0.05 | 0.422 | −7.4 |
| `e13_eps0.2`, `e18_relative_eps0.2` | ε inicial 0.2 | 0.35 / 0.38 | −13.1 / −11.0 |

- **La combinación funciona y es consistente.** Finales a los 50k (00:17): `e22` 0.435 / −4.8,
  `e23` 0.440 / −4.8, `e24` 0.468 / −3.2; promedio **0.448 / −4.3**, ~2 puntos mejor que solo reward
  relativo y ~4.5 mejor que la base. Es la mejor configuración de NFSP hasta ahora: es la base de
  `r00`–`r02` (lo mismo + rondas como episodios). `e25` (la misma combinación, 200k) a los 50k: 0.423
  / −5.9, dentro de lo esperable entre semillas.
- **Más exploración (ε 0.2) empeora**, confirmado con y sin reward relativo.
- `e12` (anticipatory 0.3 con reward relativo) fue la mejor corrida individual de NFSP; sin reward
  relativo, anticipatory 0.3 casi no cambiaba nada. Es una sola semilla.

### 3 jugadores (terminados)

| Corrida | vs random (empate 0.33) | vs heurística (empate 0.33) | diff vs heurística |
|---|---|---|---|
| `e19_3p_base` | 0.67 | 0.17 | −12.7 |
| `e20_3p_relative` | 0.73 | 0.16 | −13.4 |

Con 3 jugadores NFSP le gana bien a random pero **pierde claramente contra la heurística** (gana
la mitad de lo que daría empatar), y el reward relativo no ayuda. Canta un poco de menos (sesgo
−0.1 a −0.2), al revés que con 2 jugadores. Es el caso donde más se espera que ayuden las ideas de
la bibliografía (red compartida: 3 veces más datos por red).

### Barrido 5, resultados parciales (01:20)

| Corrida | Episodio | vs heurística | diff | acierto de canto |
|---|---|---|---|---|
| **`r10` DQN compartido + rondas** | 20.000 | **0.503** | **+1.1** | **0.359** |
| `r00`–`r02` NFSP (combinación) + rondas | 30–40k | 0.44–0.45 | −3.9 a −5.0 | 0.33 |
| referencia: `e22`–`e24` NFSP (combinación) sin rondas | 35k | 0.44 | −4.5 a −5.0 | 0.33 |

- **Primer agente que le gana a la heurística**: el DQN compartido supera 0.5 de win rate y tiene
  diferencia de puntos positiva, con menos de la mitad de episodios que NFSP. Una sola semilla y a
  mitad de la corrida: falta `r11`, `r12`.
- Para NFSP, rondas como episodios casi no cambia nada a igual cantidad de episodios. La mejora del
  DQN viene de la red compartida o del algoritmo; `r20` (DQN sin rondas) separa los dos efectos.

(El resto de los resultados se completa a medida que terminan las corridas.)

## Para decidir el lunes a la noche

- Sacar o no la compatibilidad con el código viejo y reentrenar solo la mejor configuración de NFSP
  con la estructura nueva (ver `docs/estado.md`).
- Cuántas corridas a la vez: correr `scripts/bench_parallel.py` con la máquina libre.
- Según los resultados: con qué algoritmo seguir (NFSP, DQN o PPO) y si pasar a 3 jugadores.

## Problemas de la noche

- **La VPN se cortó a la 01:33**, después de exactamente 480 minutos conectada: la VPN de la
  facultad parece cortar las sesiones a las **8 horas**. Reconectar necesita `sudo` y el login en
  el navegador, así que desde ahí no pude ver el lab. Las corridas no dependen de la VPN y
  siguieron solas: la cola lanza hasta las 06:00 y todo corta a las 06:55 guardando checkpoint.
- `e14_relative_200k` lo corté a las 23:10 (a los 52.5k, con checkpoint), antes de que Lucio pidiera
  no cortar nada. Era redundante con `e25_combo_200k` y la corrida larga.
- Un `pgrep` mal armado dejó una espera infinita (no rompió nada) y la conexión SSH compartida se
  rompió una vez del lado del lab (`unknown userid`); los dos quedaron anotados en `CLAUDE.md` y
  `docs/lab-s004.md`.

## Qué mirar el lunes

Lo que estaba en la cola a la 01:33 (en orden): `r10`–`r12`, `r20` (DQN compartido), `p10`–`p12`,
`p20` (PPO compartido), `q10`, `q20` (3 jugadores con PPO y DQN compartidos). Con 6 a la vez y dos
lugares ocupados por la corrida larga y `e25`, probablemente no llegaron a lanzarse todas antes de
las 06:00; las que quedaron a medias cortaron a las 06:55 con checkpoint y se retoman con
`--resume`.

```bash
ssh labs004 'bash -lc "cd ~/rotten-rl && . ~/scratch/work/venvs/rotten-rl/bin/activate && \
  python scripts/summarize_runs.py --filter _round --last 3; \
  python scripts/summarize_runs.py --filter _game --last 3; \
  python scripts/summarize_runs.py --filter _3p --last 3; \
  grep -l Cortado ~/scratch/logs/*.log; grep -v ^# ~/scratch/queue2.txt | awk \"{print \$1}\" | \
  grep -vxF -f ~/scratch/queue2.txt.started"'
```
