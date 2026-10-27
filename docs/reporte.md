# Reporte de la noche del 26 al 27/10/2026

Lo que se hizo después de las 22:00 del lunes, con Lucio despierto al principio. Se va completando
durante la noche; la versión final se escribe antes del corte de las 07:00. Abajo queda el reporte
de la noche anterior.

## Resumen

- **VPN**: se reconecta sola (script de root con sudo sin contraseña, login SAML en una instancia
  de Chrome aparte, monitor que reconecta si se cae o a las 7h45). Detalle en `NOTAS_LOCALES.md`.
- **Decidido**: NFSP queda como referencia, se sacó la compatibilidad con el código viejo, se pasa
  a 3 jugadores comparando DQN y PPO compartidos (ver "Resultados finales" de la noche anterior).
- **Benchmark de paralelismo** (`docs/estado.md`): hasta 8 corridas de CPU a la vez cada una se
  frena poco; el total sube hasta ~20 y con 24 baja.
- **Partidas en lote y GPU** (`--vector_games`): PPO y DQN de 3 jugadores pasan de ~210 y ~400 ms
  por partida a ~16–37 y ~30 en la RTX 4090 del lab; la evaluación, de ~78 s a ~8 s. Validado:
  aprende igual que de a una (ver abajo). 41 tests.
- **PPO le gana a DQN** con 3 jugadores (3 semillas cada uno, 200k: 0.43 contra 0.38 de win vs
  heurística, diff +6.4 contra +3.2) y con 4 (0.36–0.39 contra 0.31), y tarda la mitad.
  **Recomendación: seguir con PPO.**
- **PPO escala a más jugadores**: le gana a la heurística con 4 (0.364–0.387, empate 0.25) y con
  5 (0.332–0.346, diff ~+10, empate 0.20), 3 semillas cada uno, y ya con 7 a los 50k (0.19,
  empate 0.14).
- **Lotes más grandes, sin confirmar**: con 3 jugadores, 128 partidas por actualización dio 0.458
  (una semilla) contra 0.43 de la base de 32; con 64, una semilla dio 0.445 y otra 0.426 (igual
  que la base). Hay que repetir 128 con más semillas antes de adoptarlo. Con 32, PPO se estanca
  (igual a los 400k que a los 200k). Capas de 1024 y menos entropía no ayudan.
- **La VPN se reconectó sola** a las 06:02 (7h45), en 4 segundos.

## Qué se hizo

### Corridas

- 22:48: barrido 7 (`queue3`, 8 a la vez, de a una partida en CPU): `t10`–`t12` (PPO 3 jugadores),
  `t20`–`t22` (DQN 3 jugadores), `u10`/`u20` (PPO/DQN 4 jugadores), 200k cada una. A esa velocidad
  ninguna terminaba en la noche (PPO ~24k partidas por hora, DQN ≤15k).
- 23:19: **cortadas a pedido de Lucio** con SIGTERM (checkpoint guardado), para validar primero el
  modo en lote: PPO de 3 en ~10–11k, DQN de 3 en ~6.5–6.8k, `u10` en 8k, `u20` en 4.7k. Se
  retoman con `--resume` si hiciera falta.
- 23:16: validación del modo en lote, 3 jugadores, misma config que `q10`/`q20`, 50k, GPU:
  `v10`, `v11` (PPO, semillas 0 y 1) y `v21` (DQN, con el aprendizaje rápido; `v20` era la primera
  versión y se reemplazó). Corren desde una copia aparte del repo (`~/scratch/dev/rotten-rl`, con
  `PYTHONPATH=.`), sin commitear todavía.

### Código

- `rotten_rl/vector.py`: `VectorRunner`, K partidas a la vez con la red decidiendo en lote.
- PPO y DQN: `batch_step`/`batch_end`, una trayectoria por asiento de cada partida. DQN, además:
  transiciones de a lote, muestreo con reposición y el info-state comprimido hasta la GPU.
- `evaluate(..., parallel=True)`: todas las partidas de la evaluación a la vez.
- `scripts/train.py --vector_games K` (1 = como siempre); `bench_parallel.py --gpu`.
- Tests: con una partida, el modo en lote ve exactamente lo mismo que de a una; los rewards llegan
  completos; PPO y DQN aprenden y se retoman idéntico en lote; el buffer en lote queda igual que
  de a una; evaluar en paralelo una partida da las mismas métricas.

## Resultados

### Validación del modo en lote (3 jugadores, vs heurística)

| Episodios | `q10` (PPO de a una, anoche) | `v10` (PPO en lote, s0) | `v11` (PPO en lote, s1) |
|---|---|---|---|
| 5k | 0.273 | 0.305 | 0.266 |
| 10k | 0.292 | 0.275 | 0.312 |
| 25k | 0.347 | 0.334 | 0.363 |
| 50k | 0.343 (diff +1.45) | 0.353 (diff +2.43) | 0.381 (diff +3.01) |

| Episodios | `q20` (DQN de a una, anoche) | `v21` (DQN en lote, s0) |
|---|---|---|
| 5k | 0.186 | 0.245 |
| 15k | 0.269 | 0.266 |
| 30k | 0.314 (diff +0.20) | 0.348 (diff −0.44) |
| 50k | — (cortada a los 30k) | 0.322 (diff +0.92) |

**El modo en lote aprende igual** (las diferencias están dentro del ruido de 500 partidas por
evaluación; si algo, en lote va un poco mejor). `v10`/`v11` tardaron 35 minutos para 50k
(compartiendo la GPU con otras pruebas), contra ~110 de `q10` en CPU.

### Velocidad en la GPU

`bench_parallel.py --gpu` con PPO en lote, 3 jugadores (con las 3 de validación usando la GPU):

| A la vez | ms por partida | Partidas/s en total |
|---|---|---|
| 1 | 16.1 | 62 |
| 2 | 18.7 | 107 |
| 4 | 24.1 | 166 |
| 8 | 36.6 | 219 |
| 12 | 50.6 | 238 |

8 a la vez es el punto justo. Perfil del modo en lote y lo que queda por optimizar:
`docs/optimizacion.md`. Probado y descartado: 64 partidas a la vez en vez de 32 (misma
velocidad).

### Barrido 8: PPO con 3 jugadores (terminado ~01:45)

Promedio de las últimas 3 evaluaciones (500 partidas cada una) contra la heurística, empate 0.333:

| Corrida | Win | Diff | Acierto de canto | Sesgo |
|---|---|---|---|---|
| `w10` (s0) | 0.426 | +6.7 | 0.542 | −0.07 |
| `w11` (s1) | 0.429 | +6.4 | 0.536 | −0.06 |
| `w12` (s2) | 0.430 | +6.5 | 0.542 | −0.08 |

Las 3 semillas dan lo mismo y le ganan claro a la heurística, cantando sin sesgo. La curva sigue
subiendo despacio (~0.37 entre 40k y 80k, ~0.43 a los 200k). Tardaron ~1h50 con 8 corridas
compartiendo la GPU.

### Barrido 8: PPO con 4 jugadores, `w30` (terminado 02:19)

Contra la heurística (empate 0.25): **0.364 de win, diff +7.5**, acierto de canto 0.626, sesgo
−0.05 (promedio de las últimas 3 evaluaciones). La curva se aplana desde ~120k: 0.280 (40k),
0.327 (80k), 0.373 (120k), 0.372 (160k), 0.369 (200k). Con 4 jugadores le gana a la heurística
por más margen relativo que con 3. `w31`/`w32` (las otras semillas) van parecido a esa altura.

### Barrido 8: DQN con 3 jugadores (terminado 03:39) — **PPO gana la comparación**

Promedio de las últimas 3 evaluaciones contra la heurística (empate 0.333), 200k, 3 semillas:

| | Win | Diff | Acierto de canto | Sesgo | Tiempo (8 a la vez) |
|---|---|---|---|---|---|
| **PPO** (`w10`–`w12`) | **0.426 – 0.431** | **+6.2 a +6.7** | 0.54 | −0.07 | ~1h50 |
| DQN (`w20`–`w22`) | 0.369 – 0.394 | +1.7 a +4.6 | 0.51 | +0.03 | ~3h45 |

Los rangos de las semillas no se superponen. DQN sigue subiendo despacio (0.36 a los 80k, 0.38
a los 200k), pero a los 200k está donde PPO estaba a los ~80k, y tarda el doble por partida
(aprende cada 16 decisiones). **Recomendación: seguir con PPO.** DQN queda como segunda referencia,
como NFSP.

### PPO con 4 jugadores, 3 semillas, y lotes de 64 partidas (terminados ~04:05)

Contra la heurística, promedio de las últimas 3 evaluaciones:

| Corrida | Win | Diff | Acierto de canto | Empate |
|---|---|---|---|---|
| `w30` / `w31` / `w32` (PPO 4 jugadores) | 0.364 / 0.368 / 0.387 | +7.5 / +8.4 / +9.2 | 0.63 | 0.25 |
| `x20` (PPO 3 jugadores, 64 partidas por actualización) | 0.445 | +7.4 | 0.55 | 0.333 |

PPO con 4 jugadores es consistente entre semillas. `x20` queda apenas por encima de las 3
semillas con 32 (0.426–0.431): indicio a favor de lotes más grandes, con una sola semilla. `w10`
retomada (`x11`) sigue mejorando pasados los 200k: 0.459 y diff +8.8 a los 240k.

### PPO con 3 jugadores: tamaño del lote, entropía y entrenar más (terminados ~05:18)

Promedio de las últimas 3 evaluaciones contra la heurística (empate 0.333), a los 200k:

| Variante | Win | Diff | Acierto de canto |
|---|---|---|---|
| 32 partidas por actualización, base (`w10`–`w12`, 3 semillas) | 0.426 – 0.431 | +6.2 a +6.7 | 0.54 |
| 64 partidas (`x20`) | 0.445 | +7.4 | 0.55 |
| **128 partidas (`x21`)** | **0.458** | **+8.6** | 0.55 |
| Entropía 0.005 en vez de 0.01 (`x23`) | 0.415 | +5.5 | 0.53 |
| `w10` seguida hasta 400k (`x11`, a los 400k) | 0.427 | +6.9 | 0.54 |

- **Con lotes de 32, PPO se estanca**: `w10` a los 400k está igual que a los 200k.
- **Lotes más grandes: sin confirmar.** 128 dio la mejor corrida de 3 jugadores (una semilla).
  Con 64, `x25` (semilla 1, terminada 06:18) dio 0.426 y diff +6.7, igual que la base, contra
  0.445 de `x20`: la diferencia de `x20` parece ruido entre semillas. Para saber si 128 sirve hay
  que repetirlo con 3 semillas (MAPPO usa lotes grandes, así que es razonable probarlo).
- **Menos entropía empeora.**
- **Capas de 1024 no ayudan** (`x24`, terminada 06:05): 0.400 y diff +5.7 a los 200k, por
  debajo de la base. Arrancó más rápido (0.40 a los 80k) pero no siguió subiendo.

### Cola después del barrido 8 (agregado durante la noche)

- 00:30: `w31`, `w32` (PPO 4 jugadores, semillas 1 y 2, para tener 3 con `w30`) y `w50` (PPO 5
  jugadores, adelanto). Arrancaron 01:44.
- 01:45: `x20`, `x21` (PPO 3 jugadores con 64 y 128 partidas por actualización en vez de 32:
  MAPPO usa lotes grandes), `x10` (retoma `w10` hasta 1M para ver dónde se estanca; queda con
  checkpoint) y `x23` (PPO 3 jugadores con entropía 0.005 en vez de 0.01).
- 03:40: `x10` falló al arrancar (`No CUDA GPUs are available`): `run_queue.py` decidía si mostrar
  la GPU mirando `--device cuda` en la línea de la cola, y una corrida retomada no lo tiene (sale
  de su config). Se relanzó como `x11` con `--device cuda` (retomó bien desde los 200k) y se
  arregló `run_queue.py` (lee el device del config.json al retomar).
- 03:45: con PPO ganador, `y70` (**PPO con 7 jugadores**, una semilla: si aprende y cuánto
  cuesta), `w51`/`w52` (PPO 5 jugadores, semillas 1 y 2) y `x24` (PPO 3 jugadores con capas de
  1024). Lo que no termine a las 06:55 queda con checkpoint.
- 04:27: terminó `w40` (DQN 4 jugadores): 0.309 de win y diff +4.7 contra la heurística (empate
  0.25), por debajo de PPO con 4 (0.364–0.387), como con 3 jugadores.
- 04:39: terminó `w50` (PPO 5 jugadores): **0.337 de win y diff +10.6** contra la heurística
  (empate 0.20; promedio de las últimas 3), acierto de canto 0.685, sesgo −0.03. Le gana por
  mucho desde los 40k (0.303) y se aplana desde ~80k (0.33–0.38).
- 06:41: terminaron `w51`/`w52` (PPO 5 jugadores, semillas 1 y 2): 0.346 y 0.332, diff +10.3 y
  +9.9. Con 3 semillas, PPO con 5 jugadores es tan consistente como con 3 y 4.
- 04:30: para confirmar el indicio de `x20`, `x25`/`x26` (PPO 3 jugadores con lotes de 64,
  semillas 1 y 2) y `x27` (PPO 4 jugadores con lotes de 64).

## Problemas de la noche

- **Cambio de máquina (error de Claude)**: una espera que buscaba el corte de las 06:55 siguió
  conectándose después de las 07:00. A las 07:32, con la labo-s004-04 reiniciando, el lab la mandó
  a la labo-s004-17 y desde ahí nos asigna esa. Los logs y checkpoints de la noche (incluidos los
  de `x11`, `x26`, `x27` e `y70`, cortadas a las 06:55, cuyos números finales no se llegaron a
  leer) quedaron en `/scratch` de la -04, que no se ve desde otras máquinas. Regla nueva: nada se
  conecta al lab después de las 07:00 en días hábiles (`CLAUDE.md`, `docs/lab-s004.md`).
- **La VPN se reconectó sola** a las 06:02:48 (7h45 de conexión, cortada a propósito antes del
  límite de 8 horas): túnel arriba en 4 segundos, sin intervención y con la pantalla bloqueada;
  `ssh labs004` siguió andando. Primera prueba real de la reconexión automática: anduvo.
- ~02:20–02:45: la conexión SSH compartida se rompió del lado del lab (`unknown userid`, ya
  conocido) y el monitor de corridas fallaba en silencio: no se vio el fin de `w30` hasta las
  02:45. No afectó a las corridas (la cola lanzó `x20` sola). El monitor ahora detecta ese error,
  cierra la conexión compartida y reintenta.

---

# Noche del 25 al 26/10/2026

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

## Resultados finales (vistos el lunes 26/10 a las 22:20)

Se lanzó toda la cola. Quedaron cortadas a las 06:55 `q20` (a los 30k), la corrida larga de NFSP (a
los 375k) y `e14` (cortada a propósito). Win rate y diff contra la heurística, 300 partidas por
evaluación (el ruido de una sola evaluación es de ±0.03 en win rate):

| Corridas | Algoritmo | Rondas como episodios | Episodios | Win | Diff | Sesgo |
|---|---|---|---|---|---|---|
| `r10`–`r12` | DQN compartido | sí | 50k | **0.546 – 0.576** | **+4.4 a +6.3** | +0.15 |
| `p10`–`p12` | PPO compartido | sí | 50k | 0.526 – 0.571 | +1.9 a +6.4 | −0.16 a +0.04 |
| `r20` | DQN compartido | no | 50k | 0.502 | −0.9 | +0.13 |
| `p20` | PPO compartido | no | 50k | 0.476 | −2.7 | +0.23 |
| `r00`–`r02` | NFSP combo | sí | 50k | 0.421 – 0.460 | −6.1 a −3.5 | +0.48 |
| `e22`–`e24` | NFSP combo | no | 50k | 0.435 – 0.468 | −4.8 a −3.2 | +0.38 |
| `e25` | NFSP combo | no | 200k | 0.489 | −1.8 | +0.45 |
| `nfsp2p_1M_lr1e-3` | NFSP | no | 375k | 0.530 | +3.0 | +0.27 |

3 jugadores (empate 0.33):

| Corrida | Algoritmo | Episodios | Win | Diff |
|---|---|---|---|---|
| `e19`, `e20` | NFSP | 50k | 0.16 – 0.17 | −13 |
| `q10` | PPO compartido + rondas | 50k | **0.331** | +0.9 |
| `q20` | DQN compartido + rondas | 30k (cortada) | 0.313 | −1.1 |

- DQN y PPO compartidos con rondas como episodios le ganan a la heurística con 2 jugadores en las
  3 semillas. Entre ellos la diferencia está dentro del ruido; PPO canta mejor (sin sesgo).
- Rondas como episodios suma ~0.06 de win rate a DQN y PPO; a NFSP no le cambia nada.
- NFSP necesita 375k episodios para lo que DQN y PPO hacen en 50k, y sigue cantando de más.
- Con 3 jugadores, PPO y DQN compartidos empatan con la heurística, donde NFSP perdía por 13
  puntos (una semilla cada uno).

### Decisiones (charladas con Lucio)

- **NFSP queda como referencia.** La corrida larga no se retoma: 375k episodios alcanzan como
  baseline. Sus métricas y redes siguen en `~/scratch/runs/`.
- **Se saca la compatibilidad con el código viejo** (alias `train_nfsp.py`, checkpoints del formato
  anterior a `rotten_rl/algorithms/`, `past_nets`). Las corridas lanzadas antes de ese cambio ya no
  se pueden retomar; si hiciera falta, se vuelven a correr desde cero.
- **Se pasa a 3 jugadores**, con DQN y PPO compartidos en paralelo (3 semillas cada uno, más
  episodios), para elegir el algoritmo con el que seguir. 2 jugadores queda como banco de pruebas
  rápido para los cambios de código.
