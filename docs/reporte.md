# Reporte de la noche del 25 al 26/10/2026

Lo que se hizo mientras Lucio dormía, y lo que hay para decidir. Se va completando durante la noche;
la versión final se escribe antes del corte de las 07:00. El detalle de cada corrida está en
`docs/estado.md`.

## Resumen

(Se completa al final de la noche.)

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

(Se completan a medida que terminan las corridas.)

## Para decidir el lunes a la noche

- Sacar o no la compatibilidad con el código viejo y reentrenar solo la mejor configuración de NFSP
  con la estructura nueva (ver `docs/estado.md`).
- Cuántas corridas a la vez: correr `scripts/bench_parallel.py` con la máquina libre.
- Según los resultados: con qué algoritmo seguir (NFSP, DQN o PPO) y si pasar a 3 jugadores.

## Problemas de la noche

(Ninguno por ahora.)
