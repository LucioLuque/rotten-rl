# Valor de la partida: probabilidad de ganar

Para que la política juegue a ganar la partida sin alargar el episodio (con la partida entera
como episodio empeoró, ver `docs/reportes/2026-11-06.md`, martes 10/11), la recompensa de cada ronda
suma λ × (cambio en la probabilidad de ganar la partida), como el reward global de Suphx o el ICM
en torneos de póker (`docs/bibliografia.md`). Sumado sobre la partida da exactamente el resultado
final (gané o no), y cada ronda sigue siendo un episodio.

- `rotten_rl/match_value/win_prob.py`: modelo "normal": lo que le falta sumar a cada jugador es
  normal e independiente con la varianza de las rondas que faltan; P(terminar primero) por
  cuadratura de Gauss-Hermite. Sirve para cualquier número de jugadores.
- `rotten_rl/match_value/data.py`: partidas de una política contra sí misma, con los puntos de cada
  jugador en cada ronda.
- `scripts/match_value/fit_win_prob.py`: ajusta la varianza por ronda con la mitad de las partidas y
  mide con la otra mitad qué tan bien predice al ganador (log loss, Brier, calibración) contra una
  tabla medida (ventaja sobre el mejor de los otros × rondas jugadas) y contra 1/n.

Primera prueba (laptop, heurística, 3 jugadores, 2.000 partidas): el modelo normal queda bien
calibrado (dice 75 %, pasa el 75 %) y le gana a la tabla en log loss en casi todas las rondas.

## Validación con partidas reales (10/11/2026)

`fit_win_prob.py` con la política de cada cantidad de jugadores jugando contra sí misma (muestreando,
como entrena): `s20` 36M (400.000 partidas), `s30` 24M (200.000) y `u72` 24M (100.000). Resultados
completos en `results/` (los `.json` son los que usa `--win_prob`).

- **El modelo normal predice al ganador igual o mejor que la tabla medida**, con 2, 3 y 7
  jugadores. Log loss después de 6 rondas: 2 jugadores 0,517 (tabla 0,517, 1/n 0,693); 3 jugadores
  0,810 (0,815; 1,099); 7 jugadores 1,406 (1,419; 1,946). Solo en las últimas 1–2 rondas la tabla
  es un poco mejor (quedan pocos puntos posibles y la normal los aproxima mal).
- **Calibrado**: cuando dice 25 %, 50 % o 75 %, pasa eso ±1–2 puntos (con 7 jugadores, entre 0,5 y
  0,7 subestima ~3 puntos y arriba de 0,9 dice 96,8 % donde pasa 94,5 %).
- Varianza por ronda (σ²): con 2 jugadores ~53–60 en todas las rondas; con 3 sigue el tamaño de la
  mano (28 con 1 carta, 46 con 7); con 7, de 10 a 26.

Conclusión: sirve como potencial para las tres cantidades de jugadores, sin red ni tabla.

## Uso en el entrenamiento

`scripts/jax_engine/train_ppo.py --episode round --win_prob <.json> --match_bonus <λ>`: cada ronda
sigue siendo un episodio y su recompensa es la de siempre (puntos relativos) + λ × (cambio de la
probabilidad de ganar en esa ronda). La varianza se copia a la config de la corrida
(`round_var`), así que `--resume` no necesita el `.json`. Test: `tests/test_jax_ppo.py`
(`test_win_prob_shaping`: las recompensas de una partida suman la parte del triunfo − 1/n).
