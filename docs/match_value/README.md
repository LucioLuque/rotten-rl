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
