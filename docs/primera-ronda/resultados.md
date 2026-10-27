# La ronda de 1 carta, resuelta

Generado con `scripts/exact/solve_round1.py` (ver `rotten_rl/exact/round1.py`; qué es y por qué, en `README.md` de esta carpeta). Con una carta la jugada es
forzada: la única decisión es cantar 0 o 1, y quién gana la baza depende solo de las cartas.
Cada situación es el número del triunfo dado vuelta, la carta propia (triunfo o no y su
número: los otros tres palos son intercambiables) y lo cantado antes. El reward es el mismo con
el que entrena el agente: puntaje de la ronda menos el promedio de los rivales, en puntos.

Cómo leer las tablas:

- **Posición de canto**: 1 es el primero en cantar (y el que sale); la última tiene la
  restricción de que la suma no puede dar 1.
- **Valor en el equilibrio**: cuánto gana en promedio esa posición contra el resto si todos
  juegan el equilibrio (suma 0 entre todas).
- **Ganancia por desviarse**: cuánto ganaría quien cambie su forma de cantar mientras los demás
  siguen en equilibrio. Cerca de 0 = es un equilibrio. Es una cota algo alta (elegir lo mejor
  sobre una muestra finita suma un poco de ruido a favor).
- **Pérdida por mirar solo la carta**: cuánto pierde quien canta solo según su carta y su
  posición, sin darle peso a lo que cantaron los demás, contra rivales en equilibrio.
- **Heurística**: la de `evaluation.py` (cantar 1 si la carta "vale" media baza o más). La
  mejor respuesta es lo máximo que se le puede sacar a una mesa de heurísticas desde esa
  posición, con toda la información o mirando solo la carta.

## 2 jugadores

1,014 situaciones; 2000 iteraciones de 20,000 repartos; evaluado con otros 500,000 (74 s).

Convergencia (cuánto ganaría en promedio quien se desvíe del equilibrio, puntos por ronda): 100 it: 0.010, 500 it: 0.005, 1000 it: 0.003, 2000 it: 0.002.

| Posición de canto | 1 | 2 |
|---|---|---|
| Valor en el equilibrio | +4.75 | -4.75 |
| Ganancia por desviarse del equilibrio | 0.005 | 0.000 |
| Pérdida por mirar solo la carta (rivales en equilibrio) | — | — |
| Valor de la heurística (todos heurística) | +2.06 | -2.06 |
| Ganancia de la mejor respuesta a la heurística | 2.70 | 0.00 |
| ... mirando solo la carta | 2.70 | 0.00 |

Probabilidad de cantar 1 en el equilibrio, por carta y posición (promedio sobre el triunfo dado vuelta y los cantos anteriores, pesado por lo seguido que se da; el último solo cuenta cuando puede elegir):

| Carta | 1 | 2 |
|---|---|---|
| 2 | 0.97 | — |
| 3 | 0.99 | — |
| 4 | 0.99 | — |
| 5 | 1.00 | — |
| 6 | 1.00 | — |
| 7 | 1.00 | — |
| 8 | 1.00 | — |
| 9 | 1.00 | — |
| 10 | 1.00 | — |
| J | 1.00 | — |
| Q | 1.00 | — |
| K | 1.00 | — |
| A | 1.00 | — |
| triunfo 2 | 1.00 | — |
| triunfo 3 | 1.00 | — |
| triunfo 4 | 1.00 | — |
| triunfo 5 | 1.00 | — |
| triunfo 6 | 1.00 | — |
| triunfo 7 | 1.00 | — |
| triunfo 8 | 1.00 | — |
| triunfo 9 | 1.00 | — |
| triunfo 10 | 1.00 | — |
| triunfo J | 1.00 | — |
| triunfo Q | 1.00 | — |
| triunfo K | 1.00 | — |
| triunfo A | 1.00 | — |

## 3 jugadores

2,366 situaciones; 2000 iteraciones de 20,000 repartos; evaluado con otros 500,000 (134 s).

Convergencia (cuánto ganaría en promedio quien se desvíe del equilibrio, puntos por ronda): 100 it: 0.016, 500 it: 0.010, 1000 it: 0.008, 2000 it: 0.007.

| Posición de canto | 1 | 2 | 3 |
|---|---|---|---|
| Valor en el equilibrio | +0.55 | +3.49 | -4.04 |
| Ganancia por desviarse del equilibrio | 0.016 | 0.005 | 0.001 |
| Pérdida por mirar solo la carta (rivales en equilibrio) | — | 0.129 | -0.001 |
| Valor de la heurística (todos heurística) | +0.32 | +2.99 | -3.30 |
| Ganancia de la mejor respuesta a la heurística | 0.20 | 0.60 | 0.11 |
| ... mirando solo la carta | 0.20 | 0.36 | 0.11 |

Probabilidad de cantar 1 en el equilibrio, por carta y posición (promedio sobre el triunfo dado vuelta y los cantos anteriores, pesado por lo seguido que se da; el último solo cuenta cuando puede elegir):

| Carta | 1 | 2 | 3 |
|---|---|---|---|
| 2 | 0.00 | 0.00 | 0.00 |
| 3 | 0.00 | 0.00 | 0.00 |
| 4 | 0.00 | 0.00 | 0.00 |
| 5 | 0.00 | 0.00 | 0.00 |
| 6 | 0.00 | 0.00 | 0.00 |
| 7 | 0.01 | 0.00 | 0.00 |
| 8 | 0.02 | 0.00 | 0.00 |
| 9 | 0.26 | 0.00 | 0.00 |
| 10 | 0.97 | 0.00 | 0.00 |
| J | 0.99 | 0.00 | 0.00 |
| Q | 1.00 | 0.00 | 0.00 |
| K | 1.00 | 0.00 | 0.00 |
| A | 1.00 | 0.00 | 0.00 |
| triunfo 2 | 1.00 | 0.46 | 0.00 |
| triunfo 3 | 1.00 | 0.46 | 0.00 |
| triunfo 4 | 1.00 | 0.47 | 0.00 |
| triunfo 5 | 1.00 | 0.46 | 0.00 |
| triunfo 6 | 1.00 | 0.48 | 0.00 |
| triunfo 7 | 1.00 | 0.68 | 0.00 |
| triunfo 8 | 1.00 | 0.92 | 0.00 |
| triunfo 9 | 1.00 | 0.99 | 0.01 |
| triunfo 10 | 1.00 | 1.00 | 0.04 |
| triunfo J | 1.00 | 1.00 | 0.32 |
| triunfo Q | 1.00 | 1.00 | 0.97 |
| triunfo K | 1.00 | 1.00 | 0.99 |
| triunfo A | 1.00 | 1.00 | 1.00 |

## 4 jugadores

5,070 situaciones; 2000 iteraciones de 20,000 repartos; evaluado con otros 500,000 (170 s).

Convergencia (cuánto ganaría en promedio quien se desvíe del equilibrio, puntos por ronda): 100 it: 0.012, 500 it: 0.007, 1000 it: 0.006, 2000 it: 0.004.

| Posición de canto | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| Valor en el equilibrio | +0.09 | +2.26 | +2.41 | -4.76 |
| Ganancia por desviarse del equilibrio | 0.010 | 0.002 | 0.004 | 0.001 |
| Pérdida por mirar solo la carta (rivales en equilibrio) | — | 0.234 | 0.421 | -0.000 |
| Valor de la heurística (todos heurística) | +0.05 | +1.99 | +1.98 | -4.02 |
| Ganancia de la mejor respuesta a la heurística | 0.00 | 0.24 | 0.43 | 0.14 |
| ... mirando solo la carta | 0.00 | 0.00 | 0.00 | 0.14 |

Probabilidad de cantar 1 en el equilibrio, por carta y posición (promedio sobre el triunfo dado vuelta y los cantos anteriores, pesado por lo seguido que se da; el último solo cuenta cuando puede elegir):

| Carta | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| 2 | 0.00 | 0.00 | 0.00 | 0.00 |
| 3 | 0.00 | 0.00 | 0.00 | 0.00 |
| 4 | 0.00 | 0.00 | 0.00 | 0.00 |
| 5 | 0.00 | 0.00 | 0.00 | 0.00 |
| 6 | 0.00 | 0.00 | 0.00 | 0.00 |
| 7 | 0.00 | 0.00 | 0.00 | 0.00 |
| 8 | 0.00 | 0.00 | 0.00 | 0.00 |
| 9 | 0.00 | 0.00 | 0.00 | 0.00 |
| 10 | 0.00 | 0.00 | 0.00 | 0.00 |
| J | 0.00 | 0.00 | 0.00 | 0.00 |
| Q | 0.01 | 0.00 | 0.00 | 0.00 |
| K | 0.02 | 0.00 | 0.00 | 0.00 |
| A | 0.54 | 0.00 | 0.00 | 0.00 |
| triunfo 2 | 0.97 | 0.74 | 0.57 | 0.00 |
| triunfo 3 | 0.99 | 0.75 | 0.59 | 0.00 |
| triunfo 4 | 0.99 | 0.74 | 0.58 | 0.00 |
| triunfo 5 | 1.00 | 0.75 | 0.58 | 0.00 |
| triunfo 6 | 1.00 | 0.75 | 0.59 | 0.00 |
| triunfo 7 | 1.00 | 0.74 | 0.59 | 0.00 |
| triunfo 8 | 1.00 | 0.75 | 0.59 | 0.00 |
| triunfo 9 | 1.00 | 0.76 | 0.64 | 0.00 |
| triunfo 10 | 1.00 | 0.84 | 0.81 | 0.00 |
| triunfo J | 1.00 | 0.99 | 0.98 | 0.02 |
| triunfo Q | 1.00 | 1.00 | 0.99 | 0.18 |
| triunfo K | 1.00 | 1.00 | 1.00 | 0.94 |
| triunfo A | 1.00 | 1.00 | 1.00 | 1.00 |

## 5 jugadores

10,478 situaciones; 2000 iteraciones de 20,000 repartos; evaluado con otros 500,000 (297 s).

Convergencia (cuánto ganaría en promedio quien se desvíe del equilibrio, puntos por ronda): 100 it: 0.012, 500 it: 0.007, 1000 it: 0.005, 2000 it: 0.004.

| Posición de canto | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| Valor en el equilibrio | +0.24 | +1.53 | +1.71 | +1.84 | -5.33 |
| Ganancia por desviarse del equilibrio | 0.006 | 0.005 | 0.003 | 0.006 | 0.001 |
| Pérdida por mirar solo la carta (rivales en equilibrio) | — | 0.172 | 0.362 | 0.489 | -0.000 |
| Valor de la heurística (todos heurística) | +0.16 | +1.30 | +1.31 | +1.30 | -4.07 |
| Ganancia de la mejor respuesta a la heurística | 0.11 | 0.24 | 0.41 | 0.55 | 0.25 |
| ... mirando solo la carta | 0.11 | 0.11 | 0.10 | 0.11 | 0.25 |

Probabilidad de cantar 1 en el equilibrio, por carta y posición (promedio sobre el triunfo dado vuelta y los cantos anteriores, pesado por lo seguido que se da; el último solo cuenta cuando puede elegir):

| Carta | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| 2 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 3 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 4 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 5 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 6 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 7 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 8 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 9 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 10 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| J | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Q | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| K | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| A | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| triunfo 2 | 0.01 | 0.45 | 0.64 | 0.49 | 0.00 |
| triunfo 3 | 0.02 | 0.77 | 0.64 | 0.49 | 0.00 |
| triunfo 4 | 0.12 | 0.81 | 0.64 | 0.50 | 0.00 |
| triunfo 5 | 0.77 | 0.82 | 0.66 | 0.51 | 0.00 |
| triunfo 6 | 0.98 | 0.83 | 0.66 | 0.51 | 0.00 |
| triunfo 7 | 0.99 | 0.84 | 0.66 | 0.51 | 0.00 |
| triunfo 8 | 1.00 | 0.84 | 0.66 | 0.51 | 0.00 |
| triunfo 9 | 1.00 | 0.83 | 0.66 | 0.54 | 0.00 |
| triunfo 10 | 1.00 | 0.85 | 0.73 | 0.69 | 0.00 |
| triunfo J | 1.00 | 0.88 | 0.88 | 0.90 | 0.01 |
| triunfo Q | 1.00 | 1.00 | 0.99 | 0.97 | 0.16 |
| triunfo K | 1.00 | 1.00 | 0.99 | 0.99 | 0.75 |
| triunfo A | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

## 6 jugadores

21,294 situaciones; 2000 iteraciones de 20,000 repartos; evaluado con otros 500,000 (567 s).

Convergencia (cuánto ganaría en promedio quien se desvíe del equilibrio, puntos por ronda): 100 it: 0.013, 500 it: 0.007, 1000 it: 0.006, 2000 it: 0.005.

| Posición de canto | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| Valor en el equilibrio | +0.34 | +1.13 | +1.24 | +1.37 | +1.50 | -5.59 |
| Ganancia por desviarse del equilibrio | 0.005 | 0.005 | 0.006 | 0.004 | 0.007 | 0.002 |
| Pérdida por mirar solo la carta (rivales en equilibrio) | — | 0.095 | 0.216 | 0.358 | 0.483 | -0.000 |
| Valor de la heurística (todos heurística) | +0.07 | +0.77 | +0.78 | +0.77 | +0.78 | -3.17 |
| Ganancia de la mejor respuesta a la heurística | 0.21 | 0.31 | 0.39 | 0.49 | 0.60 | 0.01 |
| ... mirando solo la carta | 0.21 | 0.22 | 0.21 | 0.21 | 0.20 | 0.01 |

Probabilidad de cantar 1 en el equilibrio, por carta y posición (promedio sobre el triunfo dado vuelta y los cantos anteriores, pesado por lo seguido que se da; el último solo cuenta cuando puede elegir):

| Carta | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| 2 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 3 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 4 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 5 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 6 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 7 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 8 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 9 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 10 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| J | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Q | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| K | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| A | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| triunfo 2 | 0.00 | 0.00 | 0.02 | 0.54 | 0.43 | 0.00 |
| triunfo 3 | 0.00 | 0.01 | 0.15 | 0.55 | 0.43 | 0.00 |
| triunfo 4 | 0.01 | 0.02 | 0.60 | 0.57 | 0.44 | 0.00 |
| triunfo 5 | 0.01 | 0.57 | 0.70 | 0.57 | 0.45 | 0.00 |
| triunfo 6 | 0.09 | 0.82 | 0.72 | 0.59 | 0.45 | 0.00 |
| triunfo 7 | 0.62 | 0.87 | 0.73 | 0.59 | 0.46 | 0.00 |
| triunfo 8 | 0.98 | 0.87 | 0.73 | 0.61 | 0.46 | 0.00 |
| triunfo 9 | 0.99 | 0.87 | 0.74 | 0.60 | 0.51 | 0.00 |
| triunfo 10 | 1.00 | 0.88 | 0.74 | 0.65 | 0.65 | 0.00 |
| triunfo J | 1.00 | 0.90 | 0.81 | 0.79 | 0.84 | 0.01 |
| triunfo Q | 1.00 | 0.98 | 0.98 | 0.98 | 0.96 | 0.13 |
| triunfo K | 1.00 | 1.00 | 0.99 | 0.99 | 0.98 | 0.62 |
| triunfo A | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

## 7 jugadores

42,926 situaciones; 2000 iteraciones de 20,000 repartos; evaluado con otros 500,000 (1228 s).

Convergencia (cuánto ganaría en promedio quien se desvíe del equilibrio, puntos por ronda): 100 it: 0.014, 500 it: 0.008, 1000 it: 0.006, 2000 it: 0.005.

| Posición de canto | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| Valor en el equilibrio | +0.41 | +0.90 | +0.98 | +1.07 | +1.16 | +1.26 | -5.78 |
| Ganancia por desviarse del equilibrio | 0.004 | 0.004 | 0.005 | 0.007 | 0.007 | 0.007 | 0.002 |
| Pérdida por mirar solo la carta (rivales en equilibrio) | — | 0.065 | 0.140 | 0.235 | 0.334 | 0.437 | 0.000 |
| Valor de la heurística (todos heurística) | +0.23 | +0.65 | +0.66 | +0.66 | +0.66 | +0.66 | -3.53 |
| Ganancia de la mejor respuesta a la heurística | 0.10 | 0.18 | 0.24 | 0.33 | 0.41 | 0.48 | 0.01 |
| ... mirando solo la carta | 0.10 | 0.11 | 0.11 | 0.11 | 0.11 | 0.11 | 0.01 |

Probabilidad de cantar 1 en el equilibrio, por carta y posición (promedio sobre el triunfo dado vuelta y los cantos anteriores, pesado por lo seguido que se da; el último solo cuenta cuando puede elegir):

| Carta | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| 2 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 3 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 4 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 5 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 6 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 7 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 8 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 9 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 10 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| J | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Q | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| K | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| A | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| triunfo 2 | 0.00 | 0.00 | 0.00 | 0.00 | 0.02 | 0.38 | 0.00 |
| triunfo 3 | 0.00 | 0.00 | 0.00 | 0.01 | 0.43 | 0.39 | 0.00 |
| triunfo 4 | 0.00 | 0.00 | 0.01 | 0.26 | 0.49 | 0.40 | 0.00 |
| triunfo 5 | 0.00 | 0.01 | 0.05 | 0.53 | 0.51 | 0.41 | 0.00 |
| triunfo 6 | 0.01 | 0.03 | 0.51 | 0.65 | 0.53 | 0.42 | 0.00 |
| triunfo 7 | 0.05 | 0.50 | 0.75 | 0.65 | 0.53 | 0.43 | 0.00 |
| triunfo 8 | 0.52 | 0.87 | 0.77 | 0.67 | 0.55 | 0.43 | 0.00 |
| triunfo 9 | 0.98 | 0.89 | 0.79 | 0.67 | 0.55 | 0.46 | 0.00 |
| triunfo 10 | 1.00 | 0.89 | 0.79 | 0.68 | 0.59 | 0.54 | 0.00 |
| triunfo J | 1.00 | 0.90 | 0.82 | 0.73 | 0.72 | 0.73 | 0.01 |
| triunfo Q | 1.00 | 0.92 | 0.90 | 0.91 | 0.91 | 0.92 | 0.07 |
| triunfo K | 1.00 | 1.00 | 1.00 | 0.99 | 0.98 | 0.98 | 0.44 |
| triunfo A | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
