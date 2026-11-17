# 4, 5 y 6 jugadores

Etapa C del plan del 16/11 (`docs/plan.md`). La etapa B casi no cambió la receta (la tarea auxiliar y
la liga no mejoraron), así que entreno 4, 5 y 6 jugadores con la receta actual, **de cero y con el
horizonte completo de una vez** (24M partidas), que es como conviene entrenar un agente nuevo (sin el
bajón de retomar, `docs/resume/`).

| Corrida | Jugadores | Entropía | Partidas | Tiempo estimado (sin medir) |
|---|---|---|---|---|
| `f40` | 4 | 0,06 | 24M, 4 semillas | ~3 h |
| `f50` | 5 | 0,08 | 24M, 4 semillas | ~3 h 30 min |
| `f60` | 6 | 0,10 | 24M, 4 semillas | ~4 h 30 min |

Resto de la receta: tasa 3e-4 que baja a 0, 2 épocas, minilote 1024, lote de 512 partidas. La
entropía de 4 a 6 no está barrida: con 2 y 3 jugadores la mejor fue 0,06 y con 7, 0,1; para 4, 5 y 6
la interpolo.

En la cola de la -14 desde el 17/11 (al final de `queue26`, después de `s30full`).

## Resultados

Pendiente.
