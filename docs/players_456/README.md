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

En la cola de la -14 desde el 17/11, al final de `queue26`. Van **después** de `s20full` y `u72full`
(2 y 7 jugadores de cero a 24M de una vez): los agentes de 2 y 7 que tengo están hechos en tramos
retomados, y si `s30full` le gana a `s30` hay que rehacerlos antes de seguir con 4, 5 y 6. Si
`s30full` empata con `s30`, los saco de la cola y 4, 5 y 6 se adelantan.

La cola no termina en una noche. A las 08:05 del 18/11 un timer del servidor
(`podrida-launch-0805d`) arma `queue30` con lo que quedó (`scripts/lab/remaining_queue.py`): retoma
la corrida que cortó el cierre del lab y sigue con las que no empezaron.

## Resultados

Pendiente.
