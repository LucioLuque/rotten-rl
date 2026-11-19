# 4, 5 y 6 jugadores

Etapa C del plan del 16/11 (`docs/plan.md`). Empieza el 18/11, cuando terminó la etapa B sin cambios
en la receta (tarea auxiliar, liga con varios asientos y redes separadas no mejoraron). Entreno **de cero y con el horizonte completo de una vez** (24M partidas), que es como
conviene entrenar un agente nuevo (sin el bajón de retomar, `docs/resume/`).

| Corrida | Jugadores | Entropía | Partidas | Tiempo estimado (sin medir) |
|---|---|---|---|---|
| `f40` | 4 | 0,06 | 24M, 4 semillas | ~3 h |
| `f50` | 5 | 0,08 | 24M, 4 semillas | ~3 h 30 min |
| `f60` | 6 | 0,10 | 24M, 4 semillas | ~4 h 30 min |

Resto de la receta: tasa 3e-4 que baja a 0, 2 épocas, minilote 1024, lote de 512 partidas. La
entropía de 4 a 6 no está barrida: con 2 y 3 jugadores la mejor fue 0,06 y con 7, 0,1; para 4, 5 y 6
la interpolo.

Antes de esto: `s20full` y `u72full` (2 y 7 jugadores de cero a 24M de una vez), porque los agentes
de 2 y 7 que tengo están hechos en tramos retomados.

## Resultados

**`f40` (4 jugadores, 19/11, 16:43 a 19:53 en la -14, 2.108 partidas/s por semilla):** a 24M le gana
a la heurística el 54 % de las partidas (empate 25 %, las 4 semillas entre 53,9 y 54,3 %), por +18,2
puntos por partida, con 68,6 % de cantos exactos. Queda entre 3 jugadores (`s30full`: 62,4 %, +20,3) y
7 (`u72full`: 36,3 %, +14,5), como se espera al haber más rivales. Las curvas son sanas: el win rate
sigue subiendo despacio hasta el final (49 % a 1M, 54 % a 24M), la varianza explicada igual que con 3
jugadores, la KL y los pasos recortados bajan con el lr y la entropía se estabiliza en ~0,22, un poco
más baja que con 3 y 7 jugadores (0,25). Contra versiones anteriores de sí misma, ~0. Sin torneo
todavía.
