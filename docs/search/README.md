# Búsqueda: PIMC

Etapa A del plan del 16/11 (`docs/plan.md`): medir mejor. La heurística ya no distingue entre agentes
buenos y no sabemos cuántas de las partidas que pierde el agente son mala suerte y cuántas errores.
PIMC sirve para las dos cosas: como **rival más fuerte** y como **analizador de decisiones**. Más
adelante (etapa G), como jugador.

## Qué es PIMC

*Perfect Information Monte Carlo*. En cada decisión, el jugador no ve las cartas de los demás. PIMC:

1. **Inventa** muchos repartos posibles de las cartas que no ve, coherentes con lo que sabe: sus
   cartas, las que ya salieron, el triunfo, cuántas cartas tiene cada uno y de qué palos se sabe que
   no tiene (porque no siguió el palo).
2. Para cada reparto inventado y cada jugada legal, **simula el resto de la ronda** con la red de PPO
   en todos los asientos.
3. Juega la jugada que **en promedio** le dio más puntos.

Es lo que usan los mejores programas de bridge y de Skat. Límites conocidos: no deduce nada de los
cantos (solo usa los palos que faltan) y en cada reparto inventado juega como si ese fuera el real
("strategy fusion"). En Wizard la búsqueda dentro del agente no lo mejoró (`docs/bibliografia.md`),
por eso primero va como rival y analizador.

## Código

- `rotten_rl/search/pimc.py`: `sample_hands` (reparto coherente; rechaza los que le dan a alguien un
  palo que se sabe que no tiene, hasta 32 intentos), `rollout_round` (juega hasta el final de la ronda
  con la red), `action_values` (valor de cada jugada legal, como máximo 8) y `pimc_action`. Todo en
  JAX, vectorizado sobre repartos y jugadas.
- `tests/test_search_pimc.py`: los repartos respetan lo que sabe el jugador, la simulación coincide
  con un loop común y los valores cubren exactamente las jugadas legales (6 tests, pasan en la -14).
- `scripts/search/analyze_decisions.py`:
  - `--mode analyze`: el agente juega contra sí mismo y en cada decisión PIMC estima cuánto valía
    cada jugada; reporta cuánto deja el agente contra la mejor según PIMC, por fase (canto o juego) y
    tamaño de mano, y cuánto coinciden.
  - `--mode play`: PIMC en un asiento y la red en los demás, contra la red en ese asiento con los
    mismos repartos: ¿PIMC le gana a la red?

- `bid_log_likelihood` (opción `infer_bids`): pesa cada reparto inventado por la probabilidad de que,
  con esas cartas, los rivales hubieran cantado lo que cantaron (según la red). Es la inferencia de
  los programas de Skat.

## Resultados (16/11, `s20`, 2 jugadores, GPU de la -14)

PIMC jugando en un asiento contra la red en los demás, contra la red en ese asiento con los mismos
repartos (256 por asiento; puntos por partida):

| PIMC decide | Sin pesar por cantos (32 repartos) | Pesando por cantos (64 repartos) |
|---|---|---|
| Solo las cartas (canta la red) | −5,77 ± 0,88 | −3,37 ± 0,79 |
| Solo el canto (juega la red) | −5,52 ± 1,14 | — |
| Canto y cartas | −12,72 ± 2,64 (64 repartos por asiento) | −5,16 ± 1,19 |

**PIMC guiado por la red juega peor que la propia red**, como en Wizard. Pesar por los cantos lo
mejora mucho pero no alcanza. Así que no sirve como rival más fuerte, y su medida de errores tampoco
es confiable (un juez más débil que el agente). El análisis (`--mode analyze`, 64 partidas, 32
repartos, diferencia con dos grupos de repartos) dio −0,03 puntos de media: PIMC no encuentra
jugadas mejores que las del agente. Tiempo: 9 a 45 segundos por corrida con 64 a 256 partidas.

Ideas si se retoma: más repartos y simulaciones que muestreen en vez de jugar la acción más
probable; pesar también por las cartas jugadas; usar PIMC solo en las últimas bazas, donde las manos
están muy restringidas.
