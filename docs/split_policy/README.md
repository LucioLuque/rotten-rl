# Canto y juego en redes separadas

Etapa B del plan del 16/11 (`docs/plan.md`). Hasta ahora una sola red de política decide los cantos y
las cartas: las dos decisiones comparten todas las capas ocultas y solo la última capa tiene salidas
distintas (8 cantos y 52 cartas). Los trabajos de juegos parecidos separan las dos decisiones en
redes distintas (Wizard, Judgement, AlphaDou; `docs/bibliografia.md`): el canto es una predicción de
cuántas bazas se van a ganar, una decisión de un paso muy distinta de jugar una carta, y según
Edelkamp es la que más pesa en el resultado. Con una red compartida, lo que aprende para una puede
estorbar a la otra.

## Qué es

`--split_policy` en `scripts/jax_engine/train_ppo.py`: la política son **dos redes** con `--hidden`
cada una (808 → 512 → 512), una con los 8 cantos y otra con las 52 cartas. Las dos ven la observación
entera. El crítico (la red de valor) sigue siendo uno solo, compartido.

Cómo está hecho: las dos redes se guardan juntas como una sola red del doble de ancho (1024, 1024) con
los bloques cruzados en cero, y esos ceros no reciben nunca gradiente (`split_policy_mask` en
`rotten_rl/jax_engine/ppo.py`). Es exactamente la función de dos redes separadas, y todo lo que carga
una política (evaluación, torneos, `play.py`, rivales de la liga) la ve como una red común, sin
cambios. El costo es hacer cuentas con los ceros: la política hace ~1,9 veces las cuentas de la actual
(estimado, sin medir; si la idea sirve, se puede programar con dos redes de verdad y recuperar parte).

Tests (`tests/test_jax_ppo.py`, pasan en la -14): los cantos dependen solo de la red de canto y las
cartas solo de la de juego; después de entrenar, los bloques cruzados siguen exactamente en cero; se
exporta como una red común del doble de ancho. Prueba de punta a punta en la -14 (entrenar, evaluar,
torneo).

## Experimento

| Corrida | Qué | Contra |
|---|---|---|
| `h20` | 2 jugadores, redes separadas, de cero a 24M, 4 semillas, receta actual (entropía 0,06) | `s20full`: la misma receta y partidas con la red compartida |

Torneo `t_h20`: heurística, `s20full`, `h20` y `s20` (36M en tramos). Si `h20` le gana a `s20full`
por más que el ruido (~0,3 por partida), pruebo con 3 y 7 jugadores. Tiempo estimado (sin medir): ~3 h.
En la cola de la -14 (`queue26`) el 17/11, después de `s20full`.

## Resultados

Pendiente.
