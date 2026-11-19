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

**`t_h20` (18/11, 1000 repartos por cruce): las redes separadas no mejoran.** `h20` contra `s20full`
**−0,28 por partida, 49,5 % de win rate, 4/16 cruces positivos**; contra `s20` (36M) −0,38 (0/16);
contra la heurística 82,4 % (`s20full` 82,3 %). Está en el borde del ruido (~0,3), pero para el lado
malo. Métricas de entrenamiento: win rate y cantos exactos contra la heurística, entropía y varianza
explicada iguales a `s20full`; la KL y los pasos recortados un poco más altos al principio (cada red
cambia más por actualización), y nada más. Velocidad: 2.418 partidas/s por semilla (la red compartida,
~4.000): 2 h 44 min para 24M.

Conclusión: con PPO y esta observación, compartir las capas entre canto y juego no estorba. Como no
ganó con 2 jugadores, no lo pruebo con 3 y 7 (era la condición).

Igual, el 18/11 decidí probarlo con 3 jugadores, por si con más rivales el canto pesa distinto: `h30`
(3 jugadores, redes separadas, de cero a 24M, ~4 h estimadas) contra `s30full`, torneo `t_h30`. En la
cola de la -14 antes de `f40`.

**`t_h30` (19/11): con 3 jugadores, empate.** `h30` contra `s30full` **−0,03 por partida, 33,3 % (empate
33,3 %), 8/16 cruces**; contra `s30` −0,12 (3/16); contra la heurística 61,9 % (`s30full` 62,1 %). Las
curvas de entrenamiento son prácticamente las mismas. (`h30` se cortó dos veces: a las 12:25 del 18/11
por el cierre del lab y la -14 no volvió hasta las 16:00 del 19/11; se retomó con `--resume` desde
20,8M, que equivale a no cortar, `docs/resume/`.)

**Conclusión: separar canto y juego no mejora ni con 2 ni con 3 jugadores.** Queda cerrado.
