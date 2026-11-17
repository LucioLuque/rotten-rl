# Cómo retomar un agente entrenado

Al retomar un agente ya entrenado (liga, explotadores, extender el horizonte) con la tasa al valor de
partida, juega peor durante buena parte del tramo nuevo y recién vuelve al nivel de antes hacia el
final: `s30` de 12M a 24M bajó de 63,1 % a 61,4 % contra la heurística al retomar y recién volvió a
63,5 % a las 24M; lo mismo en las ligas (`docs/robustness/`) y los explotadores con tasa 3e-4.

## Preguntas

1. **¿Cómo conviene retomar?** Estudio corto: retomar `s20` (36M), `s30` (24M) y `u72` (24M) con el
   calendario de seguir 12M más, pero cortando a las 2M (`--stop_games`), evaluando cada 250.000
   partidas contra la heurística y contra el agente de partida (`--eval_against`). Cuatro variantes
   (4 semillas cada una): **a** tasa 1e-4 (como hasta ahora), **b** 3e-5, **c** 1e-4 que sube de 0 en
   el primer millón (`--lr_warmup 1000000`), **d** 3e-4 (la de entrenar de cero).
2. **¿Conviene entrenar de cero con el horizonte completo o en tramos?** `s30full`: 3 jugadores de
   cero a 24M en una sola corrida, contra `s30` (12M y después extendido a 24M) en un torneo. Ya se
   había planteado el 3/11 con 7 jugadores en PyTorch (`e70s` extendida contra `e71` de cero): `e71` la
   igualaba antes de terminar, pero la comparación final no se hizo. También con 2 y 7 jugadores:
   `s20full` contra `s20` a 24M y `u72full` contra `u72` (los dos hechos en tramos de 12M).

Cola: `queue27.txt` (agregada al final de `queue26` en la -14 el 17/11). **Después, el 17/11, saqué
el estudio de la tasa (pregunta 1) y `s20full`/`u72full`:** para entrenar un agente nuevo conviene
directamente de cero con el horizonte completo, así que retomar solo importa para la liga (que solo
sirvió con 2 jugadores) y los explotadores (ya sé que 5e-5 anda). Queda solo `s30full`, que confirma
con un dato la regla de entrenar de cero de una vez. Si un día hace falta retomar (variantes de
reglas, modelo único), lo estudio en ese momento.

## Resultados

Pendiente.
