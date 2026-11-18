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
sirvió con 2 jugadores) y los explotadores (ya sé que 5e-5 anda). Queda `s30full`, que confirma
con un dato la regla de entrenar de cero de una vez. (Más tarde el 17/11 volví a poner `s20full` y
`u72full`, antes de 4, 5 y 6: si `s30full` le gana a `s30`, los agentes de 2 y 7 hay que rehacerlos
de cero; `docs/players_456/`.) Si un día hace falta retomar (variantes de
reglas, modelo único), lo estudio en ese momento.

## Resultados

Pendiente.

## Resultado de `s30full` (17/11)

Torneo `t_s30full` (3 jugadores): `s30full` (de cero a 24M de una vez) contra `s30` (12M + 12M retomado)
**−0,07 por partida, 33,4 % de win rate (empate 33,3 %), 7/16 cruces positivos: empate**. Los dos le
ganan +0,44 a `s30` a 12M (16/16). Contra la heurística, 62,1 % y 62,9 %. Las curvas lo explican: al
retomar `s30` con `--resume --games 24000000` la tasa se recalculó para el horizonte nuevo y volvió a
1,5e-4, así que desde 12M siguió exactamente el mismo camino que una corrida de cero a 24M; solo la
primera mitad fue distinta (tasa que bajaba a 0 en 12M), y eso no dejó diferencia. **Retomar con
`--resume` (mismo optimizador, sin bajón) es equivalente a entrenar de una vez**; el bajón es solo de
`--init_from` (optimizador nuevo).

`s20full` (2 jugadores, de cero a 24M; torneo `t_s20full`): contra `s20` a 24M **−0,09, 49,9 %, 7/16:
empate**; contra `s20` a 36M −0,09 (50,2 %); contra `s20` a 12M +0,88 (16/16); contra la heurística 82,3 %.
Lo mismo que con 3 jugadores. Una semilla (s1) quedó más floja que las otras tres.

`u72full` (7 jugadores, de cero a 24M; torneo `t_u72full`): contra `u72` a 24M **−0,11, 14,2 % (empate
14,3 %), 0/16**; contra `u72` a 12M +0,07 (11/16); contra la heurística 38,2 %. Queda un poco debajo de
`u72` y en todos los cruces, aunque por poco (cada cruce tiene ±0,09). Una semilla (s1) otra vez más
floja. Con 2 y 3 jugadores fue empate; con 7, `u72` (en tramos) apenas mejor. Ningún caso justifica
haber rehecho los agentes: para la etapa C alcanza con entrenar de una vez o retomar con `--resume`.
