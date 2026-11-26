# Eficiencia: una red más chica

Etapa F del plan del 16/11 (`docs/plan.md`): cuánta velocidad se gana con una red más chica sin perder
calidad. Importa para jugar rápido y, sobre todo, si después se busca al jugar (PIMC, etapa G), que
evalúa la red muchas veces por jugada.

## Experimento (26/11)

Especialistas con formatos con dos capas de 256 en vez de 512 (`--hidden 256,256`), misma receta y 24M:
`w20` (2 jugadores) contra `m20` y `w72` (7 jugadores, con `--obs_bf16`) contra `m72`, torneo con
formatos. Vale si pierde menos de 0,3 por partida. Cola `queue42` en la -14. Si sirve, sigo con 3 a 6.

## Resultados

**`w20` (26/11, 2 jugadores, red de 256): empata con `m20` y entrena 1,65 veces más rápido.** Torneo con
formatos (1000 repartos por cruce): `w20` contra `m20` **+0,13 por partida, 49,8 %, 11/16 cruces**
(empate, dentro del ruido); contra la heurística 86,5 % (`m20` 86,2 %). Curvas iguales (win rate,
cantos exactos, entropía, varianza explicada); la KL y los pasos recortados, más bajos (con menos
parámetros, cada actualización cambia menos la política). Velocidad: 4.850 partidas/s por semilla contra
2.940 de la red de 512 con el mismo código (`m20c`; `m20` va a 2.542 porque se entrenó con relleno).

Repetido con 5000 repartos por cruce (otra semilla de repartos, 26/11): `w20` contra `m20` **+0,13 por
partida** (error estándar ~0,05, aproximado: los 16 cruces comparten repartos), 50,3 %, 11/16;
contra la heurística 86,1 % y 85,9 %. La diferencia se mantiene y es más grande que el error, pero es
chica (menos de la mitad del umbral de 0,3 que usamos para "gana"): la red de 256 es igual o apenas
mejor con 2 jugadores, no peor.

**`w72` (26/11, 7 jugadores, red de 256, 1.641 partidas/s por semilla contra ~905 de `m72`, 1,8 veces más
rápido): empate.** Torneo con formatos: `w72` contra `m72` −0,02 (5/16), `m72` contra `w72` −0,06 (5/16);
contra la heurística 43,1 % y 42,8 %.

**Conclusión: la red de 256 alcanza con 2 y con 7 jugadores** (empata, y entrena 1,65 a 1,8 veces más
rápido; al jugar, cada evaluación cuesta ~4 veces menos). Siguiente (26/11, de noche): `w30` … `w60`, para
tener los seis especialistas de 256.
