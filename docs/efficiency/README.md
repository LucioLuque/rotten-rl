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
