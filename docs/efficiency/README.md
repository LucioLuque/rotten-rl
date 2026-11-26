# Eficiencia: una red más chica

Etapa F del plan del 16/11 (`docs/plan.md`): cuánta velocidad se gana con una red más chica sin perder
calidad. Importa para jugar rápido y, sobre todo, si después se busca al jugar (PIMC, etapa G), que
evalúa la red muchas veces por jugada.

## Experimento (26/11)

Especialistas con formatos con dos capas de 256 en vez de 512 (`--hidden 256,256`), misma receta y 24M:
`w20` (2 jugadores) contra `m20` y `w72` (7 jugadores, con `--obs_bf16`) contra `m72`, torneo con
formatos. Vale si pierde menos de 0,3 por partida. Cola `queue42` en la -14. Si sirve, sigo con 3 a 6.

## Resultados

Pendiente.
