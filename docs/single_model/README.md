# Un solo modelo de 2 a 7 jugadores

Etapa E del plan del 16/11 (`docs/plan.md`): una sola red para cualquier cantidad de jugadores, contra
los especialistas de cada una. Vale si pierde poco (por ejemplo, menos de 0,3 por partida) e incluye las
variantes de reglas (los formatos de partida, `docs/rule_variants/`).

## Cómo está hecho

`train_ppo.py --players_mix 2,3,4,5,6,7 --formats`: una sola red (la observación ya tiene lugar para 7
jugadores, con los que no están en cero) y las actualizaciones se turnan entre las cantidades de
jugadores: cada una juega 512 partidas de esa cantidad (las mesas de distinto tamaño no pueden ir en el
mismo lote), con sus partidas en curso aparte. La entropía, por cantidad de jugadores con
`--ent_by_players` (la receta: 0,06 con 2 a 4, 0,08 con 5, 0,1 con 6 y 7). Se evalúa contra la
heurística y contra la versión anterior con cada cantidad de jugadores (`eval_heuristic<n>p`,
`eval_past<n>p`). Los torneos y el panel aceptan la red con cualquier cantidad de jugadores de su mezcla.
Probado en la -14 de punta a punta (entrenar, evaluar, retomar con `--resume`, torneo).

## Experimento (cola `queue35`, agregada a la de la -14 el 20/11, después de `m72`)

| Corrida | Qué | Estimado (sin medir) |
|---|---|---|
| `m40`, `m50`, `m60` | especialistas de 4, 5 y 6 con formatos, 24M, con torneos contra `f40`/`f50`/`f60` (con formatos y en la escalera vieja) | ~16 h las tres |
| `uni27` | 2 a 7 jugadores con formatos, 24M por cantidad de jugadores (144M), 4 semillas | ~29 h |
| torneos | con cada cantidad de jugadores, `uni27` contra el especialista con formatos (`m20` … `m72`) y el del juego normal (`s20full`, `s30full`, `f40` … `f60`, `u72full`), con formatos y en la escalera vieja | minutos |
| panel | `uni27` y el especialista contra el panel de cada cantidad de jugadores, con formatos | minutos |

Los especialistas de 4 a 6 van antes (lo pidió el 20/11) para que el torneo de `uni27` con 4 a 6 sea
contra rivales que también conocen los formatos, y para sumarlos al panel. Con 24M por cantidad de
jugadores, `uni27` ve lo mismo que cada especialista: si pierde, no es por falta de partidas. `uni27`
no entra antes del corte del lunes 23/11 a las 06:55: se corta y se retoma con `--resume` en la ventana
siguiente.

## Resultados

Pendiente.
