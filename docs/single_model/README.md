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

### Especialistas de 4 a 6 con formatos

**`m40` (21/11, 4 jugadores, 1.523 partidas/s por semilla; torneos de 1000 repartos por cruce):**

| | Diferencia por partida | Win rate (empate 25 %) | Cruces positivos |
|---|---|---|---|
| Formatos: `m40` contra `f40` | **+3,62** | 29,9 % | 16/16 |
| Formatos: `f40` contra `m40` | −4,18 | 20,6 % | 0/16 |
| Escalera vieja: `m40` contra `f40` | −0,02 | 24,9 % | 8/16 |
| Escalera vieja: `f40` contra `m40` | +0,05 | 25,2 % | 10/16 |

Contra la heurística con formatos, `m40` 58,8 % y `f40` 50,7 %; en la escalera vieja, 52,7 y 52,5 %. Con
4 jugadores aprender los formatos no cuesta nada en la escalera vieja (empate). Curvas: entropía más alta
(0,24 contra 0,22) y cantos exactos y varianza explicada un punto más bajos, como con 2 y 3.

**`m50` (21/11, 5 jugadores, 1.219 partidas/s por semilla):**

| | Diferencia por partida | Win rate (empate 20 %) | Cruces positivos |
|---|---|---|---|
| Formatos: `m50` contra `f50` | **+4,02** | 25,4 % | 16/16 |
| Formatos: `f50` contra `m50` | −4,52 | 15,1 % | 0/16 |
| Escalera vieja: `m50` contra `f50` | −0,10 | 19,8 % | 5/16 |
| Escalera vieja: `f50` contra `m50` | +0,07 | 20,3 % | 11/16 |

Contra la heurística con formatos, `m50` 54,2 % y `f50` 43,6 %; en la escalera vieja, 49,9 y 49,8 %.
Panel de 5 (con formatos): `m50` +2,78 de promedio sin la heurística (+4,66 contra `f50@6M`, +3,84 contra
`f50`), `f50` −1,44. Curvas como las de los otros: entropía más alta (0,26 contra 0,24), cantos exactos
y varianza explicada un poco más bajos.
