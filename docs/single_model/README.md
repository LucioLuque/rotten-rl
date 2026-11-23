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

**`m60` (21/11, 6 jugadores, 1.023 partidas/s por semilla):**

| | Diferencia por partida | Win rate (empate 16,7 %) | Cruces positivos |
|---|---|---|---|
| Formatos: `m60` contra `f60` | **+5,04** | 23,5 % | 16/16 |
| Formatos: `f60` contra `m60` | −5,87 | 11,4 % | 0/16 |
| Escalera vieja: `m60` contra `f60` | −0,11 | 16,5 % | 1/16 |
| Escalera vieja: `f60` contra `m60` | +0,09 | 16,9 % | 12/16 |

Contra la heurística con formatos, `m60` 49,9 % y `f60` 36,2 %; en la escalera vieja, 42,8 y 42,5 %.
Panel de 6: `m60` +3,19 de promedio sin la heurística, `f60` −2,12. Curvas como las de los otros.

**Resumen de los especialistas con formatos** (contra el agente de la escalera vieja de la misma
cantidad de jugadores; con formatos / en la escalera vieja): 2: +3,34 / −0,34; 3: +3,31 / −0,24; 4:
+3,62 / −0,02; 5: +4,02 / −0,10; 6: +5,04 / −0,11; 7: +4,82 / −0,08. Con más jugadores, más ganancia con
formatos y casi ningún costo en la escalera vieja.

### `uni27`

Arrancó el 21/11 a las 22:54: 1.421 partidas/s por semilla (promedio de las seis mesas), así que 144M
tardan ~28 h (termina el lunes 23/11 ~03:00, antes del corte de las 06:55).

Después, `uni27s` (decidido el 22/11): la misma red con 24M en total (4M por cantidad de jugadores, la
sexta parte), para ver cuánto pierde con menos partidas; torneos con formatos contra el especialista y
`uni27` en cada mesa. Unas 5 h: lo que corte el cierre del lunes a las 06:55 se retoma a las 08:05.

**Resultado de `uni27` (23/11, 02:42; torneos de 1000 repartos por cruce, "uni27 contra X" = `uni27`
en un asiento y X en los demás):**

| Jugadores | Con formatos, contra el especialista `m` | Escalera vieja, contra el especialista `m` | Escalera vieja, contra el del juego normal | Panel (promedio sin heurística): `uni27` / `m` |
|---|---|---|---|---|
| 2 | **−4,24** (0/16) | **−3,69** (0/16) | **−4,03** contra `s20full` (0/16) | −1,07 / +2,66 |
| 3 | **−1,07** (0/16) | −1,17 (0/16) | −1,09 contra `s30full` (0/16) | +1,71 / +2,59 |
| 4 | −0,34 (0/16) | −0,21 (0/16) | −0,21 contra `f40` (1/16) | +2,35 / +2,66 |
| 5 | −0,01 (9/16) | −0,00 (8/16) | −0,13 contra `f50` (4/16) | +2,75 / +2,78 |
| 6 | −0,08 (4/16) | −0,10 (3/16) | −0,19 contra `f60` (1/16) | +3,04 / +3,19 |
| 7 | −0,30 (0/16) | −0,24 (2/16) | −0,27 contra `u72full` (0/16) | +3,30 / +3,51 |

Contra el agente del juego normal con formatos, `uni27` gana por lo mismo que el especialista (+2,5 a
+4,8 con 3 a 7 jugadores); con 2, apenas −0,47 contra `s20full`.

**Con 5 y 6 jugadores la red única empata con el especialista; con 4 y 7 pierde ~0,3 (en el límite que
pusimos); con 3 pierde ~1 y con 2, ~4 por partida.** Contra la heurística casi no se ve (con 2, 84,2 %
contra 86,3 % de `m20`; con 3 a 7, lo mismo que el especialista): lo que pierde es contra un rival que
juega bien. En las curvas contra la heurística, `uni27` aprende más rápido al principio (lo que aprende
en una mesa le sirve en las otras) pero con 2 jugadores se estanca abajo del especialista desde ~5M.

Por qué con 2 y 3, probablemente: cada actualización hace tantos pasos de gradiente como decisiones
tiene (512 partidas por actualización, pero una partida de 2 jugadores tiene ~145 decisiones y una de 7,
~507). En cada vuelta de las seis mesas, 2 jugadores es ~6 % de los pasos de gradiente y 7 jugadores
~22 %: la red se ajusta mucho más a las mesas grandes, y lo que aprende ahí le pisa lo de 2 jugadores,
que es el juego más distinto (el segundo en cantar está casi obligado, todo es duelo). Sin medir.

Qué probaría para arreglarlo (no lanzado): (a) repartir las actualizaciones para que cada mesa tenga la
misma cantidad de pasos de gradiente (más actualizaciones de 2 y 3 jugadores); (b) una red más grande
(1024); (c) una cabeza de salida por cantidad de jugadores. Mientras tanto, la conclusión: una sola red
sirve para 4 a 7 jugadores; para 2 y 3, todavía no.

**`uni27s` (23/11, la misma red con 24M en total, 4M por mesa; torneos con formatos):**

| Jugadores | `uni27s` contra el especialista | `uni27s` contra `uni27` (144M) | (`uni27` contra el especialista) |
|---|---|---|---|
| 2 | −5,25 (0/16) | −1,16 (0/16) | −4,24 |
| 3 | −1,33 (0/16) | −0,46 (0/16) | −1,07 |
| 4 | −0,54 (0/16) | −0,22 (1/16) | −0,34 |
| 5 | −0,21 (0/16) | −0,17 (3/16) | −0,01 |
| 6 | −0,19 (0/16) | −0,13 (3/16) | −0,08 |
| 7 | −0,36 (0/16) | −0,09 (4/16) | −0,30 |

Con la sexta parte de las partidas, la red única pierde poco contra la de 144M (0,1 a 0,5 con 3 a 7
jugadores; 1,2 con 2). La diferencia con el especialista en 2 y 3 jugadores no es por falta de
partidas: está casi igual con 24M que con 144M.

Probando (a) el 23/11: `--mix_balance steps` (la próxima actualización va a la mesa con menos decisiones
entrenadas hasta ahora, así cada una tiene los mismos pasos de gradiente; las mesas chicas juegan más
partidas: de 24M, ~31 % de 2 jugadores y ~9 % de 7). `uni27b`, 24M en total, contra `uni27s` (24M con
turnos iguales), `uni27` y los especialistas. El 23/11 es feriado: el lab está abierto todo el día
(salvo el reinicio de las 07:00 a 08:00, que dejo como corte por las dudas).

**`uni27b` (23/11, pasos de gradiente parejos, 24M; torneos con formatos):**

| Jugadores | `uni27b` contra el especialista | contra `uni27s` (24M, turnos iguales) | contra `uni27` (144M) |
|---|---|---|---|
| 2 | −3,19 (0/16) | **+2,20** (16/16) | **+1,02** (16/16) |
| 3 | −0,88 (0/16) | +0,23 (14/16) | −0,11 (6/16) |
| 4 | −0,61 (0/16) | −0,13 (4/16) | −0,23 (2/16) |
| 5 | −0,39 (0/16) | −0,15 (0/16) | −0,32 (2/16) |
| 6 | −0,41 (0/16) | −0,26 (0/16) | −0,41 (0/16) |
| 7 | −0,73 (0/16) | −0,40 (0/16) | −0,45 (3/16) |

Repartir los pasos de gradiente ayuda mucho con 2 jugadores (+2,2 contra `uni27s`, y con 24M le gana
a `uni27` de 144M) y un poco con 3, y cuesta 0,1 a 0,4 con 4 a 7 (que ahora reciben menos partidas). Pero
con 2 jugadores sigue 3,2 abajo del especialista: el reparto explica una parte, no todo. Lo que queda es,
probablemente, que la red no alcanza para jugar bien todas las mesas a la vez (el juego de 2 es muy
distinto) o que lo que aprende en una pisa a la otra.

Siguiente (23/11, mientras el lab está libre): `uni27h`, lo mismo que `uni27b` con una red de 1024
(dos capas de 1024 en vez de 512), para ver si es capacidad.
