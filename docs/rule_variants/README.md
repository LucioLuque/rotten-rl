# Variantes de reglas

Etapa D del plan del 16/11 (`docs/plan.md`): reglas que juega la gente y que cambian el juego, así que
obligan a reentrenar. Se entrenan con la receta fija (la de la etapa B no cambió).

## No cantar lo mismo 3 rondas seguidas (`--no_repeat_bid`)

La regla (`docs/plan.md`, "Variantes de reglas"): nadie puede cantar la misma cantidad 3 rondas
seguidas (cualquier número, no solo 0). Quien cantó 0 y 0 en las dos rondas anteriores no puede cantar
0 en esta. Si choca con la regla del último en cantar y no le queda ningún canto posible, gana la del
último: ahí sí puede repetir.

Cómo está hecho (motor JAX, `rotten_rl/jax_engine/game.py`):
- El estado guarda los cantos de cada jugador en las dos rondas anteriores (`prev_bids`) y si la
  partida es con la variante (`no_repeat_bid`); `legal_mask` saca el canto repetido salvo que no quede
  otro.
- La observación agrega, con la variante, los dos cantos anteriores de cada jugador (por asiento
  relativo, one-hot de "ninguno, 0..7"): 808 + 126 = 934 valores. Las redes del juego normal siguen
  con sus 808.
- `scripts/jax_engine/train_ppo.py --no_repeat_bid`; las evaluaciones durante el entrenamiento (contra
  azar, heurística y versiones anteriores) juegan con la variante, y la heurística respeta la regla
  (elige entre los cantos legales). Solo con la evaluación en JAX, sin liga.
- Todavía no está en el juego de pyspiel (`rotten_rl/openspiel_game.py`): los torneos y `play.py` no
  la tienen.

Tests (`tests/test_jax_rule_variants.py`, pasan en la -14 junto con los del motor y PPO, 37/37):
partidas al azar con la variante nunca tienen 3 cantos iguales seguidos salvo cuando no queda otro;
sin la variante sí pasa; el choque con la regla del último; la historia de la observación; PPO entrena
y evalúa con la variante. Sin la variante el motor sigue igual a pyspiel jugada a jugada.

Prueba corta en la -14 (3 jugadores, 1M): 2.575 partidas/s por semilla (sin la variante, 2.753).

### Experimento

| Corrida | Qué |
|---|---|
| `d20` | 2 jugadores con la variante, de cero a 24M, 4 semillas, receta actual (entropía 0,06) |
| `d30` | 3 jugadores, ídem |
| `d72` | 7 jugadores (entropía 0,1), ídem |

Cola `queue33` en la -14, lanzada el 20/11 a las 05:13. Tiempos estimados (sin medir con 2 y 7):
`d20` ~1 h 50 min, `d30` ~2 h 40 min, `d72` ~6 h. Lo que corte el cierre de las 06:55 se retoma a las
08:05 (`relaunch_queue.sh queue33 queue34`, por timer).

Qué se mira: las curvas y la evaluación contra la heurística (jugando con la variante), y torneos
con la variante (`scripts/jax_engine/tournament.py --no_repeat_bid`): heurística, el agente del juego
normal (`s20full`, `s30full`, `u72full`) jugando con la regla pero sin ver la historia (usa sus 808
valores), y el entrenado con ella. Si el entrenado con la variante no le gana al normal, aprender la
regla no aporta.

Prueba del torneo (20/11, 200 repartos, 3 jugadores): `s30full` con la regla le gana a la heurística el
63 % (en el juego normal, 62 %), así que la regla parece pesar poco con la heurística en la mesa.

### Resultados

**`t_d20` (20/11, 2 jugadores, 1000 repartos por cruce, todos jugando con la regla): empate.** `d20`
contra `s20full` **−0,07 por partida, 49,4 % de win rate (empate 50 %), 7/16 cruces positivos**; contra
la heurística 82,4 % (`s20full` con la regla, 82,9 %). Las curvas de entrenamiento son prácticamente
las de `s20full` (las de `d20` se evalúan con la regla; las de `s20full`, en el juego normal); los
cantos exactos, un punto más bajos (49 contra 50 %). `d20` se cortó a las 06:55 a ~23M y se retomó a
las 08:05 con `--resume`.

**Sin la regla** (20/11, el mismo torneo en el juego normal; `d20` sigue viendo la historia de cantos,
que el juego normal también tiene): **también empate**, `d20` contra `s20full` −0,06, 49,5 %, 9/16;
contra la heurística 82,0 % (`s20full` 82,3 %). Desde el 20/11 las evaluaciones y los torneos en JAX
arman siempre la observación con la historia (las redes del juego normal usan sus primeros 808 valores),
así que un agente entrenado con la variante puede jugar el juego normal.

Con 2 jugadores, la regla casi no cambia el juego: el agente normal, jugando con la regla sin saberla,
rinde lo mismo que el que la aprendió.

**`t_d30` y `t_d30_sinregla` (20/11, 3 jugadores, 1000 repartos por cruce):**

| `d30` contra `s30full` | Diferencia por partida | Win rate (empate 33,3 %) | Cruces positivos |
|---|---|---|---|
| Con la regla | −0,08 | 33,2 % | 7/16 |
| Sin la regla | **−0,42** | 32,9 % | **0/16** |

Contra la heurística: con la regla `d30` 61,5 % y `s30full` 60,4 %; sin la regla 62,3 % y 62,1 %. Con la
regla, empate: el agente normal la juega sin saberla y rinde igual. **Sin la regla, `d30` pierde contra
`s30full` por más que el ruido (~0,3) y en los 16 cruces**: lo que aprendió con la regla no le sirve
en el juego normal (probablemente cuenta con que los rivales no pueden repetir el canto). Los cantos
exactos de `d30` contra la heurística quedan 2 puntos abajo (59,8 contra 61,7 %); el resto de las
curvas, igual. Es un argumento para entrenar mezclando partidas con y sin la regla (`--formats`), y
para que la red sepa en cuál está.

**`t_d72` y `t_d72_sinregla` (20/11, 7 jugadores, 1000 repartos por cruce): con 7 jugadores la regla
pesa mucho.**

| `d72` contra `u72full` | Diferencia por partida | Win rate (empate 14,3 %) | Cruces positivos |
|---|---|---|---|
| Con la regla | **+1,12** | 15,7 % | **16/16** |
| Sin la regla | **−1,33** | 13,1 % | **0/16** |

Con la regla, `d72` le gana claro al agente normal (que la juega sin saberla); sin la regla pierde por lo
mismo. Contra la heurística con la regla, `d72` 34,1 % y `u72full` 32,2 %; sin la regla, 36,4 % y
38,2 %. Curvas: con la regla los cantos exactos bajan mucho (71,5 contra 79,1 %; con 7 jugadores se
canta 0 casi siempre en las rondas chicas, y la regla obliga a cantar otra cosa cada tanto), la varianza
explicada sube (0,55 contra 0,51) y la entropía queda un poco más alta (0,254 contra 0,247).

**Conclusión de la regla de no repetir:** con 2 jugadores no cambia nada; con 3, saberla no aporta pero
entrenar siempre con ella empeora el juego sin ella; con 7, saberla vale más de un punto por partida y
entrenar siempre con ella cuesta lo mismo en el juego normal. Hace falta una red que sepa si la regla
está, entrenada con las dos: `--formats` (`m20`, `m30`, `m72`).

**Cuánto aparece la regla (20/11).** Partidas de cada agente contra sí mismo con la regla, 2000
repartos, en la escalera vieja (script de una vez, sin guardar en el repo): en qué porcentaje de los
cantos el que canta viene de cantar dos veces lo mismo y podría repetirlo ("restringido"), en cuántos el
agente, sin la regla, habría cantado justo eso ("la regla cambia el canto"), y en cuántos repite porque
no le queda otro canto:

| Jugadores | Agente | Restringidos | La regla cambia el canto | Repite porque no queda otro |
|---|---|---|---|---|
| 2 | `s20full` / `d20` | 12,8 / 12,9 % | 2,9 / 3,0 % | 0,05 % |
| 3 | `s30full` / `d30` | 16,2 / 16,6 % | 5,5 / 5,6 % | 0,2 % |
| 7 | `u72full` / `d72` | 23,9 / 24,3 % | **13,1 / 12,9 %** | 0,03 % |

Con 7 jugadores la regla cambia uno de cada 8 cantos (casi siempre un 0 que habría repetido); con 2,
uno de cada 33. No es un caso raro: con `--formats` (la regla en la mitad de las partidas) aparece
millones de veces en un entrenamiento, así que no hace falta forzarlo. `m20` con formatos: 2.542
partidas/s por semilla (sin formatos 4.002: 1,57 veces más lento).

## Formato de partida (`--formats`)

Decidido el 20/11. La mesa decide antes de empezar la partida tres cosas, y cambian de una mesa a
otra:

- **La escalera** (cuántas cartas en cada ronda). Lo más común es 1,2,3,4,5,6,7,7,6,5,4,3,2,1 (14 rondas,
  el 7 repetido); los agentes hasta el 20/11 jugaron siempre 1..7..1 (13 rondas, un solo 7). A veces se
  agrega 7,7 al final.
- **Rondas sin triunfo**: no se da vuelta carta; hay que seguir el palo que salió si se tiene, si no se
  tira cualquiera (no hay obligación de fallar); gana la carta más alta del palo que salió (el palo que
  sale hace de triunfo). La primera baza la abre el que sigue al que reparte y cada baza siguiente el
  que ganó la anterior, como siempre. Los cantos no cambian.
- **La regla de no cantar lo mismo 3 rondas seguidas**, sí o no.

Los formatos que más vi: 1,2,3,4,5,6,7,7\*,6,5,4,3,2,1,7,7\* y 1,2,3,4,5,6,7,7\*,6\*,5\*,4\*,3\*,2\*,1\* (\* =
sin triunfo). Se juega con el mazo entero (con 7 jugadores y 7 cartas se usan 49 de 52, más la de
triunfo).

**El sorteo** (`rotten_rl/jax_engine/formats.py`): cada partida de entrenamiento sortea su formato, con
algo de peso a los formatos que se ven en la mesa y el resto repartido en variaciones, para que el
agente juegue bien lo que decida cualquier mesa: 20 % 1..7,7..1; 10 % y 10 % los dos de arriba; 60 % al
azar (7 repetido o no, 7,7 extra al final o no, rondas sin triunfo: ninguna, una al azar, cada una con
probabilidad 0,2, toda la bajada o el segundo 7 de cada par). La regla de no repetir, en la mitad de
las partidas, aparte.

Simulación con 1.000.000 de partidas (`scripts/rule_variants/simulate_formats.py`): 14.758 formatos
distintos. Los más frecuentes: 1..7,7..1 32 %; 1..7,7\*,6\*..1\* 13 %; 1..7,7\*,6..1,7,7\* 11 %; 1..7..1
(la vieja) 6 %; 1..7,7..1,7,7 4 %; 1..7,7\*,6..1 3,5 %; los 10 primeros suman 76 %. Como la parte al azar
vuelve a armar los formatos nombrados, terminan pesando más que sus pesos directos (32, 13 y 11 %
contra 20, 10 y 10). 14 rondas en el 59 % de las partidas, 16 en el 20 %, 13 en el 16 %, 15 en el 5 %.
Partidas con alguna ronda sin triunfo, 56 % (1,8 en promedio); cada ronda va sin triunfo el 11 % de las
veces (las de 7 cartas, 22 %). Lo dejé así para probar.

**En el motor JAX** (`rotten_rl/jax_engine/game.py`): el estado lleva las cartas de cada ronda
(`round_h`, hasta 16 rondas) y cuáles van sin triunfo (`no_trump`); `format_game` sortea el formato y
`new_game` sigue siendo, por defecto, la escalera vieja con triunfo (igual a pyspiel jugada a jugada).
La observación de las redes con formatos agrega, a los 808 y a la historia de cantos (126), el formato:
para la ronda actual y las que siguen (hasta 16), cuántas cartas (one-hot 0..7, 0 = no hay) y si va sin
triunfo, y si está la regla de no repetir (145 valores; 1.079 en total). En los 808, una ronda sin
triunfo prende el lugar "sin triunfo" del palo de triunfo (ya existía) y el índice de ronda queda en
13 lugares (con más rondas lo dice el formato).

Entrenamiento (`train_ppo --formats`): las partidas tienen largos distintos, así que se juegan todas
hasta el largo de la más larga (16 rondas) y los pasos después del final quedan fuera de la pérdida.
Las evaluaciones durante el entrenamiento también sortean el formato. Los torneos en JAX:
`tournament.py --formats` (los agentes viejos juegan cualquier formato con la observación que conocen).
Todavía no está en pyspiel (`play.py`, torneos en CPU).

### Resultados con formatos

**`t_m20` (20/11, 2 jugadores, 1000 repartos por cruce, cada reparto con su formato sorteado): `m20`
les gana claro a los agentes de la escalera vieja.**

| Con formatos sorteados | Diferencia por partida | Win rate (empate 50 %) | Cruces positivos |
|---|---|---|---|
| `m20` contra `s20full` | **+3,34** | 53,6 % | **16/16** |
| `m20` contra `d20` | **+3,43** | 53,7 % | **16/16** |
| `d20` contra `s20full` | −0,07 | 49,8 % | 7/16 |

Contra la heurística: `m20` 86,2 % (+40,9 por partida), `d20` 84,5 % (+36,9), `s20full` 84,2 % (+36,9).
Los agentes viejos nunca vieron rondas sin triunfo ni escaleras de 14 a 16 rondas; `m20` sí. Las curvas
de `m20` contra la heurística no se comparan directo con las de `s20full` (otro juego: más rondas, más
puntos); en el entrenamiento, la entropía queda más alta (0,37 contra 0,34) y el resto igual.
Velocidad: 2.542 partidas/s por semilla, con el relleno (`m30` y `m72` ya van sin relleno, ver abajo).
El torneo en la escalera vieja (`m20` contra `s20full` en su juego) falló la primera vez (la
evaluación no armaba el formato en el juego normal; arreglado) y se repite después de `m20c`.

Tests (`tests/test_jax_formats.py`, junto con todos los de JAX: 72/72 en la -14): partidas al azar con
formatos sorteados respetan las reglas (cartas por ronda, seguir el palo, en las rondas sin triunfo se
tira cualquiera si no se tiene el palo, quién gana la baza); un formato nombrado; la observación; las
frecuencias del sorteo; PPO entrena y evalúa con formatos, y una red vieja juega con formatos.
