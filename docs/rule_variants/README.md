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

Con 2 jugadores, la regla casi no cambia el juego: el agente normal, jugando con la regla sin saberla,
rinde lo mismo que el que la aprendió.

## Rondas sin triunfo

Pendiente de definir (cuáles rondas y quién lo decide). La observación ya tiene el lugar ("sin
triunfo" en el palo de triunfo).
