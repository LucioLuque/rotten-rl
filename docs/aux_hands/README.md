# Tarea auxiliar: adivinar las manos ajenas

Etapa B del plan del 16/11 (`docs/plan.md`), la primera de la lista por ser casi gratis. Idea de la
bibliografía (`docs/bibliografia.md`, actualización del 6/11): pedirle a la red, además de elegir la
jugada, que prediga qué cartas tiene cada rival. No cambia cómo juega, pero la obliga a armar en sus
capas internas una idea de las manos ajenas, que es lo que importa para cantar y jugar bien.

## Cómo está hecho

`train_ppo.py --aux_hands <peso>` (`rotten_rl/jax_engine/ppo.py`, `aux_coef`):

- Una salida extra sobre la última capa oculta de la red de política (la misma que usa para decidir),
  con 6 × 52 valores: para cada rival por posición relativa (1 a 6) y cada carta, la probabilidad de
  que la tenga.
- Solo cuentan las cartas que el que decide no ve (ni las suyas, ni las que ya salieron, ni el
  triunfo) y los asientos que existen. Pérdida: entropía cruzada binaria, multiplicada por el peso y
  sumada a la de PPO.
- Esa salida no se exporta: la red que juega, se evalúa y entra en los torneos es la de siempre. Con
  peso 0 (por defecto) todo queda igual que antes. Se guarda y se retoma con el estado; los estados
  viejos (sin la salida extra) se siguen cargando.
- En el progreso se muestra `aux_loss` (más bajo = mejor). La precisión no sirve: como la mayoría de
  las cartas no vistas no están en una mano dada, decir siempre "no la tiene" ya acierta ~94 %.

Tests: `tests/test_jax_ppo.py` (objetivos correctos, entrena con una y varias semillas, estados viejos
cargan). Probado de punta a punta en la -14 (CPU), con retomar.

## Experimento propuesto

De cero, que es donde una representación mejor debería notarse más: `a20`, `a30` y `a70` con
`--aux_hands 0.5` y exactamente la receta de `s20`, `s30` y `u72` (12M partidas, 4 semillas), y torneo
contra esos a 12M (`@12000256`, el final de su tramo de 12M, con la misma tasa que baja a 0). Lista en
`queue25.txt`; agregada al final de `queue24` en la -14 el 17/11 a la 01:05 (arranca después de
`e70xl`, ~02:00). `a70` no llega antes del corte de las 06:55: se retoma con `--resume` desde las 08:00.

## Resultados

**2 jugadores (`t_a20`, 17/11 02:53):** `a20` (12M con `--aux_hands 0.5`) contra `s20` a 12M: +0,10,
50,3 % de win rate, 9/16 cruces: **empate**. Contra la heurística, igual (82,3 %). La red aprende a
predecir (`aux_loss` 0,20; con solo la proporción de cartas daría ~0,33), pero no juega mejor. ~15 %
más lenta (3.379 partidas/s por semilla). Con 2 jugadores hay poco oculto; falta 3 y 7.

### Qué tan bien predice (17/11, `scripts/aux_hands/eval_aux.py`, semilla 0, 64 partidas)

En cada decisión, sobre las cartas que el que decide no ve. **Error**: entropía cruzada, contra "a
ciegas" (solo sabe cuántas cartas tiene cada uno). **Aciertos**: de las c cartas que cree más
probables para un rival con c cartas, cuántas tiene de verdad, contra elegirlas al azar.

| | Error | A ciegas | Aciertos | Al azar |
|---|---|---|---|---|
| `a20`, todas las decisiones | 0,199 | 0,208 | 14,2 % | 8,8 % |
| `a20`, canto | 0,262 | 0,265 | 12,1 % | 10,3 % |
| `a20`, juego | 0,181 | 0,192 | 15,1 % | 8,2 % |
| `a30`, todas | 0,204 | 0,215 | 14,4 % | 9,1 % |
| `a30`, canto | 0,260 | 0,265 | 12,9 % | 10,3 % |
| `a30`, juego | 0,187 | 0,200 | 15,1 % | 8,5 % |

Aprende algo (≈1,6 veces mejor que al azar en aciertos; ~5 % menos error que a ciegas), sobre todo
durante el juego (los palos que faltan y las cartas que salieron dan pistas) y casi nada en el canto
(solo están los cantos de los demás). Hay poca información para adivinar: la mayoría de las cartas no
vistas ni siquiera están repartidas (quedan en el mazo). Y aun prediciendo algo, **no juega mejor**
(empate con 2 jugadores, indicio chico con 3): la red ya usaba esa información sin la tarea extra, o
no le alcanza para cambiar decisiones.

**3 jugadores (`t_a30`, 17/11 04:20):** `a30` contra `s30` a 12M: +0,12 a +0,15, 33,5 %, 12/16 cruces
(indicio, dentro del ruido de ~0,3 entre corridas iguales). **7 jugadores (`t_a70`, 17/11 09:00):**
`a70` contra `u72` a 12M: −0,01, 14,2 %, 7/16: empate.

**Conclusión: la tarea auxiliar no mejora el juego con 2, 3 ni 7 jugadores.** Descartada como mejora
de la receta (el código queda, apagado por defecto).
