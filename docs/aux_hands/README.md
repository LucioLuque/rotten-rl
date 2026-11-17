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
