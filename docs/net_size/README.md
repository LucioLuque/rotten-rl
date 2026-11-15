# Tamaño de la red

La red de PPO en JAX es una MLP de dos capas de 512 (`--hidden 512,512`). Si la meseta de 2, 3 y 7
jugadores es de capacidad, una red más grande debería pasarla.

## La prueba

`queue19` (`docs/robustness/queue19.txt`, 14/11): `k20`, `k30` y `k70` entrenan de cero con
`--hidden 1024,1024` y la receta actual (tasa 3e-4, 2 épocas, minilote 1024, lote de 512 partidas,
entropía 0,06 con 2 y 3 jugadores y 0,1 con 7), 4 semillas y 12M partidas. Los torneos `t_k20`,
`t_k30` y `t_k70` las comparan con `s20`, `s30` y `u72` a la misma cantidad de partidas
(`@12000256`, el final de su tramo de 12M) y con su última versión (24M o 36M).

No pude medir la velocidad antes de lanzar: el 14/11 a las 18:35 la GPU estaba ocupada por otro
usuario del lab y las pruebas cortas no tuvieron memoria. La referencia es `o23` (abajo): con 2
jugadores, ×2,15 más lenta que la de 512. Con 3 y 7 no está medida.

Los nombres `n20`/`n30`/`n70` ya estaban usados por corridas viejas (`n30`, `n70` del 7/11); por eso
`k`.

## Pruebas anteriores con 1024

- **26/10, `x24`** (PPO en PyTorch, 2 jugadores, 200k partidas, receta vieja): no ayudó (0,400 contra
  la heurística, por debajo de la base). Arrancó más rápido y después no siguió subiendo
  (`docs/reportes/2026-10-26.md`).
- **7/11, `o23`** (PPO en JAX, 2 jugadores, 3M, 2 épocas, entropía 0,01): le ganó a `m26` (512, misma
  receta y partidas) por +0,58 (15/16 cruces), pero era ×2,15 más lenta (6.898 partidas/s). Conclusión
  de entonces: a igual tiempo convenía duplicar partidas con la de 512 (`docs/reportes/2026-11-06.md`).

Por qué de nuevo: ahora duplicar partidas ya no mejora (meseta de `s20`, `s30` y `u72`), así que la
capacidad es lo que queda por probar. Además nunca se probó con la receta actual (entropía 0,06/0,1),
ni con 3 y 7 jugadores, ni a 12M.

## Resultados

**2 jugadores (`t_k20`, 15/11):** la red de 1024 no mejora. `k20` (12M) contra `s20` a 12M: −0,59,
50,0 % de win rate, 2 de 16 cruces; contra `s20` final (36M): −1,45, 48,7 %, 0/16. Contra la
heurística 81,2 % (`s20` a 12M, 82,3 %). Velocidad: 1.818 partidas/s por semilla (4 semillas); con
3 jugadores (`k30`), 1.228.
