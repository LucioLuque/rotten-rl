# Tamaño de la red

La red de PPO en JAX es una MLP de dos capas de 512 (`--hidden 512,512`). Si la meseta de 2, 3 y 7
jugadores es de capacidad, una red más grande debería pasarla.

## La prueba

`queue19` (`docs/robustness/queue19.txt`, 14/11): `n20`, `n30` y `n70` entrenan de cero con
`--hidden 1024,1024` y la receta actual (tasa 3e-4, 2 épocas, minilote 1024, lote de 512 partidas,
entropía 0,06 con 2 y 3 jugadores y 0,1 con 7), 4 semillas y 12M partidas. Los torneos `t_n20`,
`t_n30` y `t_n70` las comparan con `s20`, `s30` y `u72` a la misma cantidad de partidas
(`@12000256`, el final de su tramo de 12M) y con su última versión (24M o 36M).

No pude medir la velocidad antes de lanzar: el 14/11 a las 18:35 la GPU estaba ocupada por otro
usuario del lab y las pruebas cortas no tuvieron memoria. Estimado sin medir: 2 a 3 veces más lenta
que la de 512.
