# Robustez: ¿se puede explotar al agente?

Las tres cantidades de jugadores (2, 3 y 7) están en meseta con el self-play, y todo lo que probé
desde el 9/11 empata. Pero esa meseta la medí solo contra la heurística y contra versiones del mismo
agente. La pregunta de este tema: ¿el agente juega bien, o llegó a un equilibrio cómodo consigo mismo
que tiene agujeros?

## La prueba: un explotador

`scripts/jax_engine/train_ppo.py --opponent <corridas>` entrena con PPO un agente que juega contra
una versión **congelada** del agente (el último `state/ckpt.pkl` de esas corridas), en vez de
self-play:

- En cada partida, un asiento al azar es del que aprende y los demás de la red congelada, que juega
  su jugada más probable, igual que en las evaluaciones y los torneos.
- Solo se entrena con las jugadas del que aprende (T / n por partida: con n jugadores, n veces
  menos datos por partida que en self-play).
- La evaluación agrega `frozen` (el que aprende en un asiento, la congelada en los demás) en lugar de
  `past`. Ese win rate es la medida: empate 1/n.
- Con varias corridas, una por semilla en el orden de `--seeds` (o de `--init_from`). La red
  congelada puede tener otro tamaño que la que aprende.

Lectura:

- **Si el explotador queda cerca del empate**, no hay agujeros grandes: el agente es sólido y lo que
  falta es capacidad (red más grande, `docs/net_size/`) o búsqueda al jugar.
- **Si le saca bastante**, hay agujeros: el paso siguiente es entrenar contra una liga de versiones
  anteriores o PSRO (`docs/plan.md`).

## Corridas

`queue19.txt` (14/11): `e20i`, `e30i`, `e70i` arrancan de `s20`, `s30` y `u72` (última versión,
24M o 36M) y explotan a esa misma versión congelada, 4 semillas, 6M partidas. Arrancar del mismo
agente es la forma más barata de encontrar una mejora contra él: a 0 partidas da el empate exacto.
Siguen `e20z` y `e30z`: explotadores **de cero** contra `s20` y `s30` congelados, 4 semillas, 12M
partidas. El que arranca del mismo agente busca mejoras cercanas a su forma de jugar; uno de cero
puede encontrar otro estilo que le gane. Con 7 jugadores, `e70z` (en `queue20`, la -14) con 24M
partidas: el que aprende juega 1 de los 7 asientos y recibe 7 veces menos datos por partida que en
self-play, así que 24M equivalen a ~3,4M de self-play (`s70` llegó a un nivel razonable con 3M). Después, `e20zl`: el de 2 jugadores de cero con 36M partidas, porque `e20z` (12M) terminó a −2,25 y
todavía subiendo, y `e20il`: arranca de `s20` como `e20i` pero con tasa 5e-5 (no 3e-4) y 24M partidas,
porque las `i` se pasaron la primera mitad cayendo por la tasa alta. Después la misma cola sigue con la
red de 1024 (`docs/net_size/`).

## Si hay agujeros: la liga

`train_ppo.py --league <run>[@<ep>],...` (14/11, listo sin usar todavía) entrena contra un conjunto
de versiones congeladas en vez de una sola:

- El conjunto arranca con las redes que se le pasan (la última de cada corrida, o un checkpoint con
  `@<ep>`), igual para todas las semillas, con lugar para `--league_size` (32).
- En cada actualización se eligen `--league_groups` (8) al azar y cada una juega un bloque de las
  partidas. Solo se entrena el asiento del que aprende.
- `--league_add_every N` agrega cada N partidas la red actual del que aprende a su conjunto (cuando
  se llena, reemplaza la más vieja de las suyas; las iniciales quedan). `--league_self` hace que uno
  de los rivales elegidos sea la red actual.
- La evaluación agrega `league0` (contra la primera red de `--league`). El conjunto se guarda con el
  estado (`state/league.pkl`) y se retoma con `--resume`.

Es una liga simple (todos los rivales con la misma probabilidad), no PSRO: PSRO elegiría los rivales
según la tabla de quién le gana a quién.

## Resultados (14/11)

Win rate del explotador (en un asiento) contra el agente congelado (en los demás), promedio de 4
semillas, al final de cada corrida:

| Corrida | Contra | Partidas | Win rate (empate) | Diferencia |
|---|---|---|---|---|
| `e20i` (desde el agente) | `s20` | 6M | 50,1 % (50 %) | −0,47 |
| `e30i` (desde el agente) | `s30` | 6M | 32,9 % (33,3 %) | −0,46 |
| `e70i` (desde el agente) | `u72` | 6M | 14,1 % (14,3 %) | −0,24 |
| `e20z` (de cero) | `s20` | 12M | 48,6 % (50 %) | −2,25 |
| `e30z` (de cero) | `s30` | 12M | 32,6 % (33,3 %) | −1,67 |
| `e70z` (de cero) | `u72` | 24M | 13,8 % (14,3 %) | −0,58 |

Ninguno le gana a su agente. Las `i` caen al principio (~−3 con 2 jugadores, por arrancar con tasa
3e-4) y vuelven al empate cuando la tasa llega a 0. Las `z` suben desde abajo y quedan apenas por
debajo, todavía subiendo despacio. Contra la heurística quedan algo por debajo de su agente (`e20z`
76,6 % contra ~82 % de `s20`): juegan peor en general y casi empatan contra el agente.

`e20il` (tasa 5e-5, desde `s20`, 24M) es el primero que le gana a su agente: 51,6 % y +0,82 (las 4
semillas positivas, subiendo parejo). Contra la heurística ~80 % (como `s20`). `s20` tiene un agujero
chico. `e20il_to48M` lo retoma hasta 48M (la tasa vuelve a 2,5e-5 y baja a 0) para ver hasta dónde
llega. `e20zl` (de cero, 36M) terminó en empate exacto: 50,3 % y +0,02.

## La liga con 2 jugadores (al final de `queue20`)

- `l20`: arranca de `s20` y entrena 24M partidas (tasa 1e-4) contra una liga: `s20` final, a 24M y a
  12M, `e20il` y `e20zl` (las 4 semillas de cada uno, 20 redes; la primera, `s20` s0, es `league0`),
  más una copia propia cada 1M partidas y la red actual como uno de los 8 rivales de cada
  actualización.
- `c20`: el control, igual pero en self-play (misma tasa y partidas desde `s20`).
- `t_l20`: torneo entre la heurística, `s20`, `l20`, `c20`, `e20il` y `e20zl`.
- `e20xl` y `e20xc`: un explotador como `e20il` (tasa 5e-5, 24M) contra `l20` y contra `c20`.

La liga sirve si `l20` empata o le gana a `e20il`, no queda peor que `c20` contra la heurística y
`e20xl` le saca menos que `e20xc` (y que los +0,82 de `e20il` a `s20`).

`e20il_to48M` (15/11): a 48M, 51,6 % y **+1,04** contra `s20` (a 24M, +0,82): el agujero llega a ~1
punto por partida y casi no crece en la segunda mitad. Ojo: la línea de `--resume` no llevaba
`--eval_every` (no se guarda en la config), así que evaluó cada 25.000 partidas (~960 evaluaciones,
2,7 GB por semilla y ~10 minutos más); en las líneas de `--resume` pasar `--eval_every`.

### Torneo `t_l20` (15/11 17:56)

Promedio de 4 semillas; diferencia, win rate de la fila (empate 50 %) y cruces ganados:

| Fila vs columna | `s20` 36M | `c20` (control) | `e20il` 48M | `e20zl` 36M | heurística |
|---|---|---|---|---|---|
| `l20` (liga) | +0,55 · 50,5 % · 15/16 | +0,00 · 49,6 % · 7/16 | +0,09 · 49,8 % · 10/16 | +1,43 · 51,5 % · 16/16 | +32,45 · 81,2 % |
| `c20` (control) | +0,47 · 50,6 % · 14/16 | — | −0,56 · 49,4 % · 0/16 | +0,77 · 51,2 % · 16/16 | +34,25 · 82,3 % |
| `s20` | — | −0,47 | −1,10 · 48,8 % · 0/16 | +0,04 | +34,16 · 82,2 % |

- Las dos le ganan a `s20` por lo mismo (~+0,5): entrenar 24M más con tasa 1e-4 lo mejora un poco,
  con o sin liga.
- `l20` y `c20` empatan entre sí, aunque `l20` entrenó con la mitad de los datos (1 de 2 asientos).
- `l20` cierra el agujero de `e20il` (+0,09) y `c20` no (−0,56). Pero `e20il` estaba en la liga de
  `l20`: entrenó contra él. La prueba real son `e20xl`/`e20xc`, explotadores nuevos.
- Contra la heurística `l20` queda ~1,8 puntos abajo de `c20` (81,2 % contra 82,3 %): se especializa un
  poco en sus rivales de la liga.

### Explotadores nuevos contra la liga y el control (15/11)

`e20xl` contra `l20`: 50,2 %, **0,00**. `e20xc` contra `c20`: 51,2 %, **+0,99** (las 4 semillas
positivas). La liga cierra el agujero que `s20` tenía y el control conserva: la robustez viene de la
liga, no de entrenar más.

## La liga con 3 jugadores (15–16/11, -14)

Torneo `t_l30` (16/11 00:52), promedio de 4 semillas (empate 33,3 %):

| Fila vs columna | `s30` 24M | `l30` (liga) | `c30` (control) | `e30i` | `e30z` | heurística |
|---|---|---|---|---|---|---|
| `l30` | +0,12 · 10/16 | — | −0,40 · 1/16 | +0,62 · 16/16 | +1,88 · 16/16 | 62,1 % |
| `c30` | +0,42 · 16/16 | +0,26 · 14/16 | — | +0,94 · 16/16 | +2,02 · 16/16 | 62,7 % |

Con 3 jugadores la liga queda **por debajo del control** (−0,26 a −0,40) y apenas mejora a `s30`;
el control le gana a `s30` por +0,42. Probable causa: con 3 jugadores el que aprende en la liga recibe
un tercio de los datos por partida (con 2, la mitad). Falta ver los explotadores `e30xl`/`e30xc`.

Explotadores nuevos (tasa 5e-5, 24M): `e30xl` contra `l30`: 33,7 %, **+0,36**; `e30xc` contra `c30`:
33,7 %, **+0,19** (las 4 semillas positivas en los dos). **Con 3 jugadores la liga no aporta**: es
algo más débil que el control y no es menos explotable (los dos tienen un agujero chico, menor que el
de `s20` con 2 jugadores). Lo atribuyo a que el que aprende recibe un tercio de los datos por partida.
Idea para probar: que el que aprende ocupe varios asientos en las partidas de la liga.

## La liga con 7 jugadores (15–16/11, -13 y -14)

Torneo `t_l70` (16/11 05:24), promedio de 4 semillas (empate 14,3 %):

| Fila vs columna | `u72` 24M | `l70` (liga) | `c70` (control) | `e70i` | `e70z` | heurística |
|---|---|---|---|---|---|---|
| `l70` | +0,03 · 10/16 | — | −0,13 · 1/16 | +0,30 · 16/16 | +0,73 · 16/16 | 37,7 % |
| `c70` | +0,12 · 14/16 | +0,07 · 12/16 | — | +0,38 · 16/16 | +0,80 · 16/16 | 38,5 % |

Igual que con 3 jugadores: la liga queda un poco por debajo del control (el que aprende recibe 1/7 de
los datos por partida) y casi no mejora a `u72`. Explotadores `e70xl` (-14) y `e70xc` (-13, cortado
a las 06:55; se retoma con `--resume`).
