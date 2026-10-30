# La ronda de 2 cartas, resuelta sin RL

Continuación de `docs/primera-ronda/`. Código en `rotten_rl/exact/round2/` (C++), análisis en
`scripts/exact/analyze_round2.py`.

## Qué cambia respecto de la ronda de 1 carta

- **Canto de 0 a 2**, con la misma restricción para el último (la suma no puede dar 2).
- **Hay una jugada para elegir**: en la primera baza, el que sale elige siempre qué carta tirar;
  los demás solo el 44% de las veces (el resto, seguir el palo o fallar con triunfo los obliga).
  Igual con 3, 4 o 7 jugadores.
- **La segunda baza es forzada** (una carta cada uno): solo hay que calcular quién gana. La sale
  el que ganó la primera.

Las decisiones dependen de cartas que no se ven, así que no sirve una búsqueda tipo alpha-beta
(eso es para cuando todo está a la vista): el valor de cada opción es un promedio sobre las manos
que pueden tener los demás, pesadas por lo que cantaron y jugaron. Eso es lo que hace CFR.

## Cómo se agrupa

- **Manos**: los palos que no son triunfo son intercambiables. Con el número del triunfo dado
  vuelta quedan 5.083 manos distintas (contra ~16.600 sin agrupar).
- **Canto**: mano, posición y cantos anteriores. Con 3 jugadores, ~66.000 situaciones.
- **Primera baza**: además de la mano y todos los cantos, importa lo que salió antes en la baza.
  Guardar el número exacto de cada carta da decenas de millones de situaciones con 3 jugadores y
  miles de millones con 4 o más. La **agrupación B** guarda cada carta jugada por su relación con
  la mano propia:
  - de qué palo es: triunfo, palo de salida, el palo de alguna de mis cartas, u otro (los otros
    se distinguen entre sí, para saber si dos cartas son del mismo palo);
  - si es más alta o más baja que mis cartas de ese palo (si no tengo de ese palo, no importa
    el número: no puede cambiar si gano yo);
  - quién la tiró y quién va ganando la baza.
  Así se ve cuántas salieron de cada palo que importa (idea de Lucio: con 7 jugadores hay 15
  cartas en juego y eso cambia las chances de la segunda baza). Lo que se pierde es sutil
  (cuántas cartas quedan por encima o por debajo de las de los otros) y se mide comparando con la
  versión exacta (`--abstraction exact`) en 3 jugadores.

## Cómo se calcula

MCCFR con muestreo externo: en cada iteración se sortea un reparto; para el jugador que se
actualiza se prueban todas sus opciones y las de los demás se sortean según su estrategia actual.
En C++, con 3 jugadores, ~30 segundos por millón de iteraciones. La convergencia se mide con la
**ganancia por desviarse** (`round2 br`): cuánto ganaría una posición cambiando su forma de
jugar mientras los demás siguen igual. Se mide dos veces: sobre los repartos con que se elige la
desviación (exagera por ruido) y sobre repartos nuevos (cota por abajo).

La tabla se guarda y se retoma (`--save`/`--load`), así que se puede correr por sesiones en el lab
con los procesadores que quedan libres mientras se entrena.

## Resultados: 3 jugadores, agrupación B (27/10/2026)

30 millones de iteraciones, 8 minutos en la notebook, 1,2 millones de situaciones visitadas.
Se regenera con:

```bash
rotten_rl/exact/round2/build/round2 solve --n 3 --iters 30000000 --abstraction b --seed 7 --save tabla.bin
```

**Convergió.** Quien se desvía gana como mucho 0.06 puntos por ronda medido sobre los mismos
repartos (que exagera), y nada medido sobre repartos nuevos (−0.02, ruido). Con 1 millón de
iteraciones eran 0.27–0.43.

**Valor de cada posición** (todos en equilibrio, puntos por ronda contra el resto): +1.50 el
primero, +1.81 el segundo, **−3.32 el último**. Como en la ronda de 1 carta, cantar último es lo
peor.

**Contra la heurística** (una posición juega el equilibrio y las otras dos la heurística,
comparado con que juegue la heurística): +0.98 puntos por ronda saliendo primero, +1.28 segundo,
+0.62 último. Casi 1 punto por ronda de 2 cartas en promedio: la heurística deja bastante.

**Saliendo primero con un triunfo y otra carta**, la probabilidad de salir con la otra depende
sobre todo de lo que se cantó:

| Su triunfo | canto 0 | canto 1 | canto 2 |
|---|---|---|---|
| 2 | 0.23 | 0.97 | 0.41 |
| 5 | 0.27 | 0.97 | 0.35 |
| 8 | 0.41 | 0.97 | 0.22 |
| J | 0.48 | 0.97 | 0.20 |
| A | 0.50 | 0.97 | 0.06 |

- **Cantó 1**: sale con la otra casi siempre, con cualquier triunfo. Guarda el triunfo para la
  segunda baza (la intuición de Lucio, pero no solo con triunfo bajo: con cualquiera).
- **Cantó 2**: sale con el triunfo, más cuanto más alto (con el As, 94%): asegura una baza y sale
  él en la segunda.
- **Cantó 0**: sale con el triunfo la mayoría de las veces, sobre todo si es bajo: se saca de
  encima el triunfo mientras puede perder con él; con triunfo alto es mitad y mitad (depende de
  la otra carta y de los cantos).

Tabla completa: `python scripts/exact/analyze_round2.py --tsv <(round2 dump ...) --n 3`.

## 4 jugadores (agrupación B) y 3 exacta (noche del 27/10/2026)

En el lab, por sesiones de 10M iteraciones (`docs/reportes/2026-10-27.md`). 4 jugadores
agrupada: 1.900M iteraciones, 12.5M situaciones, ~1 GB; **convergida** (quien se desvía gana como
mucho 0.07 sobre los mismos repartos, nada sobre repartos nuevos). 3 jugadores exacta: 2.710M
iteraciones, ~40M situaciones, ~2.9 GB; el último todavía gana 0.12 desviándose sobre los mismos
repartos (nada sobre nuevos). 5 jugadores agrupada: cortada a los 150M por memoria (7.4 GB y
creciendo).

## B contra exacta, 3 jugadores (29/10/2026)

B (30M iteraciones, 1.2M situaciones) contra exacta (5.670M, 41M situaciones), mismos repartos:
el valor de cada posición en equilibrio coincide a la milésima (+1.50 / +1.81 / −3.32) y contra
la heurística las dos le sacan lo mismo (B +0.99 / +1.30 / +0.61; exacta +0.91 / +1.29 / +0.61).
La exacta está incluso menos convergida (quien se desvía gana hasta 0.11 sobre los mismos
repartos, contra 0.08 de B). **La agrupación B no pierde nada medible**; la exacta no se sigue.
Detalle en `docs/reportes/2026-10-29.md`.

## El agente contra la tabla (30/10/2026)

`scripts/exact/compare_round2_agent.py` (con `round2 serve`): el agente en una posición, la tabla
en las demás, contra la tabla en esa posición. Con 3 jugadores (`c10`) pierde 0.035 puntos por
ronda y con 4 (`b40`) 0.060; la heurística, 1.24 y 0.79. Elige el mismo canto que la tabla el 95%
de las veces y la misma carta el 77–79%. **El agente ya juega la ronda 2 casi en equilibrio.**
Detalle en `docs/reportes/2026-10-29.md`.

## Pendiente

- **Agrupación C** (plan del 30/10/2026): para la jugada, en vez de cada carta tirada y los cantos
  de todos, guardar qué carta va ganando y si alguna de las mías le gana, cuántos faltan por tirar,
  cuántos de los que ya tiraron y de los que faltan cantaron 0, 1 o 2, y cuántas cartas salieron
  de cada palo que importa (con tope). Casi no crece con la cantidad de jugadores. Validarla con 3
  y 4 contra B (las mismas tres medidas que exacta contra B) y, si pasa, correr **5, 6 y 7** en el
  lab por sesiones.

- Escalar a 5 y 7 jugadores: con 5 la agrupación B ya no entra en memoria junto a las corridas
  de PPO; hace falta otra agrupación o podar situaciones poco frecuentes.
