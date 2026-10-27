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

## Resultados

En curso (27/10/2026): 3 jugadores con la agrupación B. Ver más abajo cuando termine.

## Pendiente

- Comparar B con la versión exacta en 3 jugadores (cuánto se pierde agrupando).
- Escalar a 4, 5 y 7 jugadores (en el lab, por sesiones).
- Comparar con el agente de RL en las mismas situaciones, como en la ronda de 1 carta.
