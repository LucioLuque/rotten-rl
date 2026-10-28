# La ronda de 1 carta, resuelta sin RL

## Qué es

En la ronda de 1 carta (la primera y la última de cada partida) la jugada es forzada: con una
sola carta no hay nada que elegir. La única decisión es el **canto, 0 o 1**, y quién gana la
baza depende solo de las cartas. Es la única parte del juego lo bastante chica para calcular
cómo conviene jugar en cada situación, sin RL.

Sirve para dos cosas:

1. **Aprender del juego**: cuánto pesa la posición, cuánto conviene escuchar lo que cantaron los
   demás, cuánto se le puede sacar a una mesa que juega mal.
2. **Controlar al agente**: es el único caso con respuesta conocida. Si el agente de RL canta
   mal acá, probablemente también aprende mal en las rondas grandes, donde no se puede
   verificar.

## Cómo se achica

Una situación de canto queda definida por:

| Qué | Casos |
|---|---|
| Número de la carta de triunfo dada vuelta | 13 |
| La carta propia: triunfo (12 números posibles) o no triunfo (13) | 26 tipos |
| La posición en el orden de canto y lo que cantaron los anteriores (0 o 1 cada uno) | 2^posición |

Los tres palos que no son triunfo son intercambiables (tener el 9 de corazones o el de tréboles da
lo mismo si ninguno es triunfo), y del triunfo dado vuelta solo importa el número. Eso deja
13 × 26 × (2ⁿ − 1) situaciones: unas 2.300 con 3 jugadores y 41.000 con 7, contra cientos de
miles de millones de repartos posibles. En la literatura se llama abstracción sin pérdida por
simetría (Gilpin y Sandholm, "GameShrink", 2007).

## Cómo se calcula

- **Equilibrio** con CFR+ (el algoritmo con el que se resolvió el póker): en cada iteración se
  sortean 20.000 repartos, se recorren todas las combinaciones de cantos y cada situación
  acumula cuánto se "arrepiente" de no haber cantado distinto. La estrategia promedio converge
  a una en la que nadie gana cambiando su forma de cantar. Con más de 2 jugadores CFR no tiene
  garantía teórica, así que se mide: "ganancia por desviarse" cerca de 0 = equilibrio.
- **Mejor respuesta** contra cualquier forma de jugar de los demás (equilibrio, heurística,
  el agente): qué cantar en cada situación para ganar lo máximo, con toda la información o
  mirando solo la carta propia (sin darle peso a los cantos ajenos).
- El reward es el mismo con el que entrena el agente: puntaje de la ronda menos el promedio de
  los rivales. Con el objetivo de ganar la partida, la última ronda de 1 carta dependería además
  del puntaje acumulado (no está contemplado).

Una estrategia puede ser "mezclada" (cantar 1 con cierta probabilidad): sirve para no regalar
información con el canto, que los que cantan después usan. Con 1 carta los cantos ajenos no
cambian quién gana la baza; solo dicen algo de las cartas de cada uno.

## Resultados (27/10/2026)

Detalle completo en `resultados.md`. Todos convergieron: quien se desvía del equilibrio gana
menos de 0.02 puntos por ronda (en una ronda donde se juegan ~10).

**1. Cantar último es muy malo.** El último en cantar (el que reparte) pierde en promedio ~4 a
~6 puntos por ronda contra el resto (−4.0 con 3 jugadores, −5.8 con 7): la restricción de la
suma le saca justo el canto que le convenía. El resto gana, y más cuanto más tarde canta (más
información). Con 2 jugadores el primero gana +4.75 por ronda.

**2. Qué cantar** (probabilidad de cantar 1 promediada sobre las situaciones; donde no es 0 o 1,
en general es porque depende de lo que cantaron antes, no necesariamente porque haya que tirar
una moneda):

- 2 jugadores: el primero canta 1 con cualquier carta; el segundo está forzado a copiarlo.
- 3 jugadores: el primero canta 1 con 10 o más alto sin ser triunfo (el 9, a veces) o con
  cualquier triunfo; el segundo, solo con triunfo (desde el 8 casi siempre, los bajos la mitad de
  las veces según los cantos previos); el tercero, solo con triunfo Q, K o A.
- 7 jugadores: **nadie canta 1 sin triunfo**, ni con un As saliendo primero (con 6 rivales, casi
  seguro alguno tiene triunfo). Con triunfo, cuanto más tarde se canta más bajo alcanza... hasta el
  último, que solo canta 1 con K o A de triunfo.

**3. Escuchar los cantos de los demás vale, pero poco.** Contra rivales en equilibrio, cantar solo
según la carta y la posición cuesta 0.07 a 0.49 puntos por ronda, más cuanto más tarde se canta
(con 7 jugadores: 0.07 el segundo, 0.44 el sexto). Comparado con lo que pesa la posición (varios
puntos), es chico: la carta y la posición explican casi todo.

**4. La heurística de `evaluation.py` canta casi como el equilibrio**, salvo con 2 jugadores: ahí
canta 0 con cartas bajas saliendo primero y pierde 2.7 puntos por ronda de 1 carta (debería cantar
1 siempre). Con 3 a 7 jugadores, lo máximo que se le saca a una mesa de heurísticas es 0.1 a 0.6
puntos por ronda, y casi todo escuchando los cantos (la heurística no los mira).

## Archivos

- `rotten_rl/exact/round1.py`: el cálculo (repartos, CFR+, mejor respuesta, tablas de la
  heurística y de un agente).
- `scripts/exact/solve_round1.py`: resuelve de 2 a 7 jugadores y escribe `resultados.md` y
  `equilibrio.npz` (las tablas: `n{jugadores}_pos{posición}`, probabilidad de cantar 1 por
  situación, índice `(número del triunfo × 26 + tipo de carta) × 2^posición + cantos anteriores`).
- `tests/test_round1.py`: quién gana la baza coincide con el motor; con 2 jugadores el primero
  canta 1 siempre; las situaciones se arman bien en pyspiel (para preguntarle al agente).

```bash
python scripts/exact/solve_round1.py                 # ~45 min en la notebook (7 jugadores: 20), sin GPU
```

## Pendiente

- Comparar con el agente: `scripts/exact/compare_round1_agent.py --run <corrida>` (arma sus
  tablas con `round1.policy_tables` y mide, por posición, cuánto pierde cantando como el agente
  con los demás en equilibrio). Resultados en `docs/reportes/2026-10-27.md`.
- Rondas de 2 cartas: ya hay jugadas para elegir; probablemente viable con MCCFR y la misma
  simetría de palos.
