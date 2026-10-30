# round2: la ronda de 2 cartas, en C++

Solver de la ronda de 2 cartas con MCCFR (muestreo externo). Qué es, cómo se agrupan las
situaciones y los resultados: `docs/segunda-ronda/README.md`. Acá, lo técnico.

## Compilar y usar

```bash
make                           # build/round2 (g++ con C++17); desde Python, round2.binary() compila solo
./build/round2 solve --n 3 --iters 30000000 --abstraction b --report_every 5000000 --save tabla.bin
./build/round2 solve --n 3 --iters 30000000 --load tabla.bin --save tabla.bin    # seguir otra sesión
./build/round2 br    --n 3 --games 2000000 --load tabla.bin     # cuánto gana quien se desvía
./build/round2 eval  --n 3 --games 1000000 --load tabla.bin --policies avg,heur,heur
./build/round2 dump  --n 3 --load tabla.bin --save estrategia.tsv
./build/round2 trace --n 3 --games 200                          # para tests/test_round2.py
./build/round2 serve --n 3 --load tabla.bin                       # consultas por stdin (compare_round2_agent.py)
```

- `--abstraction b` agrupa las cartas jugadas antes en la primera baza por su relación con la
  mano propia; `exact` guarda el número exacto (para medir cuánto se pierde agrupando).
- `eval`: una política por posición (`avg` = la estrategia calculada, `heur` = la heurística de
  `evaluation.py`, `uniform` = al azar); imprime el reward medio de cada posición y su error.
- `br`: para cada posición, la mejor desviación (primero la mejor carta en cada situación,
  después el mejor canto sabiendo eso). `gain_in_sample` la mide sobre los mismos repartos con
  que la eligió (exagera por ruido); `gain_holdout`, sobre repartos nuevos (cota por abajo).
- `serve`: lee consultas de a una por línea (`bid <triunfo> <manos> <k> <cantos previos>` o
  `play <triunfo> <manos> <k> <todos los cantos> <cartas ya jugadas>`, posiciones en orden de
  canto) y responde si la situación está en la tabla y las probabilidades de cada acción. Lo usa
  `scripts/exact/compare_round2_agent.py` para poner al agente contra la tabla.
- La tabla se guarda en binario (`--save`) y se retoma con `--load`: sirve para correrlo por
  sesiones en el lab.

## Reglas y convenciones

- Posiciones en orden de canto: la 0 canta primero y sale; la última reparte. La suma de los
  cantos no puede dar 2. Seguir el palo; si no se tiene, fallar con triunfo; si tampoco, cualquiera.
- La segunda baza es forzada (una carta cada uno) y la sale quien ganó la primera.
- Reward: puntaje de la ronda menos el promedio de los rivales (como `reward_scheme=relative`).
- `tests/test_round2.py` compara con el motor de Python: cantos y cartas permitidos, puntajes y
  las decisiones de la heurística, con 2, 3, 4 y 7 jugadores.

## Situaciones (claves)

Etiquetas de palo desde el que decide: 0 triunfo, 1 palo de salida, 2 y 3 los palos de sus cartas
(2 = el de la más alta), 4 y 5 los demás en orden de aparición. Una carta = etiqueta × 13 + número.

- Canto: posición, número del triunfo dado vuelta, las dos cartas y los cantos anteriores.
- Jugada (solo si hay dos cartas posibles): posición, número del triunfo, las dos cartas, todos
  los cantos, quién va ganando la baza y las cartas jugadas antes (etiqueta y, en `b`, cuántas de
  mis cartas de ese palo son más bajas; en `exact`, el número).
