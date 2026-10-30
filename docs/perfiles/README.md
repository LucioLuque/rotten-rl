# Perfiles

Formas de jugar contra las que medir a los modelos, y la interfaz para jugarles. Hecho: la interfaz,
el guardado, los perfiles a mano y las **situaciones elegidas**, de las que sale el perfil de una persona.
Falta: las respuestas de Lucio.

## Jugar

```bash
.venv/bin/python scripts/play.py                         # practicar contra la heurística
.venv/bin/python scripts/play.py --perfil lucio --situaciones 150   # jugadas elegidas, lo principal
.venv/bin/python scripts/play.py --perfil lucio          # partidas enteras, guardando tus decisiones
.venv/bin/python scripts/play.py --perfil lucio --jugadores 3 --rival runs/lab/labo-s004-19/runs/<corrida>
```

Con `--perfil` cada decisión va a `data/perfiles/<perfil>/<fecha>.jsonl`, con la historia completa
de la partida: alcanza para reconstruir el estado y sacar la observación que haga falta. Esos
archivos se commitean: son datos que no se pueden regenerar. Cada partida usa el reparto siguiente de
la serie `--reparto_seed` (se saltean los ya jugados) y se rota de asiento (`reparto % jugadores`), para
poder darle después el mismo reparto, en el mismo asiento, a un modelo. Al guardar no se pueden ver las
manos ni usar el autoplay. Una partida sin línea `fin` quedó abandonada.

## Qué es un perfil

Un perfil es un rival con nombre: una función que recibe el estado y las acciones legales y elige una.
Es lo mismo que ya son `heuristic` y `random` en `rotten_rl/evaluation.py`, así que entran donde esos
ya entran: `--eval_opponents` de `train.py` y `--agents` de `scripts/eval/tournament.py`.

| Perfil | Qué es |
|---|---|
| `heuristic` | La actual, en `rotten_rl/evaluation.py` |
| `conservador`, `arriesgado`, `a_perder` | `rotten_rl/perfiles/reglas.py`: juegan como la heurística y cantan 70%, 130% de su estimación o lo mínimo |
| `lucio` | Imita a Lucio, a partir de sus partidas |
| Otras personas | Igual que `lucio`, cada una con sus partidas |

Código en `rotten_rl/perfiles/`; la interfaz es `scripts/play.py`.

Torneo con 2 jugadores (1.000 repartos duplicados, win rate de la fila contra la columna, ±0.01):

| | heuristic | arriesgado | conservador | a_perder |
|---|---|---|---|---|
| heuristic | — | 0.556 | 0.735 | 0.912 |
| arriesgado | 0.444 | — | 0.687 | 0.828 |
| conservador | 0.266 | 0.313 | — | 0.770 |
| a_perder | 0.088 | 0.172 | 0.230 | — |

Son estilos bien distintos, y ninguno le gana a la heurística.

## Aprender de una persona: situaciones elegidas

Lo principal son jugadas elegidas, mostradas en la pantalla del juego, una por vez; las partidas enteras
complementan. Está en `rotten_rl/perfiles/situaciones.py`.

- **Tipo de jugada:** cada decisión se reduce a lo que importa, en tres niveles. Canto: tamaño de la
  ronda, posición y triunfos; después cartas altas y cuánto se cantó antes. Jugada: si sale o sigue,
  si le faltan bazas y si puede ganar; después si hay rivales buscando y cuántas cartas quedan. Y en
  el nivel más detallado, en los dos casos, cómo va la partida (ganando, parejo o perdiendo por 15
  puntos, si es de las últimas 3 rondas); en la jugada, además, cuántos triunfos faltan salir y si
  tiene una carta que ya no se puede superar.
- **La respuesta se generaliza:** no se guarda "jugó el 7♠" sino "ganó con la carta más baja que gana",
  o "cantó uno más que la estimación". Por eso una respuesta sirve para otras manos parecidas.
- **Qué se pregunta:** se simulan partidas, y se priorizan los tipos frecuentes en los que la decisión
  cambia mucho el resultado de la ronda (probando cada opción). Cuotas: 40% cantos, 25% salidas, 35%
  seguir. Los tipos más importantes se repiten con otras manos, para ver si la respuesta es consistente.
- **Cuántas:** 150 como base, lo razonable para pedirle a otra persona. Se pueden pedir más cuando
  sea: no repite las ya respondidas (`--situaciones 100` son 100 nuevas), y por eso se puede cortar y
  seguir después pidiendo las que faltan. Lucio puede responder más.
- **Cómo juega el perfil:** la respuesta más repetida del tipo más detallado que tenga respuestas, y si
  no sabe nada, la heurística. Entra en el torneo y como rival por su nombre (`--agents lucio,heuristic`,
  `play.py --rival lucio`).
- **Validación:** `agreement` deja afuera un quinto de las respuestas y mide cuánto las predice. Con un
  jugador simulado (la heurística respondiendo 150 situaciones) dio 82%: prueba que generaliza, no que
  imite a Lucio. Después se compara contra una red ajustada a las mismas respuestas.
- **Limitación conocida:** con 150 respuestas el nivel más detallado casi no se repite, así que si la
  decisión cambia por los puntos o las cartas se va a ver mejor con más respuestas.

## La interfaz

`scripts/play.py` y `rotten_rl/renderer.py` (Pygame). Corre sobre el estado de OpenSpiel
(`rotten_rl/perfiles/partida.py`), así que el rival puede ser cualquier perfil o modelo. Muestra tu lugar
en el canto y en la baza, por qué falta un canto y contra quién jugás. En las situaciones, las cartas ya
jugadas van por palo en el panel, porque no se vieron caer.

`scripts/perfiles/capturas.py` juega con clicks de verdad en cada mesa y tamaño de pantalla y guarda
capturas. Revisado el 30/10/2026 con 2 a 7 jugadores en 1920×1080, 1366×768 y 1100×650. `tests/
test_renderer.py` controla que las cartas de la baza no se pisen.

## Cómo se mide cada entrenamiento

| Contra qué | Qué dice | Alcance |
|---|---|---|
| Heurística y otros perfiles | Si le gana a cada estilo | Partida entera |
| `lucio` | Si le ganaría a Lucio jugando como juega | Partida entera; depende de cuánto lo imite |
| Lucio en persona | Si le gana de verdad | Pocas partidas: mucho ruido aun con repartos repetidos |
| Equilibrio exacto | Cuánto pierde contra el equilibrio | **Solo rondas de 1 y 2 cartas** |

Con 2 jugadores, la ronda de 1 carta está resuelta y `round2 solve --n 2` corre (3M iteraciones en 12 s en
la laptop). Falta hacer la tabla definitiva y comprobar que `compare_round2_agent.py` ande con 2. Las
mismas tablas sirven para decirle a Lucio dónde pierde puntos en esas rondas.

## Orden, empezando por 2 jugadores

1. ~~Interfaz sobre el estado de OpenSpiel, con guardado y un rival cualquiera.~~
2. ~~Perfiles `conservador`, `arriesgado` y `a_perder`.~~
3. ~~Situaciones elegidas y perfil aprendido.~~
4. Lucio responde las 150 situaciones y juega contra `lucio`. Medir cuánto se predicen; si no alcanza,
   pedir más situaciones o partidas.
5. Lucio juega contra el mejor modelo de 2 jugadores con repartos repetidos: ¿le gana?
6. Sumar perfiles y exacto a la evaluación de cada entrenamiento (`--eval_opponents` ya acepta los
   perfiles, y `tournament.py --agents` también).
7. Después del punto 5: perfil del modelo por simulación, y un explotador entrenado contra él.
