# Perfiles

Formas de jugar contra las que medir a los modelos, y la interfaz para jugarles. Hecho: la interfaz
sobre el estado de OpenSpiel, el guardado y los perfiles armados a mano. Falta: las partidas de Lucio y
el perfil `lucio`.

## Jugar

```bash
.venv/bin/python scripts/play.py                         # practicar contra la heurística
.venv/bin/python scripts/play.py --perfil lucio          # guardando tus decisiones
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

## Imitar a una persona

No hace falta que el perfil se pueda leer: alcanza con que juegue parecido, para medir modelos.

- **Datos:** partidas jugadas en la interfaz, no un formulario. Se guarda cada decisión.
- **Cuántos:** con 2 jugadores una partida son 62 decisiones por jugador (13 rondas, de 1 a 7 cartas y
  vuelta), muchas forzadas. Con unas 30 partidas hay del orden de 1.000 decisiones con opciones.
- **Modelo:** ajustar una red ya entrenada a esas decisiones (entrenamiento supervisado), en vez de
  empezar de cero. Usa lo que ya está: `mlp`, `info_state_tensor` y los checkpoints.
- **Validación:** guardar partidas enteras sin usarlas al ajustar y medir cuánto coincide con Lucio,
  contra la heurística y contra el modelo sin ajustar. Partidas enteras, no decisiones sueltas, para
  no medir sobre situaciones casi iguales a las del ajuste.

## La interfaz

`scripts/play.py` y `rotten_rl/renderer.py` (Pygame). Corre sobre el estado de OpenSpiel
(`rotten_rl/perfiles/partida.py`), así que el rival puede ser cualquier perfil o modelo. Se puede revisar
sin ventana, guardando capturas (`render_mode="rgb_array"`, Pygame en modo `dummy`). Muestra tu lugar en
el canto y en la baza, y por qué falta un canto.

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
3. Lucio juega contra el mejor modelo de 2 jugadores con repartos repetidos: ¿le gana?
4. Con las partidas guardadas, ajustar `lucio` y validarlo: contra una versión de reglas fijas también.
5. Sumar perfiles y exacto a la evaluación de cada entrenamiento (`--eval_opponents` ya acepta los
   perfiles, y `tournament.py --agents` también).
6. Después del punto 3: perfil del modelo por simulación, y un explotador entrenado contra él.
