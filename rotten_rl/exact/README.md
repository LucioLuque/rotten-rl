# exact: partes del juego resueltas sin RL

El agente de RL aprende a jugar la partida entera, pero no hay forma de saber si juega bien en
general: no conocemos la respuesta correcta. Esta carpeta junta las partes del juego que son lo
bastante chicas para **calcular cómo conviene jugar de forma exacta** (o casi), con dos
objetivos:

- **Entender el juego**: cuánto pesa la posición, cuánto conviene escuchar los cantos de los
  demás, cuánto se le saca a una mesa que juega mal.
- **Controlar al agente**: comparar lo que hace la red con la respuesta calculada, en los únicos
  casos donde la tenemos. Si se equivoca acá, probablemente también aprende mal en el resto.

## Qué hay

| Archivo | Qué resuelve |
|---|---|
| `round1.py` | La ronda de 1 carta (primera y última de cada partida), de 2 a 7 jugadores |
| `round2/` | La ronda de 2 cartas, en C++ (MCCFR); hoy 3 jugadores ([README](round2/README.md)) |

## La idea, en corto

Para que sea viable se agrupan las situaciones que son iguales: los tres palos que no son
triunfo son intercambiables, y del triunfo dado vuelta solo importa el número. Con 1 carta eso
deja 13 × 26 × (2ⁿ − 1) situaciones de canto (41.000 con 7 jugadores) en vez de cientos de miles
de millones de repartos. Sobre eso:

- **Equilibrio con CFR+**: la forma de cantar en la que nadie gana cambiando la suya (el
  algoritmo con el que se resolvió el póker).
- **Mejor respuesta**: qué conviene cantar contra una forma concreta de jugar de los demás (la
  heurística, el equilibrio, el agente), con toda la información o mirando solo la carta propia.

Con 2 cartas ya hay jugadas para elegir y el problema crece mucho: se resuelve en C++ agrupando
también las cartas jugadas en la baza (`round2/`, `docs/segunda-ronda/`). De 3 cartas en adelante
ya no se puede de forma exacta.

## Cómo se usa

```bash
python scripts/exact/solve_round1.py      # ronda de 1 carta, 2 a 7 jugadores, ~45 min en CPU
make -C rotten_rl/exact/round2            # ronda de 2 cartas: ver round2/README.md
```

Escribe `docs/primera-ronda/resultados.md` y las tablas en `docs/primera-ronda/equilibrio.npz`.
La explicación completa y los resultados están en `docs/primera-ronda/`. Tests:
`tests/test_round1.py`.
