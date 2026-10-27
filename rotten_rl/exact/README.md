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

## La idea, en corto

Para que sea viable se agrupan las situaciones que son iguales: los tres palos que no son
triunfo son intercambiables, y del triunfo dado vuelta solo importa el número. Con 1 carta eso
deja 13 × 26 × (2ⁿ − 1) situaciones de canto (41.000 con 7 jugadores) en vez de cientos de miles
de millones de repartos. Sobre eso:

- **Equilibrio con CFR+**: la forma de cantar en la que nadie gana cambiando la suya (el
  algoritmo con el que se resolvió el póker).
- **Mejor respuesta**: qué conviene cantar contra una forma concreta de jugar de los demás (la
  heurística, el equilibrio, el agente), con toda la información o mirando solo la carta propia.

Con 2 cartas o más hay jugadas para elegir y el problema crece mucho: 2 cartas parece viable con
MCCFR y más trabajo; de 3 en adelante ya no se puede de forma exacta (ver
`docs/primera-ronda/README.md`, "Pendiente").

## Cómo se usa

```bash
python scripts/exact/solve_round1.py      # 2 a 7 jugadores, ~30–40 min en CPU
```

Escribe `docs/primera-ronda/resultados.md` y las tablas en `docs/primera-ronda/equilibrio.npz`.
La explicación completa y los resultados están en `docs/primera-ronda/`. Tests:
`tests/test_round1.py`.
