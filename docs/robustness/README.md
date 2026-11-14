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
Después la misma cola sigue con la red de 1024 (`docs/net_size/`).
