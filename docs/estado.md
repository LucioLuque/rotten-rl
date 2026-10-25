# Estado del proyecto

Última actualización: 25/10/2026.

## Hecho

- Motor del juego, wrapper de OpenSpiel (observación de 808, 2 a 7 jugadores) y entrenamiento NFSP.
- Evaluación contra random, heurística y la versión anterior del agente, rotando asiento.
- Heurística con pesos de canto calibrados según la cantidad de jugadores.
- Entorno funcionando en el laboratorio S004 (`labo-s004-04`), unas 6 veces más rápido que la laptop.

## Corridas

| Corrida | Dónde | Config | Estado | Resultado |
|---|---|---|---|---|
| `baseline2p_20260726-230529` | otra máquina | 2 jugadores, 1M episodios | Solo llegó a la evaluación del episodio 0 | — |
| `nfsp2p_50k_20261025-175802` | lab, `~/rotten-rl/runs/` | 2 jugadores, 50k episodios, eval cada 5k con 300 partidas | En curso (tmux `nfsp2p`, log `~/scratch/logs/nfsp2p_50k.log`) | — |

## Próximos pasos

1. Analizar la corrida de 50k: curvas de win rate y de canto contra cada rival.
2. Checkpoints completos y retomar el entrenamiento desde el último (las corridas largas no entran
   en una noche por el reinicio de las 07:30).
3. Corridas largas de 2 jugadores; después 3 y 4.
4. PSRO simétrico con 3 jugadores.
5. Un modelo para n de 2 a 4; después escalar a 7.
