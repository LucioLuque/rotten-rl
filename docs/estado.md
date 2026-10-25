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
| `nfsp2p_50k_20261025-175802` | lab | NFSP, 2 jugadores, 50k episodios, **con los dos bugs** | Terminada (49 min) | Plana: vs random 0.51, vs heurística 0.22, acierto de canto 0.25, igual que sin entrenar |
| `diag_dqn2p` | lab | DQN solo vs random, 2 jugadores, 20k episodios, reward arreglado | Terminada (20 min) | vs random 0.71, vs heurística 0.44 (de 0.14), acierto de canto 0.37 (de 0.26); canta de más (sesgo +0.8 vs heurística) |
| `nfsp2p_50k_fix_20261025-184740` | lab | NFSP, 2 jugadores, 50k episodios, con los dos arreglos | En curso (tmux `nfspfix`) | — |

Los resultados de cada corrida están en `~/rotten-rl/runs/<corrida>/` y el log en
`~/scratch/logs/`.

## Bugs encontrados (25/10/2026)

1. **Reward perdido entre turnos.** El juego da reward al cerrar cada ronda, pero cada agente solo
   ve los TimeSteps de sus turnos y `rl_environment` no acumula: el reward le llegaba solo al que
   jugaba justo después del cierre. Se perdía el 46% del reward con 2 jugadores, 69% con 4 y 74% con
   7. Arreglado con `rotten_rl.training.with_pending_reward`; lo cubre un test.
2. **Exploración del DQN de NFSP.** OpenSpiel 2.x usa por defecto ε de 1.0 a 0.1 en 1M pasos: la
   best-response jugaba casi al azar y la avg-policy lo imitaba. Ahora el script usa 0.06 → 0.001
   (valores del paper).

## Próximos pasos

1. Ver si NFSP aprende con los dos arreglos (`nfsp2p_50k_fix`).
2. ~~Loop de entrenamiento propio y checkpoints~~ Hecho: `rotten_rl/training.py`
   (`EpisodeRunner`, 1.65× más rápido) y `rotten_rl/checkpoint.py`; `train_nfsp.py` tiene
   `--resume`, `--stop_at` y corte prolijo por señal. Cortar y retomar da resultados idénticos.
3. Bajar la memoria de NFSP: una corrida de 50k ya usa 5.5 GB y el reservoir de 2M podría llegar a
   ~7 GB por agente.
4. Corridas largas de 2 jugadores; después 3 y 4.
4. PSRO simétrico con 3 jugadores.
5. Un modelo para n de 2 a 4; después escalar a 7.
