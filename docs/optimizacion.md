# Optimización

Qué queda por optimizar en velocidad y memoria del entrenamiento, y cuándo conviene. Lo ya hecho
(loop propio, buffers comprimidos, partidas en lote con la red en la GPU, evaluación en paralelo,
replay buffer de DQN en la GPU) está en `docs/estado.md` y en los reportes de cada noche.

## Dónde se va el tiempo hoy (26/10/2026)

DQN y PPO compartidos con `--vector_games 32 --device cuda`, 3 jugadores, en el lab (i7-14700K
con 8 núcleos rápidos y 12 lentos, RTX 4090): ~16–30 ms por partida si corren pocas a la vez. Con
8 a la vez en la misma GPU, ~37 ms cada una y ~220 partidas por segundo en total; con 12, apenas
más (~238). Perfil de PPO (cProfile, 1.280 partidas):

| Parte | % aprox. |
|---|---|
| Actualización de PPO (backprop en GPU) | 27% |
| Decidir en lote (una llamada a la GPU por paso, ~1 ms, casi todo latencia) | 25% |
| Armar la observación de 808 valores (`info_state_tensor`, ~26 µs por decisión en Python) | 22% |
| Simular el juego (jugadas, acciones legales, reparto) | 18% |

El juego en Python (observación más simulación) pasó de ~7% a ~40%. Con 7 jugadores pesa más
(más decisiones por partida).

## Pendiente

1. **Armar la observación más rápido.** Hoy se arma de cero en Python en cada decisión. Se puede
   armar de forma incremental (casi todo cambia poco entre una decisión y la siguiente), con numpy
   vectorizado o con Numba. Podría bajar ese 22% a la mitad o menos. Medio día. Hacerlo antes de
   pasar a 5–7 jugadores.
2. **Simular el juego más rápido.** Conversiones de cartas (`_card_id`, `card_to_id`, millones de
   llamadas), `legal_action_ids` calculado dos veces por paso, `public_state` en cada jugada. Unos
   arreglos chicos en `engine.py` y `openspiel_game.py`, quizás 30–50% de ese 18%.
3. **Usar varias máquinas del laboratorio.** La palanca más grande: hay 22 máquinas; si las otras
   tienen la misma GPU, casi ×22. Hay que ver si se puede saltar de una a otra desde adentro del
   lab (el home es NFS compartido) y si está bien usar varias a la vez (son compartidas): lo
   decide Lucio.
4. **Varios procesos en una GPU.** Con 8 corridas la GPU se reparte por turnos (cada proceso manda
   muchas operaciones chicas). Opciones: CUDA MPS (necesita un servicio corriendo en una máquina
   compartida: no por ahora), CUDA graphs o `torch.compile` para la red de decisión (ganancia
   incierta), o entrenar varias semillas en un mismo proceso. Probado: 64 partidas a la vez en vez
   de 32 no cambia la velocidad.
5. **TF32 en la GPU** (`torch.backends.cuda.matmul.allow_tf32`). Una línea; poca ganancia con
   redes de 512.
6. **NFSP en lote.** NFSP (el de OpenSpiel) sigue jugando de a una partida. Solo si vuelve a ser
   un algoritmo principal.
7. **El juego en C++.** Solo si con 7 jugadores o PSRO el juego sigue siendo el límite después de
   1 y 2.

Recordatorio: más que la velocidad, importa cuánto aprende el agente por partida
(hiperparámetros, algoritmo). Una mejora ahí rinde más que cualquiera de estas.
