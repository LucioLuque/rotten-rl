# Optimización

Qué queda por optimizar en velocidad y memoria del entrenamiento, y cuándo conviene. Lo ya hecho
(loop propio, buffers comprimidos, partidas en lote con la red en la GPU, evaluación en paralelo,
replay buffer de DQN en la GPU, observación más rápida) está en `docs/estado.md` y en los
reportes de cada noche.

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

El 27/10 se aceleró el armado de la observación (medido en la laptop, sin la red): de 36.6 a
25.4 µs con 3 jugadores y de 61.2 a 32.5 µs con 7, con el mismo resultado bit a bit. La partida
completa (simulación más observación) bajó ~8–10%: ahora la simulación pesa más que la
observación (~50–60 µs contra ~25–33 por decisión en la laptop).

## Pendiente

1. **Simular el juego más rápido.** Es lo que más pesa ahora en el juego. `legal_action_ids` se
   calcula dos veces por paso (para validar en `apply_action` y para pyspiel), `public_state` se
   arma en cada jugada aunque nadie lo use, `card_to_id` usa `tuple.index` y cada llamada de
   pyspiel a Python tiene un costo fijo. Arreglos chicos en `engine.py` y `openspiel_game.py`,
   quizás 30–50% de la simulación.
2. **Observación: lo que queda.** Ya no hay un punto caro: el tiempo está repartido en muchas
   líneas cortas (unas 60 operaciones de Python por decisión) más ~10 µs de escrituras en numpy.
   Para bajar mucho más habría que armar las observaciones de todas las partidas del lote juntas
   en numpy, o pasar el motor a arrays con Numba (ver el punto 7). Solo si el juego vuelve a ser
   el límite.
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
7. **El juego en C++.** Python sirvió para entender y armar el juego; para velocidad no hay
   razón para quedarse ahí. Cuánto rinde depende de para qué:
   - **Entrenamiento con la red en la GPU**: el juego (simulación más observación) es ~35% del
     tiempo con 3 jugadores, así que con C++ iría ~1.5 veces más rápido (la mitad del tiempo es la
     red, que no cambia); con 7 jugadores pesa más, quizás ~2 veces. Medirlo con las corridas de
     7 (27/10): si el juego pasa de ~40% del tiempo, hacerlo antes de las corridas largas de 7.
     **Medido (27/10, `docs/reportes/2026-10-27.md`)**: con 7 jugadores y 8 corridas compartiendo
     la GPU, el juego es ~25–30% del tiempo y esperar a la GPU ~35%. No es urgente.
   - **Lo que no usa la red** (PIMC con búsqueda, CFR de la ronda de 2 cartas, torneos grandes):
     ahí es imprescindible, la diferencia puede ser de 50 a 100 veces.
   - Cómo: OpenSpiel trae un Oh Hell en C++ (`oh_hell`), pero hay que verificar sus reglas (hasta
     donde se sabe juega una sola ronda, sin la escalera, con otro puntaje y sin la obligación de
     fallar); lo probable es escribir el nuestro basado en ese. Se valida contra lo que ya hay:
     `tests/test_rules.py` y las observaciones de referencia (`tests/data/info_states.npz`), que
     tienen que dar idéntico. Falta ver que el lab tenga compilador. ~1–2 días.

Recordatorio: más que la velocidad, importa cuánto aprende el agente por partida
(hiperparámetros, algoritmo). Una mejora ahí rinde más que cualquiera de estas.
