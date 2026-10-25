# Optimización

Dónde se va el tiempo y la memoria del entrenamiento, y qué conviene optimizar (y qué no).

## Tiempo (medido el 25/10/2026 en el lab, NFSP 2 jugadores, 1 hilo)

| Versión | ms por episodio |
|---|---|
| Con `rl_environment` | ~46 |
| Con el loop propio (`rotten_rl.training.EpisodeRunner`) | **~34** |

Reparto con el loop propio (cProfile, 300 episodios):

| Parte | % |
|---|---|
| PyTorch: pasadas de la red, backward, Adam | 47% |
| NFSP / DQN de OpenSpiel (Python) | 19% |
| pyspiel y builtins | 11% |
| Wrapper + observación (`openspiel_game.py`) | 10% |
| Buffers y conversiones (numpy, dm-tree) | 7% |
| Motor del juego (`engine.py`) | 5% |

El juego (motor + observación) ya es ~15% del tiempo. La mitad se va en PyTorch, casi todo en
pasadas de la red **de a un ejemplo**: una por decisión, ~124 por partida con 2 jugadores.

## Memoria

Cada agente NFSP tiene sus propios buffers, guardados en `float32`:

| Buffer | Capacidad | Por muestra | Lleno |
|---|---|---|---|
| Reservoir (avg-policy) | 2M | info-state 808×4 B + probs 60×4 B + máscara 60 B ≈ 3.5 KB | ~7 GB |
| Replay (DQN) | 200k | 2 info-states + resto ≈ 6.6 KB | ~1.3 GB |

En una corrida larga los dos se llenan: **~8.3 GB por agente**. Con 2 jugadores son ~17 GB (entra
una corrida larga, no dos a la vez); con 3, ~25 GB (justo); con 4, ~33 GB: **no entra** en los
31 GB de las máquinas del lab. Una corrida de 50k episodios con 2 jugadores usa ~5.5 GB.

## Opciones

1. **Buffers más chicos en memoria.** El info-state es casi todo 0/1: guardarlo en `float16` (o
   `uint8` para las partes one-hot) reduce los buffers 2× a 4×. Poco trabajo.
   **Obligatorio antes de las corridas largas con 3 o más jugadores** (con 4 no entran), y útil
   para correr varias corridas largas de 2 jugadores en paralelo. Mientras tanto, se puede bajar
   `--reservoir_buffer_capacity`.
2. **Arreglos chicos en el camino caliente**: forward en `torch.inference_mode`, buffer propio sin
   dm-tree, observación más eficiente. 5–10% cada uno, quizás 20–30% en total; implica tocar o
   reemplazar partes del NFSP de OpenSpiel. No por ahora.
3. **Muchas partidas en paralelo con forward en lote** (vectorizar): una pasada de la red para 64
   decisiones en vez de 64 pasadas de una. Es lo único que ataca el 47% de PyTorch, y recién ahí
   sirve la GPU. Podría dar 2× a 5×, pero hay que reescribir NFSP (el de OpenSpiel procesa los
   pasos de a uno). Más adelante, si la velocidad pasa a ser el límite (PSRO, 7 jugadores).
4. **El juego en C++.** Ahorraría como mucho ~15%. No vale la pena.

Más que la velocidad del código, hoy importa cuánto aprende el agente por episodio
(hiperparámetros, algoritmo): una mejora ahí rinde más que cualquiera de estas optimizaciones.
