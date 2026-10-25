# Contribuciones a OpenSpiel

Problemas de OpenSpiel que encontramos usándolo en este proyecto, y qué hacer con cada uno.

## Requisitos para contribuir

Según el
[CONTRIBUTING.md](https://github.com/google-deepmind/open_spiel/blob/master/CONTRIBUTING.md) y
[docs/contributing.md](https://github.com/google-deepmind/open_spiel/blob/master/docs/contributing.md):

- Firmar el CLA de Google (una sola vez): <https://cla.developers.google.com/>. El mail de los
  commits tiene que estar asociado a la cuenta que firmó.
- Arreglos chicos: PR directo. Cambios grandes o con decisiones de diseño: abrir primero un issue.
- Estilo: guía de Python de Google (2 espacios de sangría, como el resto del código).
- Mergean en tandas, al menos cada dos semanas; los arreglos de bugs, más rápido.

Estado del CLA: **pendiente de firmar**.

## 1. Guardar y cargar DQN y NFSP de PyTorch está roto (2.0.2 y master)

Reproducido el 25/10/2026 en el laboratorio:

- `dqn.DQN.save` guarda las claves `"model"`/`"optimiser"` y `load` lee
  `"model_state_dict"`/`"optimizer_state_dict"`: `KeyError: 'model_state_dict'`.
- `nfsp.NFSP.save(carpeta)` usa la misma ruta como carpeta (para el DQN interno) y como archivo
  (para `torch.save`): falla si la carpeta no existe y también si existe. `restore` tiene el mismo
  problema de claves.
- Viene de la reescritura de enero de 2026 (PR #1419).

**Ya está reportado y arreglado en un PR de otra persona:**
[#1620 "Fix PyTorch DQN and NFSP checkpointing"](https://github.com/google-deepmind/open_spiel/pull/1620)
(`ArchawinL`, abierto el 18/09/2026, esperando revisión). Le dimos 👍 el 25/10/2026. No hay que
mandar otro PR.

En este repo lo esquivamos con `rotten_rl/checkpoint.py`, que guarda todo a mano.

## 2. Propuesta nuestra: guardar los buffers en los checkpoints (pendiente)

El #1620 dice explícitamente "**Not included:** saving the replay and reservoir buffers". Sin los
buffers, retomar un entrenamiento de NFSP no es lo mismo que seguirlo: el reservoir de la política
promedio se vacía y hay que volver a llenarlo.

Nosotros ya lo tenemos resuelto en `rotten_rl/checkpoint.py`:

- Guarda solo la parte llena de cada buffer (`ReservoirBuffer` de NFSP y `ReplayBuffer` de DQN), un
  `.npy` por campo, más sus contadores (`add_calls`, `entry_index`).
- Guarda también los contadores del DQN interno (que controlan ε y la red target) y los estados de
  los RNG.
- `tests/test_checkpoint.py` verifica que cortar y retomar da pesos y partidas idénticos a entrenar
  de corrido.

Plan:

1. Esperar a que mergeen el #1620: nuestro cambio se tiene que apoyar en su estructura de archivos
   (`q_network.pt` y `avg_network.pt` dentro de la carpeta del checkpoint).
2. Firmar el CLA.
3. Abrir un issue proponiendo guardar los buffers (es una decisión de diseño: pueden pesar varios
   GB, así que probablemente deba ser opcional, p.ej. `save(..., save_buffers=False)`).
4. Si les interesa, mandar el PR adaptando `checkpoint.py` a su estilo, con un test de ida y vuelta
   en `nfsp_pytorch_test.py` y `dqn_pytorch_test.py`.

## Otras cosas que encontramos (no son bugs)

- **ε por defecto de NFSP.** NFSP hereda la exploración por defecto del DQN (ε de 1.0 a 0.1 en 1M
  pasos), que no sirve para NFSP. Sus ejemplos (`nfsp_leduc_pytorch.py`) pasan 0.06 → 0.001 a mano.
  Es una trampa fácil de pisar; podría proponerse que NFSP tenga esos valores por defecto.
- **Rewards intermedios con `rl_environment`.** Los agentes solo ven los TimeSteps de sus turnos y
  `rl_environment` no acumula rewards entre turnos, así que en juegos con reward antes del final se
  pierden (en este juego, 46% a 74% según la cantidad de jugadores). Casi todos los juegos de
  OpenSpiel dan reward solo al final. Podría valer un issue para documentarlo.
- **Info-state de juegos en Python.** Pedirlo por `pyspiel` copia el estado y arma el tensor dos
  veces. Es correcto, pero lento; acá lo evitamos con `openspiel_game.info_state_tensor`.
