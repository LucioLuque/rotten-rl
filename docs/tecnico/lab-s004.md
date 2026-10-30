# Correr en el laboratorio S004

Las corridas de entrenamiento se hacen en las máquinas del laboratorio S004 de Ingeniería (UdeSA),
no en la laptop. Documentación oficial: <https://it-ing-udesa.github.io/>. Problemas de acceso
(físico o remoto), fallas o pedidos de software: `soporte-s004@udesa.edu.ar`.

## Máquinas

22 máquinas iguales:

| | |
|---|---|
| CPU | Intel i7-14700K, 28 hilos |
| GPU | RTX 4090, 24 GB (driver 580, CUDA 13) |
| RAM | 32 GB |
| Disco local (`/scratch`) | ~670 GB libres |
| Sistema | Ubuntu 24.04, Python 3.12, `uv`, `tmux`, `nvtop`, `btop` |

## Horarios

| Cuándo | Acceso remoto |
|---|---|
| Lunes a viernes | 22:00 a 07:00. A las 07:30 se reinician las máquinas y **se pierde todo lo que esté corriendo**, incluido `tmux`. |
| Sábados, domingos y feriados | Todo el día (de viernes 22:00 a lunes 07:30 son ~57 h seguidas) |

Además, el primero de cada mes hay mantenimiento nocturno.

## Conectarse

### 1. VPN

Hace falta `openfortivpn` 1.23 o más nuevo para el login con SAML. Ubuntu 22.04 trae la 1.17 en
`apt`, así que hay que compilarlo:

```bash
sudo apt install build-essential autoconf automake pkg-config libssl-dev ppp git
git clone https://github.com/adrienverge/openfortivpn.git ~/openfortivpn && cd ~/openfortivpn
git checkout $(git describe --tags --abbrev=0)
./autogen.sh && ./configure --prefix=/usr/local --sysconfdir=/etc && make && sudo make install
```

Conectar (dejar esa terminal abierta mientras se use el lab):

```bash
sudo openfortivpn gateway.udesa.edu.ar:443 -u <usuario> --saml-login \
  --trusted-cert a30a4a249dc9e6bb3eb1b1b187160187fb77a9f87023ba78d53b3b4bc24d7b2e \
  --set-dns=0 --pppd-use-peerdns=0
```

- Se abre el navegador: Chrome muestra **"Your connection is not private"**
  (`NET::ERR_CERT_AUTHORITY_INVALID`). El certificado es legítimo (`*.udesa.edu.ar`, emitido por
  GoDaddy, vence el 27/12/2026), pero el gateway termina la cadena en la raíz nueva
  "GoDaddy TLS Root CA - R1", que Chrome y Ubuntu todavía no reconocen. Hay que ir a
  *Advanced → Proceed* y loguearse con la cuenta de la facultad.
- `--set-dns=0 --pppd-use-peerdns=0` evitan que la VPN ponga su DNS (`172.16.0.1`) primero en
  `/etc/resolv.conf`. Ese servidor no responde: cada consulta espera 5 s antes de pasar al
  siguiente, y las sesiones de Claude Code que se abren así arrancan sin los conectores de
  claude.ai (Gmail, Drive, etc.), que tienen 5 s para cargar. Sin ese DNS se siguen alcanzando las
  máquinas de UdeSA (`labs004.udesa.edu.ar` resuelve en el DNS público). Si la VPN ya se conectó
  sin esas opciones: `time getent ahosts api.anthropic.com` tarda unos 5 s, y se arregla con
  `sudo sed -i '/^nameserver 172.16.0.1$/d' /run/systemd/resolve/stub-resolv.conf` (repetirlo
  en cada reconexión) y abriendo una sesión nueva.
- Sin `--trusted-cert`, `openfortivpn` falla con `Failed to retrieve SAML cookie from HTTP`.
- Cuando el certificado se renueve (después del 27/12/2026) la huella cambia. La nueva se obtiene
  con:
  ```bash
  echo | openssl s_client -connect gateway.udesa.edu.ar:443 2>/dev/null \
    | openssl x509 -noout -fingerprint -sha256
  ```
- Listo cuando la terminal dice `Tunnel is up and running`.
- **La sesión de VPN dura como mucho 8 horas** (el 26/10/2026 se cortó sola después de 480.1
  minutos, a la 01:33). Las corridas del lab no dependen de la VPN y siguen; lo que se corta es el
  acceso para mirarlas o lanzar cosas. Si se quiere seguir de noche, conectarse más tarde o
  reconectar antes de que se cumplan las 8 horas.

### 2. SSH

```bash
ssh <usuario>@labs004.udesa.edu.ar
```

- Huella del host: `SHA256:7Z4z4U9EjmYBgcpDWe8KDzMPc6smtJ7BjrtjoKoB95s` (ED25519).
- `labs004` es un punto de entrada que asigna una máquina (`[lab-s004] slot 1 -> ...`). Según la
  documentación oficial, cada conexión va a la misma máquina, así que `~/scratch` se conserva entre
  sesiones. **Pero la asignación puede cambiar**: el 27/10/2026 una conexión a las 07:32, mientras
  la máquina de siempre (labo-s004-04) se reiniciaba, cayó en la labo-s004-17, y desde ahí el lab
  nos asigna esa. Por eso no conectarse nunca fuera de horario (después de las 07:00 en días
  hábiles), tampoco desde esperas o monitores automáticos.
- **Máximo 2 conexiones simultáneas por usuario** (`ssh`, `scp`, `rsync` cuentan cada uno).

Para entrar sin contraseña y que todas las sesiones compartan una sola conexión:

```bash
ssh-copy-id -i ~/.ssh/id_ed25519.pub <usuario>@labs004.udesa.edu.ar
```

y en `~/.ssh/config`:

```
Host labs004
    HostName labs004.udesa.edu.ar
    User <usuario>
    IdentityFile ~/.ssh/id_ed25519
    ControlMaster auto
    ControlPath ~/.ssh/cm/%r@%h:%p
    ControlPersist 10m
    ServerAliveInterval 60
```

(`mkdir -p ~/.ssh/cm && chmod 700 ~/.ssh/cm`). Después alcanza con `ssh labs004`.

Si `ssh labs004` empieza a fallar con `Cannot find current user: user: unknown userid <número>`
(pasó el 25/10/2026 con la conexión compartida abierta varias horas), la conexión compartida quedó
rota del lado del lab. Se arregla cerrándola y reconectando:

```bash
ssh -O exit labs004
ssh labs004 hostname
```

### Vigilar una noche: `scripts/lab/watch_queue.sh`

Desde la laptop, en segundo plano. Termina cuando hay algo que atender: terminó otra corrida de la
cola (`TERMINO <n>`), apareció un archivo esperado, como la salida de un torneo (`ARCHIVO`), son
las 06:35 (`HORA`: respaldo y reporte) o el lab no responde (`FALLA`). Cada consulta abre su propia
conexión (`-o ControlPath=none`), así que no la afecta que se rompa la compartida; si falla dos
veces seguidas (~5 minutos), cierra la compartida y sale con `FALLA`, en vez de seguir como si no
pasara nada. No consulta el lab entre las 06:58 y las 22:00.

```bash
scripts/lab/watch_queue.sh queue7.log 3 torneo_3p.md   # rutas relativas a ~/scratch
```

Lo motivó la noche del 29/10: el vigilante anterior descartaba los errores de `ssh` y, con la
conexión compartida rota desde las ~02:10, estuvo 3 horas sin ver nada.

## Almacenamiento

| Dónde | Tamaño | Persistencia | Para qué |
|---|---|---|---|
| `~` (NFS, compartido entre máquinas) | 2 GB y 25.000 archivos | Persistente | Código (`~/rotten-rl`), checkpoints elegidos |
| `~/scratch` (disco local) | Sin cuota | **Se borra cada cuatrimestre** | Entorno virtual, cachés, `runs/` |

**Ojo con los comandos no interactivos.** Los cachés (`XDG_CACHE_HOME`, `PIP_CACHE_DIR`,
`TMPDIR`, etc.) apuntan a `~/scratch` desde `/etc/profile.d/50-usuario.sh`, que solo se carga en
shells de login. Un `ssh labs004 'comando'` no lo carga: `uv` escribe su caché en `~/.cache` y
llena la cuota del home en segundos (pasó instalando torch). Siempre envolver en un shell de login:

```bash
ssh labs004 'bash -lc "..."'
```

## Entorno

```bash
git clone https://github.com/LucioLuque/rotten-rl.git ~/rotten-rl
uv venv --python 3.12 ~/scratch/work/venvs/rotten-rl
source ~/scratch/work/venvs/rotten-rl/bin/activate
cd ~/rotten-rl && uv pip install -e ".[dev]"
pytest
```

Se instala `open_spiel` 2.0.2, que ya trae `pytorch/nfsp.py`, y torch con CUDA
(`torch.cuda.is_available()` da `True`).

## Respaldo en la laptop

`~/scratch` es de cada máquina: si el lab nos asigna otra (pasó el 27/10, de la -04 a la -17), lo
que hay ahí deja de verse. `scripts/lab/pull_from_lab.sh [--tables]` copia con `rsync` a
`runs/lab/<máquina>/` las corridas (CSVs, snapshots y checkpoints completos), los logs, las colas
y, con `--tables`, las tablas de la ronda de 2 cartas. Es incremental (~14 MB/s por la VPN; la
primera copia, 2.3 GB, tardó 3 minutos). Correrlo durante la noche y una última vez después del
corte de las 06:55, antes de las 07:00. Para retomar en otra máquina: copiar la corrida a
`~/scratch/runs/` y usar `--resume`. Las corridas de `scripts/lab/no_state.txt` no traen `state/`
(no se retoman; las redes están en `checkpoints/`): agregar ahí las que terminen y no sigan.
Dentro de cada corrida la copia sigue al lab (los checkpoints rotados se borran también en la
laptop), pero una corrida que desaparece del lab no se borra de la laptop. Nada se borra solo:
cada noche suma ~1–2 GB de corridas más las tablas de la ronda 2 (se pisan, no se acumulan);
lo que ya no sirva se borra a mano de `runs/lab/`.

## Lanzar, cortar y retomar corridas

Las corridas van en `tmux` (sobrevive al corte de SSH y de la VPN) y con los resultados en
`~/scratch/runs`, porque los checkpoints con buffers pesan GB y el home tiene 2 GB:

```bash
tmux new -s nfsp2p
cd ~/rotten-rl && source ~/scratch/work/venvs/rotten-rl/bin/activate
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= python scripts/train.py --algo dqn --players 2 \
  --num_train_episodes 1000000 --runs_dir ~/scratch/runs --run_name nfsp2p --stop_at 06:55 \
  2>&1 | tee ~/scratch/logs/nfsp2p.log
# salir de tmux sin cortarlo: Ctrl+b y después d. Volver: tmux attach -t nfsp2p
```

- Cada `--checkpoint_every` partidas (10.000 por defecto) se guarda el estado completo en
  `<corrida>/state/ckpt_ep<N>/` (redes, optimizadores, contadores, buffers y RNGs). Se conservan los
  últimos `--keep_checkpoints` (2).
- `--stop_at 06:55` corta solo a esa hora guardando un checkpoint, antes del corte de SSH de las
  07:00 y del reinicio de las 07:30. También corta prolijo con `SIGTERM`, `SIGHUP` o `Ctrl+C`:
  termina la partida en curso, guarda y sale.
- Para seguir otra noche:
  ```bash
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= python scripts/train.py \
    --resume ~/scratch/runs/nfsp2p_<timestamp> --stop_at 06:55
  ```
  Usa la config guardada en `config.json`; solo se pueden cambiar flags que no afectan al modelo
  (`--num_train_episodes`, `--eval_every`, `--checkpoint_every`, ...).
- Verificado (25/10/2026): cortar y retomar, por `--stop_at` o por `SIGTERM`, da exactamente las
  mismas métricas que entrenar de corrido.

## Velocidad medida (25/10/2026)

Entrenamiento corto: 3 jugadores, 400 episodios, 3 evaluaciones de 30 partidas.

| Dónde | Tiempo total | Por episodio de entrenamiento |
|---|---|---|
| Laptop | 3 min 50 s | ~0.5 s |
| Lab, torch con todos los hilos | 52 s | ~0.115 s |
| Lab, `OMP_NUM_THREADS=1` | 37 s | ~0.08 s |
| Lab, loop propio sin `rl_environment` | **24 s** | **~0.053 s** |

- Con redes tan chicas, torch multihilo es **más lento**: correr siempre con
  `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`.
- Las corridas son de CPU, pero si la GPU está visible torch crea un contexto CUDA igual: ~386 MB de
  VRAM y ~90 MB de RAM por corrida, sin ningún uso. Con `CUDA_VISIBLE_DEVICES=` no lo crea y los
  resultados son idénticos (medido el 25/10/2026). `run_queue.py` ya lo pone.
- Así cada corrida usa un solo núcleo, y se pueden correr varias en paralelo (semillas,
  configuraciones, cantidad de jugadores). Por ser máquinas compartidas, no ocupar todos los hilos:
  mirar `btop` antes de lanzar.
- Estimación con el loop propio: 1M de episodios con 3 jugadores ≈ 15 h. En una noche de semana
  entran ~550k.
- Memoria: una corrida NFSP de 2 jugadores con 50k episodios usa ~5.5 GB, y crece con el reservoir
  (hasta ~7 GB por agente con la capacidad de 2M). Es lo que limita cuántas corren a la vez.
- La GPU por ahora no aporta: el cuello de botella es simular el juego en Python.
