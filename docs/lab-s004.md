# Correr en el laboratorio S004

Las corridas de entrenamiento se hacen en las máquinas del laboratorio S004 de Ingeniería (UdeSA),
no en la laptop. Documentación oficial: <https://it-ing-udesa.github.io/>.

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
  --trusted-cert a30a4a249dc9e6bb3eb1b1b187160187fb77a9f87023ba78d53b3b4bc24d7b2e
```

- Se abre el navegador: Chrome muestra **"Your connection is not private"**
  (`NET::ERR_CERT_AUTHORITY_INVALID`). El certificado es legítimo (`*.udesa.edu.ar`, emitido por
  GoDaddy, vence el 27/12/2026), pero el gateway termina la cadena en la raíz nueva
  "GoDaddy TLS Root CA - R1", que Chrome y Ubuntu todavía no reconocen. Hay que ir a
  *Advanced → Proceed* y loguearse con la cuenta de la facultad.
- Sin `--trusted-cert`, `openfortivpn` falla con `Failed to retrieve SAML cookie from HTTP`.
- Cuando el certificado se renueve (después del 27/12/2026) la huella cambia. La nueva se obtiene
  con:
  ```bash
  echo | openssl s_client -connect gateway.udesa.edu.ar:443 2>/dev/null \
    | openssl x509 -noout -fingerprint -sha256
  ```
- Listo cuando la terminal dice `Tunnel is up and running`.

### 2. SSH

```bash
ssh <usuario>@labs004.udesa.edu.ar
```

- Huella del host: `SHA256:7Z4z4U9EjmYBgcpDWe8KDzMPc6smtJ7BjrtjoKoB95s` (ED25519).
- `labs004` es un punto de entrada que asigna una máquina (`[lab-s004] slot 1 -> ...`). Siempre
  asigna la misma, así que `~/scratch` se conserva entre sesiones.
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
curl -sL https://raw.githubusercontent.com/google-deepmind/open_spiel/master/open_spiel/python/pytorch/nfsp.py \
  -o "$(python -c 'import open_spiel,os;print(os.path.dirname(open_spiel.__file__))')/python/pytorch/nfsp.py"
pytest
```

torch se instala con CUDA (`torch.cuda.is_available()` da `True`).

## Velocidad medida (25/10/2026)

Entrenamiento corto: 3 jugadores, 400 episodios, 3 evaluaciones de 30 partidas.

| Dónde | Tiempo total | Por episodio de entrenamiento |
|---|---|---|
| Laptop | 3 min 50 s | ~0.5 s |
| Lab, torch con todos los hilos | 52 s | ~0.115 s |
| Lab, `OMP_NUM_THREADS=1` | **37 s** | **~0.08 s** |

- Con redes tan chicas, torch multihilo es **más lento**: correr siempre con
  `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`.
- Así cada corrida usa un solo núcleo, y se pueden correr varias en paralelo (semillas,
  configuraciones, cantidad de jugadores). Por ser máquinas compartidas, no ocupar todos los hilos:
  mirar `btop` antes de lanzar.
- Estimación: 1M de episodios con 3 jugadores ≈ 23 h. Entra en un fin de semana; en una noche de
  semana entran ~350k.
- La GPU por ahora no aporta: el cuello de botella es simular el juego en Python.
