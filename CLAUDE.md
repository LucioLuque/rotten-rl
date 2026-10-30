# rotten-rl

RL para La Podrida (Oh Hell) sobre OpenSpiel. Estructura y uso en `README.md`; plan y métricas en
`docs/plan.md`; estado actual, corridas y próximos pasos en `docs/estado.md` (leerlo al empezar y
actualizarlo al terminar algo); bibliografía y qué aplicar en `docs/bibliografia.md`; un reporte
por noche de corridas en `docs/reportes/<fecha en que empieza la noche>.md`; índice de todo en
`docs/README.md`. Mantener actualizados todos los `.md` que cambien con el trabajo.

Orden del repo: cada tema nuevo en su carpeta (código en `rotten_rl/<tema>/`, scripts en
`scripts/<tema>/`, docs en `docs/<tema>/` con un `README.md` que explique qué es), no como
archivos sueltos. Arriba de `scripts/` solo lo principal (`train.py`, `run_queue.py`, `play.py`).

Entrenamiento: `scripts/train.py --algo nfsp|dqn|ppo`; cada algoritmo en `rotten_rl/algorithms/`
implementa la interfaz de `base.py` (evaluación, logging, checkpoints y cola son comunes).

## Dónde se corre

- **No entrenar en la laptop.** Las corridas van al laboratorio S004; todo el detalle está en
  `docs/tecnico/lab-s004.md`.
- La VPN la levanta Claude, sin pedírsela al usuario: `~/.local/bin/udesa-vpn-connect` (~3 s, no
  pide nada) y después `setsid nohup ~/.local/bin/udesa-vpn-watch > /dev/null 2>&1 &`, que la
  reconecta si se cae y corre hasta las 07:00 (detalle en `NOTAS_LOCALES.md`, fuera de git). Con
  la VPN arriba, `ssh labs004` entra sin contraseña (alias en `~/.ssh/config` de la laptop).
- Cada noche ver `hostname`: el lab puede asignar otra máquina (-04 hasta el 27/10, -17 el 27,
  -19 el 28). Si cambió, armar el entorno de cero (`docs/tecnico/lab-s004.md`, "Entorno") y subir
  desde `runs/lab/<máquina anterior>/` lo que haya que retomar.
- Las corridas viven en `tmux` en el lab: siguen aunque se corte la VPN o se apague la laptop.
- Comandos remotos siempre como `ssh labs004 'bash -lc "..."'`: sin shell de login no se cargan
  los cachés en `~/scratch` y se llena la cuota de 2 GB del home.
- Entrenar con `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=` (más rápido con estas
  redes; sin GPU visible torch no reserva VRAM ni ~90 MB de RAM por corrida) y dentro de
  `tmux`, con `--runs_dir ~/scratch/runs`, `--stop_at 06:55` y el log en
  `~/scratch/logs/<corrida>.log`. Las corridas largas se retoman con `--resume <corrida>`.
- Nunca buscar procesos por patrón de texto (`pkill -f`, `pgrep -f`) desde un `ssh labs004 '...'`:
  el patrón aparece también en el comando de la propia sesión y coincide con ella (pasó dos veces:
  un `pkill` mató la sesión y un `pgrep` dejó una espera infinita). Usar el PID, o
  `run_queue.count_training_processes()` para contar entrenamientos.
- Si `ssh labs004` empieza a devolver `Cannot find current user: user: unknown userid`, la conexión
  compartida (ControlMaster) quedó rota del lado del lab: `ssh -O exit labs004` y reconectar.
- **Vigilar una noche siempre con `scripts/lab/watch_queue.sh`**, nunca con un loop propio que
  descarte errores: consulta con conexión propia (no la compartida), sale con `FALLA` si el lab
  no responde ~5 minutos y con `HORA` a las 06:35. Ante `FALLA`, arreglar la conexión y relanzarlo
  en el momento. Nunca dar por "no pasó nada" una consulta que falló. (El 29/10 un vigilante con
  `2>/dev/null` estuvo 3 horas ciego por la conexión compartida rota y no se reemplazaron 7
  corridas.)
- Horario remoto: lunes a viernes de 22:00 a 07:00 (reinicio 07:30, se pierde todo lo que corre);
  fines de semana y feriados todo el día. Máximo 2 conexiones simultáneas.
- **Nada se conecta al lab fuera de horario**: toda espera o monitor que haga `ssh labs004` tiene
  que terminar a las 07:00 en días hábiles. El 27/10 una espera siguió y a las 07:32, con la -04
  reiniciando, el lab nos asignó otra máquina (la -17) y quedamos sin acceso a `~/scratch` de la -04.

## En el laboratorio

- Repo en `~/rotten-rl`, entorno en `~/scratch/work/venvs/rotten-rl` (`open_spiel` 2.0.2).
- NFSP de OpenSpiel 2.x es una reescritura de enero de 2026; sus defaults de exploración del DQN
  (ε de 1.0 a 0.1 en 1M pasos) no sirven para NFSP. El script los fija con `--epsilon_*`.

## Convenciones

- Código, comentarios y docs en castellano; identificadores en inglés.
- `runs/` y `checkpoints/` no se commitean.
- Tests con `pytest` (los de OpenSpiel se saltean si no está `pyspiel`). En la laptop hay un
  entorno en `.venv` (OpenSpiel y torch de CPU, sin GPU): `.venv/bin/python -m pytest -p no:anyio`
  (el pytest del sistema choca con el plugin de anyio). Sirve para tests y para medir código de
  CPU, no para entrenar.
