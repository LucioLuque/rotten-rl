# rotten-rl

RL para La Podrida (Oh Hell) sobre OpenSpiel. Estructura y uso en `README.md`; plan y métricas en
`docs/plan.md`; estado actual, corridas y próximos pasos en `docs/estado.md` (leerlo al empezar y
actualizarlo al terminar algo).

## Dónde se corre

- **No entrenar en la laptop.** Las corridas van al laboratorio S004; todo el detalle está en
  `docs/lab-s004.md`.
- La VPN la levanta el usuario (necesita `sudo` y login en el navegador). Con la VPN arriba,
  `ssh labs004` entra sin contraseña (alias en `~/.ssh/config` de la laptop).
- Comandos remotos siempre como `ssh labs004 'bash -lc "..."'`: sin shell de login no se cargan
  los cachés en `~/scratch` y se llena la cuota de 2 GB del home.
- Entrenar con `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` (más rápido con estas redes) y dentro de
  `tmux`, con `--runs_dir ~/scratch/runs`, `--stop_at 06:55` y el log en
  `~/scratch/logs/<corrida>.log`. Las corridas largas se retoman con `--resume <corrida>`.
- Al matar procesos, nunca `pkill -f` con un patrón que también aparezca en el comando de la
  propia sesión SSH (se mata a sí misma). Usar el PID.
- Horario remoto: lunes a viernes de 22:00 a 07:00 (reinicio 07:30, se pierde todo lo que corre);
  fines de semana y feriados todo el día. Máximo 2 conexiones simultáneas.

## En el laboratorio

- Repo en `~/rotten-rl`, entorno en `~/scratch/work/venvs/rotten-rl` (`open_spiel` 2.0.2).
- NFSP de OpenSpiel 2.x es una reescritura de enero de 2026; sus defaults de exploración del DQN
  (ε de 1.0 a 0.1 en 1M pasos) no sirven para NFSP. El script los fija con `--epsilon_*`.

## Convenciones

- Código, comentarios y docs en castellano; identificadores en inglés.
- `runs/` y `checkpoints/` no se commitean.
- Tests con `pytest` (los de OpenSpiel se saltean si no está `pyspiel`).
