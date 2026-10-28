#!/bin/bash
# Copia a la laptop lo que hay en ~/scratch del lab (corridas, logs, colas y, con --tables, las
# tablas de la ronda 2), para no perderlo si el lab nos cambia de máquina o se borra el scratch.
# Incremental (rsync): correrlo varias veces en la noche y una última después del corte de las
# 06:55, antes de las 07:00. Queda en runs/lab/<máquina>/ (runs/ no se commitea).
#
#   scripts/lab/pull_from_lab.sh [--tables]
#
# Para retomar en otra máquina: copiar runs/lab/<máquina>/runs/<corrida> a ~/scratch/runs/ y
# usar --resume (los checkpoints de state/ son el estado completo).
set -euo pipefail
cd "$(dirname "$0")/../.."
host=$(timeout 40 ssh labs004 hostname)
dest=runs/lab/$host
mkdir -p "$dest"
opts=(-a --partial --timeout=60 -e "ssh")
rsync "${opts[@]}" --exclude 'prof*' labs004:scratch/runs/ "$dest/runs/"
rsync "${opts[@]}" labs004:scratch/logs/ "$dest/logs/"
rsync "${opts[@]}" --include '*.txt*' --include '*.sh' --exclude '*' labs004:scratch/ "$dest/"
if [ "${1:-}" = "--tables" ]; then
  rsync "${opts[@]}" --exclude '*.prev' --exclude '*.tmp' labs004:scratch/round2/ "$dest/round2/"
fi
echo "$(date +%H:%M:%S) copiado de $host a $dest ($(du -sh "$dest" | cut -f1))"
