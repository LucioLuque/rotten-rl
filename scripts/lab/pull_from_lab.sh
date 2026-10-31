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
# La conexión compartida (ControlMaster) a veces queda rota del lado del lab ("unknown userid"):
# se cierra y se reconecta.
if ! host=$(timeout 40 ssh labs004 hostname 2>/dev/null); then
  ssh -O exit labs004 >/dev/null 2>&1 || true
  host=$(timeout 40 ssh labs004 hostname)
fi
dest=runs/lab/$host
mkdir -p "$dest"
opts=(-a --partial --timeout=60 -e "ssh")
# Código 24 = archivos que desaparecieron durante la copia (checkpoints que una corrida rotó): no
# es un error.
rsync() { command rsync "$@" || [ $? -eq 24 ]; }
# Only the newest checkpoint of each run comes to the laptop: the older ones the lab keeps are
# excluded by name. --delete-after removes the previous copy once the new one has arrived, so a cut
# in the middle leaves the old one. Checkpoints are renamed from .tmp when complete, so *.tmp is
# skipped. 'P /*' protects the run folders: a run that disappears from the lab (scratch wiped) is
# not deleted here. no_state.txt: runs that will not resume, whose state/ is skipped.
older=$(mktemp)
trap 'rm -f "$older"' EXIT
timeout 40 ssh labs004 'cd ~/scratch/runs && ls -d */state/ckpt_ep* 2>/dev/null' | grep -v '[.]tmp$' \
  | sort -V | awk -F/ '{ if ($1 == run) print "/" prev "/"; run = $1; prev = $0 }' > "$older" || true
rsync "${opts[@]}" --delete --delete-after --filter 'P /*' --exclude 'prof*' --exclude '*.tmp' \
  --exclude-from "$older" --exclude-from scripts/lab/no_state.txt labs004:scratch/runs/ "$dest/runs/"
# The older copies stay out of --delete because they are excluded: drop them here, keeping the
# newest complete one of each run.
for state in "$dest"/runs/*/state; do
  [ -d "$state" ] || continue
  newest=$(ls -d "$state"/ckpt_ep* 2>/dev/null | grep -v '[.]tmp$' | sort -V \
    | while read -r c; do if [ -f "$c/meta.pt" ]; then echo "$c"; fi; done | tail -1 || true)
  [ -n "$newest" ] || continue
  # grep exits 1 when the newest is the only one: nothing to drop.
  ls -d "$state"/ckpt_ep* | { grep -vxF "$newest" || true; } | xargs -r rm -rf
done
rsync "${opts[@]}" labs004:scratch/logs/ "$dest/logs/"
rsync "${opts[@]}" --include '*.txt*' --include '*.sh' --exclude '*' labs004:scratch/ "$dest/"
if [ "${1:-}" = "--tables" ]; then
  rsync "${opts[@]}" --exclude '*.prev' --exclude '*.tmp' --exclude 'n3_exact.bin' labs004:scratch/round2/ "$dest/round2/"
fi
echo "$(date +%H:%M:%S) copiado de $host a $dest ($(du -sh "$dest" | cut -f1))"
