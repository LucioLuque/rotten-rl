#!/bin/bash
# scripts/lab/relaunch_queue.sh  —  When the lab reopens: queue what was left and watch it (run on the server).
#
#   scripts/lab/relaunch_queue.sh <old queue> <new queue> [<older queue>]
#
# On the lab machine of LAB_SSH (default labs004b), builds ~/scratch/<new>.txt with what is left of
# ~/scratch/<old>.txt (scripts/lab/remaining_queue.py over its log: cut runs resumed, finished ones
# out) and starts it in tmux (start_queue.sh). If <old>.txt is not there (the previous relaunch
# failed), uses <older> instead. Then, on the server, the watcher (podrida-watch) and the backup
# (podrida-backup, until the lab closes) as systemd units, and a notice to the phone.
# Meant for a persistent systemd timer at the time the lab reopens; never ssh while it is closed.
set -u
old=$1 new=$2 older=${3:-}
LAB=${LAB_SSH:-labs004b} HOST=${WATCH_HOST:-labo-s004-14}
cd "$(dirname "$0")/../.."
R=$PWD; L=$R/runs/lab/$HOST/logs; mkdir -p "$L"
E="-E HOME=$HOME -E PATH=$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin -E LAB_SSH=$LAB"
st=$(env -u PYTHONPATH python3 -m rotten_rl.lab_hours state)
[ "$st" = open ] || { avisar "$new" "el lab no está abierto ($st): no lancé nada" 4; exit 1; }
vpn on
SSH="timeout 120 ssh -o ControlPath=none $LAB"
ok=0
for i in 1 2 3 4; do
  $SSH "bash -lc 'cd ~/rotten-rl && git pull -q; q=$old; [ -f ~/scratch/\$q.txt ] || q=$older; \
python3 scripts/lab/remaining_queue.py --queue ~/scratch/\$q.txt --log ~/scratch/logs/\$q.log \
> ~/scratch/$new.txt && cat ~/scratch/$new.txt'" > "$L/${new}_generada.txt" && { ok=1; break; }
  sleep 60
done
[ $ok = 1 ] || { avisar "$new" "no pude armar la cola de lo que quedó" 4; exit 1; }
[ -s "$L/${new}_generada.txt" ] || { avisar "$new" "no quedaba nada para lanzar" 3; exit 0; }
$SSH "bash -lc 'hostname; bash ~/rotten-rl/scripts/lab/start_queue.sh $new; sleep 30; tail -1 ~/scratch/logs/$new.log'"
touch "$L/watch_$new.out"
until=$(env -u PYTHONPATH python3 -m rotten_rl.lab_hours stop)
sudo -n systemctl stop podrida-watch podrida-backup 2>/dev/null
sudo -n systemctl reset-failed podrida-watch podrida-backup 2>/dev/null
sudo -n systemd-run --unit=podrida-watch --uid="$(id -u)" --gid="$(id -g)" --working-directory="$R" $E \
  -E WATCH_HOST="$HOST" bash -c 'n=0; while true; do out=$(scripts/lab/watch_queue.sh '"$new"'.log $n); echo "$(date "+%d %H:%M") $out" >> '"$L/watch_$new.out"'; case "$out" in TERMINO*) n=$(echo "$out" | awk "{print \$2}"); avisar "'"$new"' (-14)" "$out" 3;; *) avisar "'"$new"' (-14)" "$out" 4; break;; esac; done'
sleep 60
sudo -n systemd-run --unit=podrida-backup --uid="$(id -u)" --gid="$(id -g)" --working-directory="$R" $E \
  bash -c 'out=$(scripts/lab/backup_loop.sh "'"$until"'" 120); echo "$(date +%H:%M) $out" >> '"$L/backup_loop_$new.out"'; case "$out" in FALLA*) avisar "Respaldo lab" "$out" 4;; esac'
avisar "$new" "lancé lo que quedaba: $(cut -d' ' -f1 "$L/${new}_generada.txt" | tr '\n' ' ')" 3
echo LANZADO
