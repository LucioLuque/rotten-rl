#!/bin/bash
# Watches a night of runs from the laptop and exits (so Claude attends it) when something happens:
# another run of the queue finished, an expected file appeared (a tournament), it is backup time,
# or the lab cannot be trusted any more. It must never keep watching blind.
#
#   scripts/lab/watch_queue.sh <queue log> <finished runs already seen> [file to wait for]
#   e.g.  scripts/lab/watch_queue.sh queue7.log 3 torneo_3p.md     (paths relative to ~/scratch)
#
# Output (one line): TERMINO <n> | ARCHIVO | HORA | NADA_CORRE | FALLA <detail>.
# - Every query opens its own connection (ControlPath=none): it does not depend on the shared
#   connection, which breaks on the lab side ("unknown userid"; night of 29/10: 3 blind hours).
# - The machine is checked on every query. The first answer sets it, or WATCH_HOST if given. If it
#   changes, or the queue log is missing, it exits with FALLA right away. Stderr is kept apart from
#   the parsed output: on 31/10 a failed grep still printed a valid-looking "0" and the watcher saw
#   nothing for 30 minutes after the lab moved us to another machine.
# - No training process running: exits with NADA_CORRE.
# - If a query fails twice in a row (~5 minutes), closes the shared connection and exits with FALLA.
# - Monday to Friday: exits with HORA at 06:35 and does not query the lab until 22:00, because it
#   closes at 07:00. Saturdays and Sundays it keeps watching all day.
log=$1; known=$2; file=${3:-}
host=${WATCH_HOST:-}
fails=0
err=$(mktemp)
trap 'rm -f "$err"' EXIT
remote="hostname
if [ -f ~/scratch/logs/$log ]; then grep -c terminó ~/scratch/logs/$log; else echo NOLOG; fi
if [ -n '$file' ] && [ -f ~/scratch/$file ]; then echo 1; else echo 0; fi
ps -eo args | awk '\$2 ~ /train[.]py\$/ {n++} END {print n+0}'"
while true; do
  now=$((10#$(date +%H%M))); dow=$(date +%u)          # 1 = Monday ... 7 = Sunday
  if [ "$dow" -le 5 ] && [ $now -ge 635 ] && [ $now -lt 2200 ]; then
    echo "HORA $(date +%H:%M)"; exit 0
  fi
  out=$(timeout 60 ssh -o ControlPath=none -o ConnectTimeout=20 labs004 "$remote" 2>"$err")
  got=$(echo "$out" | sed -n 1p); n=$(echo "$out" | sed -n 2p)
  f=$(echo "$out" | sed -n 3p); running=$(echo "$out" | sed -n 4p)
  if ! [[ "$running" =~ ^[0-9]+$ ]]; then
    fails=$((fails + 1))
    if [ $fails -ge 2 ]; then
      ssh -O exit labs004 >/dev/null 2>&1
      echo "FALLA $(date +%H:%M) $(tail -1 "$err")"; exit 1
    fi
    sleep 150; continue
  fi
  fails=0
  [ -n "$host" ] || host=$got
  if [ "$got" != "$host" ]; then echo "FALLA $(date +%H:%M) máquina cambiada: $host -> $got"; exit 1; fi
  if [ "$n" = "NOLOG" ]; then echo "FALLA $(date +%H:%M) no existe $log en $got"; exit 1; fi
  if [ "$n" -gt "$known" ]; then echo "TERMINO $n"; exit 0; fi
  if [ "$f" = "1" ]; then echo "ARCHIVO"; exit 0; fi
  if [ "$running" -eq 0 ]; then echo "NADA_CORRE $(date +%H:%M) en $got"; exit 0; fi
  sleep 300
done
