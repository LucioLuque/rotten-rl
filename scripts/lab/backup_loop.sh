#!/bin/bash
# Backs up the lab to the laptop during a night or a weekend, with pull_from_lab.sh, so that a
# restart or a change of machine loses at most one interval of work. Runs on the laptop; nothing
# runs on the lab.
#
#   scripts/lab/backup_loop.sh <until "YYYY-MM-DD HH:MM"> [minutes between backups, 60 by default]
#
# Output (one line, it only exits for these): FIN | FALLA <detail>.
# - No backups from 10 minutes before the lab closes until it reopens (rotten_rl/lab_hours.py: room
#   calendar with its margins, weekday reboot at 07:30). One last backup 15 minutes before it closes.
# - If a backup fails twice in a row it exits with FALLA.
cd "$(dirname "$0")"
until_epoch=$(date -d "$1" +%s) || exit 2
every=$((${2:-60} * 60))
last=0; last_slot=""; fails=0
while [ "$(date +%s)" -lt "$until_epoch" ]; do
  epoch=$(date +%s)
  closed=0; slot=0; due=0
  if ! closes=$(cd ../.. && env -u PYTHONPATH python3 -m rotten_rl.lab_hours closes 2>&1); then
    echo "FALLA $(date +%H:%M) calendario: $(echo "$closes" | tail -1)"; exit 1
  fi
  if [ "$closes" = "-" ]; then
    closed=1
  else
    left=$(($(date -d "$closes" +%s) - epoch))
    if [ $left -lt 600 ]; then closed=1; elif [ $left -lt 900 ]; then slot=1; fi
  fi
  if [ $slot -eq 1 ] && [ "$last_slot" != "$closes" ]; then due=1; fi
  [ $((epoch - last)) -ge "$every" ] && due=1
  if [ $closed -eq 0 ] && [ $due -eq 1 ]; then
    if output=$(./pull_from_lab.sh 2>&1); then
      echo "$output" | tail -1 >> ../../runs/lab/backup.log
      last=$(date +%s); fails=0
      if [ $slot -eq 1 ]; then last_slot=$closes; fi
    else
      fails=$((fails + 1))
      if [ $fails -ge 2 ]; then echo "FALLA $(date +%H:%M) $(echo "$output" | tail -1)"; exit 1; fi
    fi
  fi
  sleep 60
done
echo FIN
