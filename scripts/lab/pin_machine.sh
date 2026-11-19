#!/bin/bash
# scripts/lab/pin_machine.sh  —  Keeps a master ssh connection on one temporary lab machine.
#
#   scripts/lab/pin_machine.sh <machine, e.g. labo-s004-18> <until "YYYY-MM-DD HH:MM"> [ssh host]
#
# When our machine does not answer, the lab gives a different temporary one on each connection.
# This opens a master connection (ControlPath ~/.ssh/cm/ten, the one the `lab21` alias reuses),
# keeps it only if it landed on <machine> and, if it drops, tries again. Run it as a systemd unit;
# watchers and backups then go with LAB_SSH=lab21 LAB_PINNED=1. Stops at <until> (before the lab
# closes: never connect while it is closed).
target=$1; until_epoch=$(date -d "$2" +%s) || exit 2; LAB=${3:-labs004b}
P=$HOME/.ssh/cm/ten
while [ "$(date +%s)" -lt "$until_epoch" ]; do
  ssh -S "$P" -O exit "$LAB" >/dev/null 2>&1
  ssh -o ControlMaster=yes -o ControlPath="$P" -o ControlPersist=no -o ServerAliveInterval=30 \
      -o ServerAliveCountMax=4 -N "$LAB" 2>/dev/null &
  pid=$!
  sleep 15
  h=$(timeout 30 ssh -o ControlMaster=no -o ControlPath="$P" "$LAB" hostname 2>/dev/null | tail -1)
  echo "$(date '+%d %H:%M') $h"
  if [ "$h" = "$target" ]; then
    left=$((until_epoch - $(date +%s)))
    timeout "$left" tail --pid="$pid" -f /dev/null      # holds while the master lives
    kill "$pid" 2>/dev/null
  else
    kill "$pid" 2>/dev/null
    sleep 30
  fi
done
ssh -S "$P" -O exit "$LAB" >/dev/null 2>&1
