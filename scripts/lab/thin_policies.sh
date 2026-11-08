#!/bin/bash
# Thins the saved policies (checkpoints/policy_ep<N>.pt, one every eval) of the runs under a folder:
# keeps the first one of every KEEP_EVERY episodes (2M by default: JAX saves them at episodes like
# 2000128, never exact multiples) and the newest of each run, and deletes the rest. Training never
# reads these files (the "past" opponent lives in state/); only tournaments and play.py do, and a
# milestone every 2M plus the newest is enough for them. The lab keeps them all.
#
#   scripts/lab/thin_policies.sh <folder with runs> [--dry-run]
#   scripts/lab/thin_policies.sh --filter < listing     # e.g. the lab's, from pull_from_lab.sh
#
# --filter reads paths "<run>/checkpoints/policy_ep<N>.pt" (one per line) and prints, as "/<path>",
# the ones that thinning drops, for rsync --exclude-from.
set -euo pipefail
every=${KEEP_EVERY:-2000000}

drop() {
  sed -n 's|^\(.*\)/checkpoints/policy_ep\([0-9][0-9]*\)[.]pt$|\1 \2 &|p' | sort -k1,1 -k2,2n \
    | awk -v every="$every" '
        $1 != run { flush(); run = $1; n = 0 }
        { n++; ep[n] = $2 + 0; path[n] = $3 }
        END { flush() }
        function flush(   i) {
          for (i = 2; i < n; i++) if (int(ep[i] / every) == int(ep[i - 1] / every)) print path[i]
        }'
}

if [ "${1:-}" = "--filter" ]; then
  drop | sed 's|^|/|'
  exit 0
fi
dir=${1:?folder with runs}
cd "$dir"
# find, not a glob: ~80k files overflow the argument list and the glob silently listed nothing.
find . -mindepth 3 -maxdepth 3 -path './*/checkpoints/policy_ep*.pt' | sed 's|^[.]/||' | drop | while read -r f; do
  if [ "${2:-}" = "--dry-run" ]; then echo "would delete $f"; else rm -f "$f"; fi
done
