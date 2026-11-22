#!/usr/bin/env bash
# Tune the yield governor on training-day scenes (never the held-out test day).
# Runs LSTM prediction and each governor profile on 3 scenes, left to right.
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
export OUT_ROOT=results/tuning RENDER=0
first=1
for scene in tune180522_20 tune180522_80 tune180522_140; do
  for v in lstm social-strict social-relaxed; do
    run="${scene}_LR_${v/-/}"
    [ -f "$REPO/results/tuning/$run/metrics.json" ] && continue
    mode=${v%%-*}; gov=${v#*-}; [ "$gov" = "$v" ] && gov=relaxed
    SKIP_BUILD=$([ $first = 1 ] && echo 0 || echo 1) "$REPO/experiments/run_episode.sh" "$run" \
      scene:=$scene mode:=$mode governor:=$gov x:=-7.0 y:=-1.0 yaw:=0.0 goals:=7.5,1.0,0.0
    first=0
    echo "$(date +%H:%M:%S) $run $(tr -d '\n ' < "$REPO/results/tuning/$run/metrics.json" 2>/dev/null)" >> "$REPO/results/tuning/progress.log"
  done
done
