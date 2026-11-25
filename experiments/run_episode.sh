#!/usr/bin/env bash
# Build the workspace, run one recorded episode inside the experiment container,
# then compute metrics.json and (unless RENDER=0) render demo.mp4.
# Usage: experiments/run_episode.sh <run-name> [ros2 launch args, e.g. scene:=thor_sc3 mode:=lstm]
# Results go to results/<run-name>/ (set OUT_ROOT to change the parent folder).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
RUN="${1:-run_$(date +%Y%m%d_%H%M%S)}"; shift || true
OUT_ROOT="${OUT_ROOT:-results}"
mkdir -p "$REPO/$OUT_ROOT/$RUN"
SCENE=thor_sc3
for a in "$@"; do [[ $a == scene:=* ]] && SCENE="${a#scene:=}"; done
VIDEO_FLAG=""; [[ "${RENDER:-1}" == 0 ]] && VIDEO_FLAG="--no-video"
BUILD="colcon build --symlink-install --event-handlers console_direct- > /dev/null &&"
[[ "${SKIP_BUILD:-0}" == 1 ]] && BUILD=""

podman run --rm --name "social-nav-run-$$" \
  --device /dev/dri/card1 --device /dev/dri/renderD128 --group-add keep-groups \
  -v "$REPO:/ws:Z" -w /ws/ros2_ws \
  localhost/social-nav:jazzy bash -c "
    source /opt/ros/jazzy/setup.bash && $BUILD
    source install/setup.bash &&
    ros2 launch social_nav_sim episode.launch.py out_dir:=/ws/$OUT_ROOT/$RUN $* ;
    python3 /ws/experiments/render_run.py /ws/$OUT_ROOT/$RUN $SCENE $VIDEO_FLAG
  " > "$REPO/$OUT_ROOT/$RUN/console.log" 2>&1
