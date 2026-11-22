#!/usr/bin/env bash
# Build the workspace, run one recorded episode inside the experiment container,
# then render demo.mp4 (scene camera + top-down view) and metrics.json.
# Usage: experiments/run_episode.sh <run-name> [extra ros2 launch args...]
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
RUN="${1:-run_$(date +%Y%m%d_%H%M%S)}"; shift || true
mkdir -p "$REPO/results/$RUN"

podman run --rm --name social-nav-run \
  --device /dev/dri/card1 --device /dev/dri/renderD128 --group-add keep-groups \
  -v "$REPO:/ws:Z" -w /ws/ros2_ws \
  localhost/social-nav:jazzy bash -c "
    source /opt/ros/jazzy/setup.bash &&
    colcon build --symlink-install --event-handlers console_direct- > /dev/null &&
    source install/setup.bash &&
    ros2 launch social_nav_sim episode.launch.py out_dir:=/ws/results/$RUN $* ;
    python3 /ws/experiments/render_run.py /ws/results/$RUN
  " 2>&1 | tee "$REPO/results/$RUN/console.log"
