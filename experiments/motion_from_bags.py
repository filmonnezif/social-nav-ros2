"""Compute motion_metrics for every recorded run (inside the container, needs rosbag2).

For each results/<root>/<run>/bag it rebuilds the robot's map-frame trajectory
from /odom and the map->odom transform, takes the replayed people from the
scene's people file, and writes motion.npz (per-sample distributions) and
motion.json (per-episode scalars) next to the bag. Only the part from the first
goal to the end of the episode is used.

Usage: python3 experiments/motion_from_bags.py results/h3_grid
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from motion_metrics import compute  # noqa: E402
from render_run import PKG, read_bag, robot_in_map  # noqa: E402


def main():
    root = Path(sys.argv[1]).resolve()
    for run in sorted(p.parent.parent for p in root.glob('*/bag/metadata.yaml')):
        if (run / 'motion.json').exists() and '--force' not in sys.argv:
            continue
        scene = run.name.rsplit('_', 2)[0]
        odom, map_odom, *_ = read_bag(run / 'bag')
        robot = robot_in_map(odom, map_odom)
        ep = json.loads((run / 'episode.json').read_text())
        m = (robot[:, 0] >= ep['goals'][0]['start_s']) & (robot[:, 0] <= ep['end_s'])
        tracks = pd.read_csv(PKG / 'config' / f'{scene}_people.csv')
        people = {b: (g.t.values, g.x.values, g.y.values) for b, g in tracks.groupby('body')}
        out = compute(robot[m, 0], robot[m, 1], robot[m, 2], people)
        if out is None:
            continue
        np.savez(run / 'motion.npz', **out['samples'])
        (run / 'motion.json').write_text(json.dumps(out['scalars'], indent=2))
        print(run.name, json.dumps({k: round(v, 3) for k, v in out['scalars'].items()}))


if __name__ == '__main__':
    main()
