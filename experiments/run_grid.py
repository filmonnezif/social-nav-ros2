"""H3 experiment grid: every scene x direction x set-up, recorded and scored.

Scenes are replay windows from the held-out test day (17 May, never seen by the
predictor). For each scene and direction the three set-ups run back to back:
  none  plain Nav2
  cv    Nav2 + prediction layer, constant-velocity forecasts
  lstm  Nav2 + prediction layer, MATLAB-trained LSTM forecasts
  social  LSTM forecasts + yielding speed governor + limits fitted to the human driver
          (run separately with --modes social)
Each run leaves a rosbag, the scene-camera MP4 and metrics.json in
results/h3_grid/<scene>_<dir>_<mode>/. Finished runs are skipped, so the grid
can be stopped and resumed.

Usage (host): .venv/bin/python experiments/run_grid.py --tracks <tracks_dir> --maps <maps_dir>
"""
import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNS = ['170522_SC3A_R1', '170522_SC3A_R2', '170522_SC3B_R1', '170522_SC3B_R2']
T0S = [10, 50, 90, 130, 170]
DURATION = 90.0
DIRECTIONS = {
    'LR': ('-7.0', '-1.0', '0.0', '7.5,1.0,0.0'),
    'RL': ('7.5', '1.0', '3.14159', '-7.0,-1.0,3.14159'),
}
MODES = ['none', 'cv', 'lstm']


def build_scenes(tracks, maps, people='actors'):
    scenes = []
    for run in RUNS:
        for t0 in T0S:
            name = f't{run.lower()}_{t0:03d}' + ('_b' if people == 'bodies' else '')
            if not (REPO / 'ros2_ws/src/social_nav_sim/worlds' / f'{name}.sdf').exists():
                r = subprocess.run([sys.executable, str(REPO / 'experiments/build_scene.py'),
                                    str(tracks / f'{run}.csv'), str(maps / '1705_SC3_map.png'),
                                    '--name', name, '--t0', str(t0), '--duration', str(DURATION),
                                    '--people', people],
                                   capture_output=True, text=True)
                if r.returncode:
                    print('skip', name, r.stderr.strip().splitlines()[-1:])
                    continue
            if max_speed(name) > 3.0:
                print('skip', name, '(tracking jump in the replay)')
                continue
            scenes.append(name)
    return scenes


def max_speed(name):
    """Fastest step of any replayed person; a jump means a tracking glitch the cleaning missed."""
    last, vmax = {}, 0.0
    with open(REPO / 'ros2_ws/src/social_nav_sim/config' / f'{name}_people.csv') as f:
        for r in csv.DictReader(f):
            t, x, y = float(r['t']), float(r['x']), float(r['y'])
            if r['body'] in last:
                t0, x0, y0 = last[r['body']]
                if t > t0:
                    vmax = max(vmax, math.hypot(x - x0, y - y0) / (t - t0))
            last[r['body']] = (t, x, y)
    return vmax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tracks', type=Path, required=True)
    ap.add_argument('--maps', type=Path, required=True)
    ap.add_argument('--limit', type=int, default=0, help='run at most this many episodes')
    ap.add_argument('--modes', default=','.join(MODES), help='comma-separated set-ups to run')
    ap.add_argument('--body', default='tb3', help='tb3 | darko')
    ap.add_argument('--reactive', action='store_true',
                    help='people react to the robot (data-calibrated model; uses simple-body scenes)')
    ap.add_argument('--out', default='results/h3_grid')
    ap.add_argument('--reaction', default='/ws/results/reaction/reaction_calibration.json',
                    help='reaction model used with --reactive (container path)')
    args = ap.parse_args()

    scenes = build_scenes(args.tracks, args.maps, 'bodies' if args.reactive else 'actors')
    out_root = REPO / args.out
    extra = [f'body:={args.body}']
    if args.reactive:
        extra.append(f'reaction:={args.reaction}')
    out_root.mkdir(parents=True, exist_ok=True)
    jobs = [(s, d, m) for s in scenes for d in DIRECTIONS for m in args.modes.split(',')]
    log = open(out_root / 'progress.log', 'a')
    print(f'{len(scenes)} scenes, {len(jobs)} runs', file=log, flush=True)

    built = False
    done = 0
    for scene, d, mode in jobs:
        run = f'{scene}_{d}_{mode}'
        if (out_root / run / 'metrics.json').exists():
            continue
        if args.limit and done >= args.limit:
            break
        x, y, yaw, goal = DIRECTIONS[d]
        env = dict(os.environ, OUT_ROOT=args.out, RENDER='0', SKIP_BUILD='1' if built else '0')
        t = time.time()
        subprocess.run([str(REPO / 'experiments/run_episode.sh'), run, f'scene:={scene}', f'mode:={mode}',
                        f'x:={x}', f'y:={y}', f'yaw:={yaw}', f'goals:={goal}', *extra], env=env)
        built = True
        done += 1
        m = out_root / run / 'metrics.json'
        summary = json.loads(m.read_text()) if m.exists() else {'error': 'no metrics'}
        print(f'{time.strftime("%H:%M:%S")} {run} {time.time() - t:.0f}s {json.dumps(summary)}',
              file=log, flush=True)


if __name__ == '__main__':
    main()
