"""Behaviour-cloning data: what the human driver of the THÖR-MAGNI robot did next.

For every 0.4 s of every Scenario 3 recording where the teleoperated robot is
tracked, the input describes what the driver could see over the last 1.6 s
(4 steps at 2.5 Hz), in a frame centred on the robot and pointing towards where
it went (its position 4 s later, a "hindsight subgoal"; if it moved less than
0.3 m, the frame keeps the previous direction):
  own velocity (2), subgoal position (2), and the 6 nearest people's position,
  velocity and presence (6 x 5)                      -> 34 features x 4 steps
The target is the driver's mean velocity over the next 0.8 s in that frame
(forward, sideways). Speed and yielding are what we transfer to our robot.

Split by day as for the predictor: test = 17 May, val = 18 May SC3B_R2.
Usage: python learning/data/make_bc_dataset.py <tracks_dir> <out_dir>
       (writes bc_meta.mat + bc_partNN.mat, each small enough to upload)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import savemat

DT, HIST, K = 0.4, 4, 6
HORIZON_GOAL, HORIZON_ACT = 4.0, 0.8
ROBOT, SKIP = 'DARKO_Robot', ('LO1',)
TEST_DAY, VAL_RUNS = '170522', ('180522_SC3B_R2',)


def on_grid(g, grid):
    t = g.t.values
    i = np.clip(np.searchsorted(t, grid), 1, len(t) - 1)
    ok = (grid >= t[0]) & (grid <= t[-1]) & (t[i] - t[i - 1] <= 0.15)
    p = np.full((len(grid), 2), np.nan)
    p[ok, 0], p[ok, 1] = np.interp(grid[ok], t, g.x.values), np.interp(grid[ok], t, g.y.values)
    return p


def rot(v, c, s):
    return np.stack([c * v[..., 0] + s * v[..., 1], -s * v[..., 0] + c * v[..., 1]], -1)


def run_samples(df):
    grid = np.arange(df.t.min(), df.t.max(), DT)
    r = on_grid(df[df.body == ROBOT].sort_values('t'), grid)
    people = [on_grid(g.sort_values('t'), grid) for b, g in df.groupby('body') if b not in SKIP + (ROBOT,)]
    P = np.stack(people)                                    # [N x T x 2]
    V = np.full_like(P, np.nan)
    V[:, 1:] = (P[:, 1:] - P[:, :-1]) / DT
    rv = np.full_like(r, np.nan)
    rv[1:] = (r[1:] - r[:-1]) / DT
    ng, na = int(HORIZON_GOAL / DT), int(HORIZON_ACT / DT)
    X, Y, near = [], [], []
    last_dir = None
    for k in range(HIST, len(grid) - ng):
        hist = slice(k - HIST + 1, k + 1)
        if np.isnan(rv[hist]).any() or np.isnan(r[k + ng]).any() or np.isnan(r[k:k + na + 1]).any():
            last_dir = None
            continue
        disp = r[k + ng] - r[k]
        if np.hypot(*disp) > 0.3:
            last_dir = np.arctan2(disp[1], disp[0])
        elif last_dir is None:
            continue
        c, s = np.cos(last_dir), np.sin(last_dir)
        o = r[k]
        own = rot(rv[hist], c, s)
        goal = np.repeat(rot(disp, c, s)[None], HIST, 0)
        d_now = np.linalg.norm(P[:, k] - o, axis=1)
        order = np.argsort(np.where(np.isnan(d_now), np.inf, d_now))[:K]
        ppl = np.zeros((HIST, K, 5))
        for j, i in enumerate(order):
            if np.isnan(d_now[i]):
                break
            m = ~np.isnan(V[i, hist, 0])
            ppl[m, j, :2] = rot(P[i, hist][m] - o, c, s)
            ppl[m, j, 2:4] = rot(V[i, hist][m], c, s)
            ppl[m, j, 4] = 1
        # people features ordered field-major like the predictor: [x_1..x_K, y_1..y_K, ...]
        feats = np.concatenate([own, goal, np.transpose(ppl, (0, 2, 1)).reshape(HIST, -1)], 1)
        target = rot((r[k + na] - r[k]) / HORIZON_ACT, c, s)
        X.append(feats)
        Y.append(target)
        near.append(bool(np.nanmin(np.where(np.isnan(d_now), np.inf, d_now)) < 2.0))
    return np.array(X, np.float32), np.array(Y, np.float32), np.array(near)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('tracks_dir', type=Path)
    ap.add_argument('out_dir', type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    splits = {'train': [], 'val': [], 'test': []}
    for f in sorted(args.tracks_dir.glob('*_SC3*.csv')):
        split = 'test' if f.stem.startswith(TEST_DAY) else 'val' if f.stem in VAL_RUNS else 'train'
        X, Y, near = run_samples(pd.read_csv(f))
        print(f'{f.stem:16s} {split:5s} {len(X):5d} samples, {int(near.sum()):5d} with a person within 2 m, '
              f'median speed {np.median(np.hypot(*Y.T)) if len(Y) else 0:.2f} m/s')
        splits[split].append((X, Y, near))
    meta = {'info': 'Behaviour cloning of the THOR-MAGNI robot driver. x: 4 steps x 34 features '
                    '[own vx vy, subgoal x y, people x(6) y(6) vx(6) vy(6) present(6)], robot frame towards '
                    'the 4 s hindsight subgoal; y: mean velocity over the next 0.8 s (forward, sideways).'}
    part = 0
    for split, items in splits.items():
        X = np.concatenate([i[0] for i in items])
        Y = np.concatenate([i[1] for i in items])
        near = np.concatenate([i[2] for i in items]).astype(np.float32)[:, None]
        print(f'{split}: {len(X)} samples, {int(near.sum())} near people')
        for a in range(0, len(X), 12000):
            part += 1
            savemat(args.out_dir / f'bc_part{part:02d}.mat', {f'{split}_X': X[a:a + 12000], f'{split}_Y': Y[a:a + 12000],
                                                              f'{split}_near': near[a:a + 12000]}, do_compression=True)
    savemat(args.out_dir / 'bc_meta.mat', meta)
    for f in sorted(args.out_dir.glob('bc_*.mat')):
        print(f.name, f'{f.stat().st_size / 1e6:.1f} MB')


if __name__ == '__main__':
    main()
