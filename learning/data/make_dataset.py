"""Cut cleaned THÖR-MAGNI tracks into trajectory-prediction snippets for MATLAB.

Each snippet is one person over 8 s at 2.5 Hz: 8 observed steps (3.2 s) and
12 future steps (4.8 s), the usual ETH/UCY set-up. Only fully recorded spans are
used (no interpolated or bridged samples). Everything is expressed in the
person's own frame at the last observed step: origin at their position, x axis
along their heading.

Per snippet:
  ego  [8 x 4]      own position and velocity (x, y, vx, vy)
  nbr  [8 x K x 5]  K nearest other people: x, y, vx, vy, present
  rob  [8 x 5]      the (teleoperated) robot: x, y, vx, vy, present
  fut  [12 x 2]     future positions (the target)
  near              robot within 3 m at the last observed step

Split by recording day so the test set is never seen in training:
test = 17 May (all four runs, includes the demo scene), val = 18 May run
SC3B_R2, train = everything else.

Usage: python learning/data/make_dataset.py <tracks_dir> <out.mat>
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import savemat

DT = 0.4
OBS, FUT = 8, 12
K = 4
VMAX = 2.5
ROBOT = 'DARKO_Robot'
SKIP = ('LO1',)
TEST_DAY = '170522'
VAL_RUNS = ('180522_SC3B_R2',)


def grid_positions(g, grid):
    """Positions of one body on the grid; NaN where it was not recorded on both sides."""
    t, x, y = g.t.values, g.x.values, g.y.values
    i = np.clip(np.searchsorted(t, grid), 1, len(t) - 1)
    ok = (grid >= t[0]) & (grid <= t[-1]) & (t[i] - t[i - 1] <= 0.15)
    p = np.full((len(grid), 2), np.nan)
    p[ok, 0] = np.interp(grid[ok], t, x)
    p[ok, 1] = np.interp(grid[ok], t, y)
    return p


def to_local(xy, origin, c, s):
    d = xy - origin
    return np.stack([c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]], -1)


def rot(v, c, s):
    return np.stack([c * v[..., 0] + s * v[..., 1], -s * v[..., 0] + c * v[..., 1]], -1)


def run_snippets(df):
    people = sorted(b for b in df.body.unique() if b not in SKIP + (ROBOT,))
    grid = np.arange(df.t.min(), df.t.max(), DT)
    P = np.stack([grid_positions(df[df.body == b].sort_values('t'), grid) for b in people])  # [B,T,2]
    V = np.full_like(P, np.nan)
    V[:, 1:] = (P[:, 1:] - P[:, :-1]) / DT
    r = df[df.body == ROBOT].sort_values('t')
    R = grid_positions(r, grid) if len(r) else np.full((len(grid), 2), np.nan)
    RV = np.full_like(R, np.nan)
    RV[1:] = (R[1:] - R[:-1]) / DT

    out = {k: [] for k in ('ego', 'nbr', 'rob', 'fut', 'near')}
    L = OBS + FUT
    valid = ~np.isnan(V[..., 0])
    for b in range(len(people)):
        for s in range(1, len(grid) - L + 1):
            if not valid[b, s:s + L].all():
                continue
            pos, vel = P[b, s:s + L], V[b, s:s + L]
            if np.hypot(*vel.T).max() > VMAX:
                continue
            o = pos[OBS - 1]
            h = pos[OBS - 1] - pos[OBS - 3]
            yaw = np.arctan2(h[1], h[0]) if np.hypot(*h) > 0.05 else 0.0
            c, sn = np.cos(yaw), np.sin(yaw)
            ego = np.concatenate([to_local(pos[:OBS], o, c, sn), rot(vel[:OBS], c, sn)], -1)
            fut = to_local(pos[OBS:], o, c, sn)

            others = [j for j in range(len(people)) if j != b and valid[j, s + OBS - 1]]
            others.sort(key=lambda j: np.hypot(*(P[j, s + OBS - 1] - o)))
            nbr = np.zeros((OBS, K, 5))
            for k, j in enumerate(others[:K]):
                m = valid[j, s:s + OBS]
                nbr[m, k, :2] = to_local(P[j, s:s + OBS][m], o, c, sn)
                nbr[m, k, 2:4] = rot(V[j, s:s + OBS][m], c, sn)
                nbr[m, k, 4] = 1
            rob = np.zeros((OBS, 5))
            m = ~np.isnan(RV[s:s + OBS, 0])
            rob[m, :2] = to_local(R[s:s + OBS][m], o, c, sn)
            rob[m, 2:4] = rot(RV[s:s + OBS][m], c, sn)
            rob[m, 4] = 1
            near = bool(rob[-1, 4] and np.hypot(*rob[-1, :2]) < 3.0)
            for k, v in zip(('ego', 'nbr', 'rob', 'fut', 'near'), (ego, nbr, rob, fut, near)):
                out[k].append(v)
    return {k: np.asarray(v, np.float32) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('tracks_dir', type=Path)
    ap.add_argument('out', type=Path)
    args = ap.parse_args()
    splits = {'train': [], 'val': [], 'test': []}
    for f in sorted(args.tracks_dir.glob('*_SC3*.csv')):
        run = f.stem
        split = 'test' if run.startswith(TEST_DAY) else 'val' if run in VAL_RUNS else 'train'
        snip = run_snippets(pd.read_csv(f))
        splits[split].append((run, snip))
        print(f'{run:16s} {split:5s} {len(snip["fut"]):6d} snippets, {int(snip["near"].sum()):5d} near robot')
    mat = {}
    for split, items in splits.items():
        for k in ('ego', 'nbr', 'rob', 'fut', 'near'):
            mat[f'{split}_{k}'] = np.concatenate([s[k] for _, s in items])
        mat[f'{split}_runs'] = np.array([r for r, _ in items], dtype=object)
        mat[f'{split}_run_counts'] = np.array([len(s['fut']) for _, s in items], np.float64)
        print(f'{split}: {len(mat[split + "_fut"])} snippets, {int(mat[split + "_near"].sum())} near robot')
    mat['info'] = ('THOR-MAGNI Scenario 3 snippets: 8 obs + 12 future steps at 2.5 Hz, person-centric frame. '
                   'ego=[x y vx vy], nbr=[x y vx vy present] for 4 nearest people, rob=[x y vx vy present], fut=[x y].')
    savemat(args.out, mat, do_compression=True)
    print('wrote', args.out, f'{args.out.stat().st_size / 1e6:.1f} MB')


if __name__ == '__main__':
    main()
