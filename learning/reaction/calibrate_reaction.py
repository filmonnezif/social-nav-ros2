"""Calibrate a reactive-crowd model to how close people let the robot come in THÖR-MAGNI.

Evidence (see `occupancy`): compared with a placebo robot (the real robot's path
shifted in time, so it is unrelated to where people are), people are found far
less often near the real robot: almost never within 0.75 m of its centre, about
0.2x as often at 0.75-1.0 m and 0.6x at 1.0-1.25 m, and normally beyond 1.5 m.
Acceleration-based fits miss this because passing around the robot curves people
towards it, which hides the push away.

Reaction model used in simulation: each replayed person keeps their recorded path
plus an offset o driven by a repulsion from the robot and a damped spring back to
the path:  o'' = A exp(-(d - R)/B) n - k o - c o',  with d the person-robot
distance, n the unit vector robot -> person, R = 0.7 m (DARKO half-length + a
person's radius), k = 2 s^-2, c = 2.8 s^-1 (critically damped). A and B are chosen
so that replaying the training days against placebo robots reproduces the real
occupancy ratios. Caveat: in the data the human driver also steered away from
people, so attributing all of the avoidance to the people is an upper bound.

Usage: python learning/reaction/calibrate_reaction.py <tracks_dir>  -> results/reaction/reaction_calibration.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import fit_reaction as F  # noqa: E402

BINS = np.array([0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0])
R, K, C, DT = 0.7, 2.0, 2.8, 0.1
SHIFTS = (-90, -60, -30, 30, 60, 90)
OUT = Path(__file__).resolve().parents[2] / 'results/reaction'


def grids(s):
    """Per run: time grid, people positions [P x T x 2] (NaN when untracked), robot [T x 2]."""
    out = []
    for run, g in s.groupby('run'):
        t = np.round(np.sort(g.t.unique()), 2)
        idx = {v: i for i, v in enumerate(t)}
        bodies = sorted(g.body.unique())
        P = np.full((len(bodies), len(t), 2), np.nan)
        for j, b in enumerate(bodies):
            h = g[g.body == b]
            ii = [idx[v] for v in np.round(h.t.values, 2)]
            P[j, ii, 0], P[j, ii, 1] = h.x.values, h.y.values
        r = g.drop_duplicates('t').set_index('t').reindex(t)[['x_r', 'y_r']].values
        out.append((run, t, P, r))
    return out


def occupancy(dist_list):
    d = np.concatenate([x[np.isfinite(x)] for x in dist_list])
    h, _ = np.histogram(d, np.concatenate([[0], BINS]))
    return h / max(len(d), 1)


def simulate(P, robot, A, B):
    """Replay people with the reactive offset against a robot path; returns distances [P x T]."""
    o, v = np.zeros(P.shape[::2]), np.zeros(P.shape[::2])
    D = np.full(P.shape[:2], np.nan)
    for k in range(P.shape[1]):
        q = P[:, k] + o
        rel = q - robot[k]
        d = np.linalg.norm(rel, axis=1)
        ok = np.isfinite(d)
        f = np.zeros_like(o)
        if A > 0:
            f[ok] = (A * np.exp(-(d[ok] - R) / B) / np.maximum(d[ok], 1e-6))[:, None] * rel[ok]
        acc = f - K * o - C * v
        v = np.where(ok[:, None], v + acc * DT, 0)
        o = np.where(ok[:, None], o + v * DT, 0)
        D[:, k] = d
    return D


def main():
    tracks = Path(sys.argv[1])
    F.NEAR = 99
    s = pd.concat([d for d in (F.samples(f) for f in sorted(tracks.glob('*_SC3*.csv'))) if d is not None])
    G = grids(s)
    train = [g for g in G if not g[0].startswith(F.TEST_DAY)]

    real = occupancy([np.linalg.norm(P - r[None], axis=2).ravel() for _, _, P, r in G])

    def shifted(r, sh):
        n = int(round(sh / DT))
        return np.roll(r, n, axis=0)

    placebo = occupancy([np.linalg.norm(P - shifted(r, sh)[None], axis=2).ravel()
                         for _, _, P, r in G for sh in SHIFTS])
    target = real / np.maximum(placebo, 1e-12)
    print('real / placebo occupancy by distance:', dict(zip(BINS, np.round(target, 2))))

    best = None
    for A in (8.0, 16.0, 32.0, 64.0):
        for B in (0.5, 0.7, 1.0):
            sim = occupancy([simulate(P, shifted(r, sh), A, B).ravel() for _, _, P, r in train for sh in SHIFTS[:2]])
            ratio = sim / np.maximum(placebo, 1e-12)
            err = float(np.sum((ratio[1:5] - target[1:5]) ** 2))      # fit the 0.5-1.5 m bins
            print(f'A {A:4.1f} B {B:.1f}: ratios {np.round(ratio[:6], 2)}  err {err:.3f}')
            if best is None or err < best[0]:
                best = (err, A, B, ratio)
    err, A, B, ratio = best
    res = {'A': A, 'B': B, 'R': R, 'k': K, 'c': C, 'fit_error': err,
           'bins_m': BINS.tolist(), 'target_ratio': target.tolist(), 'simulated_ratio': ratio.tolist(),
           'note': 'upper bound: the human driver also steered away from people'}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'reaction_calibration.json').write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k in ('A', 'B', 'fit_error')}))


if __name__ == '__main__':
    main()
