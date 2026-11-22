"""Learn how people in THÖR-MAGNI react to the (human-driven) robot.

Model: an anisotropic social-force repulsion from the robot acting on each person
  F = A * exp(-(d_eff - R) / B) * w(phi) * n
where n points from the robot to the person, d_eff is their closest distance
over the next `tau` seconds if both keep their velocity (so people react to a
robot coming at them, not only to where it is), R = 0.6 m (sum of radii),
phi is the angle between the person's heading and the direction to the robot,
and w(phi) = lam + (1 - lam) * (1 + cos phi) / 2 weights robots in front more
than behind. A, B, lam (and tau, by grid search) are fitted by least squares to
the people's measured accelerations while the robot is within 4 m; elsewhere
people's mean acceleration is close to zero, so a consistent push away from the
robot is the reaction we are after.

Fitted on the training days (12, 13, 18 May), validated on the held-out day
(17 May): does adding the fitted reaction to a constant-velocity extrapolation
predict where people are 1-2 s later better than no reaction?

Usage: python learning/reaction/fit_reaction.py <tracks_dir>   (writes results/reaction/)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.signal import savgol_filter

DT, WIN, R, NEAR = 0.1, 11, 0.6, 4.0
OUT = Path(__file__).resolve().parents[2] / 'results/reaction'
TEST_DAY = '170522'


def smooth_segments(g):
    """Split one body's track at gaps and return smoothed p, v, a on a 10 Hz grid."""
    t = g.t.values
    out = []
    brk = np.where(np.diff(t) > 0.15)[0]
    for a, b in zip([0, *(brk + 1)], [*(brk + 1), len(t)]):
        if b - a < WIN + 4:
            continue
        tt = np.arange(t[a], t[b - 1], DT)
        x, y = np.interp(tt, t[a:b], g.x.values[a:b]), np.interp(tt, t[a:b], g.y.values[a:b])
        f = lambda s, d: savgol_filter(s, WIN, 3, deriv=d, delta=DT)  # noqa: E731
        out.append(pd.DataFrame({'t': np.round(tt, 2), 'x': f(x, 0), 'y': f(y, 0), 'vx': f(x, 1), 'vy': f(y, 1),
                                 'ax': f(x, 2), 'ay': f(y, 2)}))
    return pd.concat(out) if out else None


def samples(path):
    d = pd.read_csv(path)
    r = d[d.body == 'DARKO_Robot'].sort_values('t')
    rk = smooth_segments(r)
    if rk is None:
        return None
    rk = rk.set_index('t')
    rows = []
    for b, g in d[~d.body.isin(['DARKO_Robot', 'LO1'])].groupby('body'):
        pk = smooth_segments(g.sort_values('t'))
        if pk is None:
            continue
        m = pk.join(rk, on='t', rsuffix='_r', how='inner')
        rows.append(m.assign(body=b))
    s = pd.concat(rows)
    s['dist'] = np.hypot(s.x - s.x_r, s.y - s.y_r)
    s['speed'] = np.hypot(s.vx, s.vy)
    return s[(s.dist < NEAR) & (s.speed > 0.2)].assign(run=path.stem)


def force(params, s, tau):
    A, B, lam = params
    rel = np.stack([s.x - s.x_r, s.y - s.y_r], 1)                 # robot -> person
    vrel = np.stack([s.vx - s.vx_r, s.vy - s.vy_r], 1)
    tc = np.clip(-(rel * vrel).sum(1) / np.maximum((vrel ** 2).sum(1), 1e-6), 0, tau)
    d_eff = np.linalg.norm(rel + vrel * tc[:, None], axis=1)
    d = np.linalg.norm(rel, axis=1)
    n = rel / np.maximum(d, 1e-6)[:, None]
    heading = np.stack([s.vx, s.vy], 1) / np.maximum(s.speed.values, 1e-6)[:, None]
    cos_phi = -(n * heading).sum(1)                                  # robot ahead of the person -> +1
    w = lam + (1 - lam) * (1 + cos_phi) / 2
    return (A * np.exp(-(d_eff - R) / B) * w)[:, None] * n


def fit(s, tau):
    acc = np.stack([s.ax, s.ay], 1)
    res = least_squares(lambda p: (force(p, s, tau) - acc).ravel(), x0=[0.5, 0.5, 0.5],
                        bounds=([0, 0.05, 0], [10, 3, 1]))
    return res.x, float(np.mean((force(res.x, s, tau) - acc) ** 2)), float(np.mean(acc ** 2))


def rollout_error(s, params, tau, horizon):
    """Where is the person `horizon` s later: CV vs CV + fitted reaction (robot on its real path)?"""
    key = s.set_index(['run', 'body', 't'])
    errs_cv, errs_rx = [], []
    for (run, body), g in s.groupby(['run', 'body']):
        g = g.set_index('t')
        for t0 in g.index[::5]:
            t1 = round(t0 + horizon, 2)
            if t1 not in g.index:
                continue
            steps = int(round(horizon / DT))
            ts = [round(t0 + k * DT, 2) for k in range(steps)]
            if not all(t in g.index for t in ts):
                continue
            p = g.loc[t0, ['x', 'y']].values.astype(float)
            v = g.loc[t0, ['vx', 'vy']].values.astype(float)
            cv = p + v * horizon
            q, u = p.copy(), v.copy()
            for t in ts:                                             # integrate the reaction to the real robot
                row = g.loc[t].copy()
                row['x'], row['y'], row['vx'], row['vy'] = q[0], q[1], u[0], u[1]
                row['speed'] = max(np.hypot(*u), 1e-6)
                f = force(params, pd.DataFrame([row]), tau)[0]
                u = u + f * DT
                q = q + u * DT
            true = g.loc[t1, ['x', 'y']].values.astype(float)
            errs_cv.append(np.linalg.norm(cv - true))
            errs_rx.append(np.linalg.norm(q - true))
    return np.array(errs_cv), np.array(errs_rx)


def main():
    tracks = Path(sys.argv[1])
    OUT.mkdir(parents=True, exist_ok=True)
    data = [samples(f) for f in sorted(tracks.glob('*_SC3*.csv'))]
    s = pd.concat([d for d in data if d is not None])
    train, test = s[~s.run.str.startswith(TEST_DAY)], s[s.run.str.startswith(TEST_DAY)]
    print(f'samples within {NEAR} m of the robot: train {len(train)}, test {len(test)}')

    best = None
    for tau in (0.0, 0.5, 1.0, 1.5, 2.0):
        p, mse, var = fit(train, tau)
        print(f'tau {tau:.1f}: A {p[0]:.3f} B {p[1]:.3f} lam {p[2]:.2f}  explained {100 * (1 - mse / var):.2f}%')
        if best is None or mse < best[1]:
            best = (tau, mse, p, var)
    tau, _, p, var = best
    acc = np.stack([test.ax, test.ay], 1)
    test_mse = float(np.mean((force(p, test, tau) - acc) ** 2))
    result = {'A': p[0], 'B': p[1], 'lambda': p[2], 'tau': tau, 'R': R,
              'train_explained_var': 1 - best[1] / var,
              'test_explained_var': 1 - test_mse / float(np.mean(acc ** 2))}
    for h in (1.0, 2.0):
        cv, rx = rollout_error(test, p, tau, h)
        near = None
        result[f'test_{h:.0f}s_err_cv'] = float(cv.mean())
        result[f'test_{h:.0f}s_err_reaction'] = float(rx.mean())
        result[f'test_{h:.0f}s_n'] = int(len(cv))
        rng = np.random.default_rng(0)
        diff = rx - cv
        boots = rng.choice(diff, (2000, len(diff))).mean(1)
        result[f'test_{h:.0f}s_diff_ci'] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
        del near
    (OUT / 'reaction_model.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
