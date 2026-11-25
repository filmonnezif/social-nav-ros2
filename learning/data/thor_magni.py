"""Load a THÖR-MAGNI scenario CSV and reduce it to 2D tracks per tracked body.

Output columns: t [s], body, role, x [m], y [m], src. Positions are helmet (or
robot) centroids in the motion-capture frame, resampled to a fixed rate.

The motion-capture software often fails to solve a helmet's rigid body (no
centroid) while its individual markers are still visible. In those frames the
position is the mean of the visible markers (src=markers); on frames where both
exist this lands within ~8 cm of the centroid with one marker and ~2-4 cm with
three. src=centroid otherwise.

Both sources occasionally snap onto another helmet's markers for a frame or
two (jumps of several metres). Samples more than 0.3 m from a 0.7 s rolling
median of their own track are dropped, then a speed gate drops samples that
would need more than 2.5 m/s from the last kept one (longer mis-solves); after
1.5 s of rejections the gate re-locks onto the track.
"""
import argparse
import csv
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

N_META_ROWS = 16


def read_meta(path):
    meta = {}
    with open(path, newline='') as f:
        for _, row in zip(range(N_META_ROWS), csv.reader(f)):
            meta[row[0]] = [c for c in row[1:] if c != '']
    return meta


warnings.filterwarnings('ignore', message='Mean of empty slice')


def despike(part, window=7, max_dev=0.3):
    if len(part) < window:
        return part
    mx = part.x.rolling(window, center=True, min_periods=3).median()
    my = part.y.rolling(window, center=True, min_periods=3).median()
    return part[np.hypot(part.x - mx, part.y - my) <= max_dev]


def speed_gate(part, vmax=2.5, relock_s=1.5):
    t, x, y = part.t.values, part.x.values, part.y.values
    keep = np.zeros(len(t), bool)
    last = 0
    keep[0] = True
    for i in range(1, len(t)):
        dt = t[i] - t[last]
        if np.hypot(x[i] - x[last], y[i] - y[last]) <= vmax * dt + 0.05 or dt > relock_s:
            keep[i] = True
            last = i
    return part[keep]


def load_tracks(path, rate_hz=10.0):
    meta = read_meta(path)
    bodies, roles = meta['BODY_NAMES'], meta['BODY_ROLES']
    header = pd.read_csv(path, skiprows=N_META_ROWS, nrows=0).columns
    markers = {b: [c[:-2] for c in header if c.startswith(f'{b} - ') and c.endswith(' X')] for b in bodies}
    cols = set(['Time'] + [f'{b} Centroid_{a}' for b in bodies for a in 'XY']
               + [f'{m} {a}' for ms in markers.values() for m in ms for a in 'XY'])
    df = pd.read_csv(path, skiprows=N_META_ROWS, usecols=lambda c: c in cols)
    step = int(round(100 / rate_hz))  # mocap runs at 100 Hz
    df = df.iloc[::step]
    out = []
    for body, role in zip(bodies, roles):
        x, y = df.get(f'{body} Centroid_X'), df.get(f'{body} Centroid_Y')
        if x is None:
            continue
        x, y = x.values.copy(), y.values.copy()
        src = np.where(np.isfinite(x), 'centroid', '')
        ms = markers[body]
        if ms:
            mx = np.nanmean(df[[f'{m} X' for m in ms]].values, axis=1)
            my = np.nanmean(df[[f'{m} Y' for m in ms]].values, axis=1)
            use = ~np.isfinite(x) & np.isfinite(mx)
            x[use], y[use] = mx[use], my[use]
            src[use] = 'markers'
        part = pd.DataFrame({'t': df['Time'].values, 'body': body, 'role': role.strip(),
                             'x': x / 1000.0, 'y': y / 1000.0, 'src': src})
        out.append(speed_gate(despike(part.dropna())))
    return pd.concat(out, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv', type=Path)
    ap.add_argument('out', type=Path)
    ap.add_argument('--rate', type=float, default=10.0)
    args = ap.parse_args()
    tracks = load_tracks(args.csv, args.rate)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tracks.to_csv(args.out, index=False, float_format='%.3f')
    print(tracks.groupby(['body', 'role']).agg(n=('t', 'size'), t0=('t', 'min'), t1=('t', 'max'),
                                               xmin=('x', 'min'), xmax=('x', 'max'),
                                               ymin=('y', 'min'), ymax=('y', 'max')).round(2))


if __name__ == '__main__':
    main()
