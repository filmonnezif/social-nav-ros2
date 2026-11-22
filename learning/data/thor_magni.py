"""Load a THÖR-MAGNI scenario CSV and reduce it to 2D tracks per tracked body.

Output columns: t [s], body, role, x [m], y [m]. Positions are helmet (or robot)
centroids in the motion-capture frame, resampled to a fixed rate.
"""
import argparse
import csv
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


def load_tracks(path, rate_hz=10.0):
    meta = read_meta(path)
    bodies, roles = meta['BODY_NAMES'], meta['BODY_ROLES']
    cols = ['Time'] + [f'{b} Centroid_{a}' for b in bodies for a in 'XY']
    df = pd.read_csv(path, skiprows=N_META_ROWS, usecols=lambda c: c in cols)
    step = int(round(100 / rate_hz))  # mocap runs at 100 Hz
    df = df.iloc[::step]
    out = []
    for body, role in zip(bodies, roles):
        x, y = df.get(f'{body} Centroid_X'), df.get(f'{body} Centroid_Y')
        if x is None:
            continue
        part = pd.DataFrame({'t': df['Time'].values, 'body': body, 'role': role.strip(),
                             'x': x.values / 1000.0, 'y': y.values / 1000.0})
        out.append(part.dropna())
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
