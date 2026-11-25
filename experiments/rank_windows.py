"""Rank THÖR-MAGNI replay windows by how completely the people in them are tracked.

For every run and every window start (step 10 s), coverage is the share of
10 Hz samples that were really recorded, over each person's span inside the
window. A window with coverage 100% needs no gap filling at all.

Usage: python experiments/rank_windows.py tracks/*.csv --duration 110
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SKIP_BODIES = ('DARKO_Robot', 'LO1')


def score(df, t0, duration, max_gap=0.15):
    w = df[(df.t >= t0) & (df.t <= t0 + duration)]
    expected = recorded = 0
    people = worst_gap = 0
    for _, g in w.groupby('body'):
        t = np.sort(g.t.values)
        if len(t) < 10:
            continue
        people += 1
        expected += round((t[-1] - t[0]) / 0.1) + 1
        recorded += len(t)
        gaps = np.diff(t)
        worst_gap = max(worst_gap, gaps.max() if (gaps > max_gap).any() else 0)
    cov = recorded / expected if expected else 0
    return people, cov, worst_gap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('tracks', nargs='+', type=Path)
    ap.add_argument('--duration', type=float, default=110.0)
    ap.add_argument('--top', type=int, default=15)
    args = ap.parse_args()
    rows = []
    for path in args.tracks:
        df = pd.read_csv(path)
        df = df[~df.body.isin(SKIP_BODIES)]
        for t0 in np.arange(0, df.t.max() - args.duration + 1e-9, 10.0):
            people, cov, worst = score(df, t0, args.duration)
            rows.append({'run': path.stem, 't0': t0, 'people': people,
                         'coverage_%': round(100 * cov, 1), 'longest_gap_s': round(worst, 1)})
    table = pd.DataFrame(rows).sort_values(['coverage_%', 'people'], ascending=False)
    print(table.head(args.top).to_string(index=False))
    print('\nbest window per run:')
    print(table.groupby('run').head(1).to_string(index=False))


if __name__ == '__main__':
    main()
