"""Motion statistics of the human-driven robot in THÖR-MAGNI Scenario 3 (condition A).

In these 8 recordings an operator teleoperated the DARKO robot through the room
while participants walked around it. This is our human reference for
"human-like" robot driving. Writes results/human_reference/{samples.npz, scalars.csv}.

Usage: python experiments/human_reference.py <tracks_dir>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from motion_metrics import compute  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / 'results/human_reference'


def main():
    tracks = Path(sys.argv[1])
    OUT.mkdir(parents=True, exist_ok=True)
    pooled, rows = {}, []
    for f in sorted(tracks.glob('*_SC3A_*.csv')):
        d = pd.read_csv(f)
        r = d[d.body == 'DARKO_Robot'].sort_values('t')
        people = {b: (g.t.values, g.x.values, g.y.values)
                  for b, g in d[~d.body.isin(['DARKO_Robot', 'LO1'])].groupby('body')}
        # split the robot track at tracking gaps so smoothing never bridges them
        brk = np.where(np.diff(r.t.values) > 0.5)[0]
        for a, b in zip([0, *(brk + 1)], [*(brk + 1), len(r)]):
            seg = r.iloc[a:b]
            m = compute(seg.t.values, seg.x.values, seg.y.values, people)
            if m is None:
                continue
            for k, v in m['samples'].items():
                pooled.setdefault(k, []).append(v)
            rows.append({'run': f.stem, **m['scalars']})
    np.savez(OUT / 'samples.npz', **{k: np.concatenate(v) for k, v in pooled.items()})
    df = pd.DataFrame(rows)
    w = df.duration_s / df.duration_s.sum()
    summary = (df.drop(columns='run').multiply(w, axis=0).sum()).rename('duration-weighted mean')
    df.to_csv(OUT / 'scalars.csv', index=False)
    print(summary.round(3).to_string())
    print({k: f'median {np.median(np.concatenate(v)):.3f}' for k, v in pooled.items()})


if __name__ == '__main__':
    main()
