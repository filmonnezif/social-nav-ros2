"""Smoothness, comfort and human-likeness of each set-up versus the human driver.

Reads motion.json / motion.npz from every run in a grid folder (written by
motion_from_bags.py) and the human reference (human_reference.py). Writes
<grid>/motion_summary.md, motion_summary.csv and motion.png.

Human-likeness: for each motion feature, the 1-Wasserstein distance between the
set-up's pooled samples and the human driver's, in units of the human's
interquartile range (0 = same distribution). The score is the mean over features.

Usage: python experiments/analyze_motion.py results/h3_grid
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from motion_metrics import wasserstein_norm  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
MODES = ['none', 'blind', 'cv', 'lstm', 'social', 'learned']
LABEL = {'none': 'Nav2', 'cv': 'Nav2 + CV prediction', 'lstm': 'Nav2 + LSTM prediction', 'social': 'Social (LSTM + yielding + human-fitted limits)', 'blind': 'Nav2 blind to people (bulldozer)', 'learned': 'Learned driver (LSTM + cloned human speed)', 'human': 'Human driver'}
FEATURES = {'speed': 'Speed (m/s)', 'accel': '|Acceleration| (m/s²)', 'jerk': '|Jerk| (m/s³)',
            'turn_rate': 'Turn rate (rad/s)', 'robot_approach_close': 'Robot speed towards nearest person, < 1.5 m (m/s)',
            'pass_clearance': 'Clearance when passing (m)'}
SCALARS = ['mean_speed', 'rms_jerk', 'heading_change_per_m', 'stop_starts_per_min',
           'time_ttc_under_2s', 'mean_robot_approach_close', 'min_distance',
           'people_displaced_s', 'people_displacement_ms', 'mean_peak_displacement_m']


def boot_ci(x, n=5000):
    rng = np.random.default_rng(0)
    x = np.asarray(x, float)
    return np.percentile(rng.choice(x, (n, len(x))).mean(1), [2.5, 97.5])


def main():
    grid = REPO / sys.argv[1]
    human = dict(np.load(REPO / 'results/human_reference/samples.npz'))
    hscal = pd.read_csv(REPO / 'results/human_reference/scalars.csv')
    w = hscal.duration_s / hscal.duration_s.sum()

    rows, samples = [], {m: {} for m in MODES}
    present = {f.parent.name.rsplit('_', 2)[2] for f in grid.glob('*/motion.json')}
    modes = [m for m in MODES if m in present]
    for f in sorted(grid.glob('*/motion.json')):
        scene, d, mode = f.parent.name.rsplit('_', 2)
        rows.append({'scene': scene, 'dir': d, 'mode': mode, **json.loads(f.read_text())})
        for k, v in np.load(f.parent / 'motion.npz').items():
            samples[mode].setdefault(k, []).append(v)
    df = pd.DataFrame(rows)
    samples = {m: {k: np.concatenate(v) for k, v in s.items()} for m, s in samples.items() if s}

    table = []
    for mode in modes:
        sub = df[df['mode'] == mode]
        hl = {k: wasserstein_norm(samples[mode].get(k, []), human[k]) for k in FEATURES}
        table.append({'setup': LABEL[mode], 'runs': len(sub), **{k: sub[k].mean() if k in sub else np.nan for k in SCALARS},
                      'human_likeness_distance': np.nanmean(list(hl.values())),
                      **{f'W_{k}': v for k, v in hl.items()}})
    table.append({'setup': LABEL['human'], 'runs': len(hscal),
                  **{k: float((hscal[k] * w).sum()) if k in hscal else np.nan for k in SCALARS},
                  'human_likeness_distance': 0.0})
    T = pd.DataFrame(table)
    T.to_csv(grid / 'motion_summary.csv', index=False)

    have = [k for k in SCALARS + ['duration_s'] if k in df]
    piv = df.pivot_table(index=['scene', 'dir'], columns='mode', values=have).dropna()
    lines = ['# Smoothness, comfort and human-likeness', '',
             T[['setup', 'runs', 'mean_speed', 'rms_jerk', 'heading_change_per_m', 'stop_starts_per_min',
                'time_ttc_under_2s', 'mean_robot_approach_close', 'human_likeness_distance']].round(3).to_markdown(index=False),
             '', 'Burden on people (reactive crowds only): person-seconds pushed > 0.2 m off their path, '
             'displacement integral (m*s), mean peak displacement (m):', '',
             T[['setup', 'people_displaced_s', 'people_displacement_ms', 'mean_peak_displacement_m']].round(2).to_markdown(index=False),
             '', 'Human-likeness distance per feature (Wasserstein / human IQR; lower is more human-like):', '',
             T[['setup'] + [f'W_{k}' for k in FEATURES]].iloc[:len(modes)].round(2).to_markdown(index=False), '']
    for mode in modes[1:]:
        if mode == 'none' or ('rms_jerk', mode) not in piv:
            continue
        lines.append(f'## {LABEL[mode]} vs Nav2 (paired over {len(piv)} scene/direction pairs)')
        lines.append('')
        for k in [k for k in ['rms_jerk', 'heading_change_per_m', 'stop_starts_per_min', 'time_ttc_under_2s',
                              'people_displaced_s', 'people_displacement_ms'] if (k, mode) in piv]:
            d = piv[(k, mode)] - piv[(k, 'none')]
            lo, hi = boot_ci(d)
            rel = d.mean() / piv[(k, 'none')].mean()
            lines.append(f'- {k}: {d.mean():+.3f} ({100 * rel:+.0f}%), 95% CI [{lo:+.3f}, {hi:+.3f}]')
        lines.append('')
    (grid / 'motion_summary.md').write_text('\n'.join(lines))
    print('\n'.join(lines))

    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
    colors = {'none': '#8a8f98', 'blind': '#3d3d3d', 'cv': '#4c8bd6', 'lstm': '#e07b28', 'social': '#7b4fc9',
              'learned': '#c2185b', 'human': '#2a9d5c'}
    for ax, (k, title) in zip(axes.ravel(), FEATURES.items()):
        hi = np.nanpercentile(np.concatenate([human[k]] + [samples[m].get(k, np.array([0.0])) for m in modes]), 98)
        lo = min(0.0, np.nanpercentile(np.concatenate([human[k]] + [samples[m].get(k, np.array([0.0])) for m in modes]), 2))
        bins = np.linspace(lo, hi, 40)
        for m in modes + ['human']:
            data = human[k] if m == 'human' else samples[m].get(k, [])
            ax.hist(data, bins=bins, density=True, histtype='step', lw=2 if m in ('human', 'social') else 1.3,
                    color=colors[m], label=LABEL[m])
        ax.set_title(title, fontsize=10)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Robot motion vs the human driver in THÖR-MAGNI (distributions over all runs)')
    fig.tight_layout()
    fig.savefig(grid / 'motion.png', dpi=150)


if __name__ == '__main__':
    main()
