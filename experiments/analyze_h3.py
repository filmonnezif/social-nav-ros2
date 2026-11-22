"""Summarise the H3 grid: plain Nav2 vs Nav2 + constant-velocity / LSTM prediction.

Reads results/h3_grid/*/metrics.json (run names are <scene>_<LR|RL>_<mode>) and
writes results/h3_grid/summary.csv, summary.md and h3.png.

H3 (set before the runs): compared with plain Nav2, prediction raises the
minimum distance to people, cuts time within 0.5 m by at least 30%, adds at
most 20% time to goal, and adds no failures. Differences are paired per
scene and direction; 95% intervals come from a bootstrap over those pairs.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1] / 'results/h3_grid'
MODES = ['none', 'cv', 'lstm', 'social']
LABEL = {'none': 'Nav2', 'cv': 'Nav2 + CV prediction', 'lstm': 'Nav2 + LSTM prediction', 'social': 'Social (LSTM + yielding + human-fitted limits)'}


def load():
    rows = []
    for m in sorted(ROOT.glob('*/metrics.json')):
        scene, direction, mode = m.parent.name.rsplit('_', 2)
        d = json.loads(m.read_text())
        rows.append({'scene': scene, 'dir': direction, 'mode': mode,
                     'success': d['goals_reached'] == d['goals_total'],
                     'time_s': d['time_to_goals_s'][0], 'path_m': d['path_length_m'],
                     'min_dist_m': d['min_distance_to_person_m'],
                     't_within_05_s': d['time_within_personal_space_s'],
                     't_within_1_s': d['time_within_1m_s'],
                     'robot_close_05_s': d.get('robot_closing_within_0.5m_s', np.nan),
                     'robot_close_1_s': d.get('robot_closing_within_1m_s', np.nan)})
    return pd.DataFrame(rows)


def boot_ci(x, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    means = rng.choice(x, (n, len(x))).mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def main():
    df = load()
    modes = [m for m in MODES if m in set(df['mode'])]
    pairs = df.pivot_table(index=['scene', 'dir'], columns='mode',
                           values=['success', 'time_s', 'min_dist_m', 't_within_05_s', 't_within_1_s',
                                   'robot_close_05_s', 'robot_close_1_s'])
    pairs = pairs.dropna()                       # only scene/direction pairs with all three modes
    n = len(pairs)

    summary = []
    for mode in modes:
        s = pairs.xs(mode, axis=1, level=1)
        ok = s['success'] > 0.5
        summary.append({'setup': LABEL[mode], 'runs': n, 'success_%': 100 * ok.mean(),
                        'time_to_goal_s (successful)': s['time_s'][ok].mean(),
                        'min_dist_m (mean)': s['min_dist_m'].mean(),
                        'time_within_0.5m_s (mean)': s['t_within_05_s'].mean(),
                        'time_within_1m_s (mean)': s['t_within_1_s'].mean(),
                        'robot-caused within 0.5m_s (mean)': s['robot_close_05_s'].mean(),
                        'robot-caused within 1m_s (mean)': s['robot_close_1_s'].mean()})
    table = pd.DataFrame(summary)
    table.to_csv(ROOT / 'summary.csv', index=False)

    lines = [f'# H3 results ({n} paired scene/direction runs per set-up)', '',
             table.round(2).to_markdown(index=False), '']
    base = pairs.xs('none', axis=1, level=1)
    for mode in modes[1:]:
        s = pairs.xs(mode, axis=1, level=1)
        both = (s['success'] > 0.5) & (base['success'] > 0.5)
        d05 = s['t_within_05_s'] - base['t_within_05_s']
        dmin = s['min_dist_m'] - base['min_dist_m']
        red05 = 1 - s['t_within_05_s'].mean() / max(base['t_within_05_s'].mean(), 1e-9)
        dtime = s['time_s'][both].mean() / base['time_s'][both].mean() - 1 if both.any() else float('nan')
        fail_extra = int(((s['success'] < 0.5) & (base['success'] > 0.5)).sum())
        fail_fixed = int(((s['success'] > 0.5) & (base['success'] < 0.5)).sum())
        ok = (dmin.mean() > 0) and red05 >= 0.30 and dtime <= 0.20 and \
            (s['success'] > 0.5).sum() >= (base['success'] > 0.5).sum()
        lo, hi = boot_ci(d05)
        lo2, hi2 = boot_ci(dmin)
        drc = s['robot_close_1_s'] - base['robot_close_1_s']
        red_rc = 1 - s['robot_close_1_s'].mean() / max(base['robot_close_1_s'].mean(), 1e-9)
        lo3, hi3 = boot_ci(drc)
        lines += [f'## {LABEL[mode]} vs Nav2', '',
                  f'- time within 0.5 m: {-100 * red05:+.0f}% change in the mean '
                  f'(paired difference {d05.mean():+.2f} s, 95% CI [{lo:+.2f}, {hi:+.2f}])',
                  f'- minimum distance: {dmin.mean():+.3f} m (95% CI [{lo2:+.3f}, {hi2:+.3f}])',
                  f'- time to goal (runs both completed): {100 * dtime:+.0f}%',
                  f'- robot-caused time within 1 m: {-100 * red_rc:+.0f}% '
                  f'(paired difference {drc.mean():+.2f} s, 95% CI [{lo3:+.2f}, {hi3:+.2f}])',
                  f'- failures: {fail_extra} new, {fail_fixed} fixed',
                  f'- **H3 {"SUPPORTED" if ok else "NOT SUPPORTED"}** '
                  '(needs: min distance up, time within 0.5 m down >= 30%, time to goal <= +20%, no extra failures)', '']
    (ROOT / 'summary.md').write_text('\n'.join(lines))
    print('\n'.join(lines))

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    colors = ['#8a8f98', '#4c8bd6', '#e07b28', '#7b4fc9'][:len(modes)]
    for ax, (col, title) in zip(axes, [('success_%', 'Reached goal (%)'),
                                       ('time_within_0.5m_s (mean)', 'Time within 0.5 m (s)'),
                                       ('time_within_1m_s (mean)', 'Time within 1 m (s)'),
                                       ('time_to_goal_s (successful)', 'Time to goal (s)')]):
        ax.bar(range(len(modes)), table[col], color=colors)
        ax.set_xticks(range(len(modes)), [{'none': 'Nav2', 'cv': '+CV', 'lstm': '+LSTM', 'social': 'Social'}[m] for m in modes])
        ax.set_title(title)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle(f'H3: {n} paired runs per set-up on the held-out test day')
    fig.tight_layout()
    fig.savefig(ROOT / 'h3.png', dpi=150)


if __name__ == '__main__':
    main()
