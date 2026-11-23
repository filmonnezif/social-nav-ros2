"""Make the paper's figures and number macros from the saved results.

    python paper/make_figures.py     (from the repo root; needs numpy + matplotlib)

Writes paper/figs/*.pdf and paper/generated/macros.tex.
"""
import csv
import json
import subprocess
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'results'
FIGS = ROOT / 'paper' / 'figs'
GEN = ROOT / 'paper' / 'generated'
FIGS.mkdir(parents=True, exist_ok=True)
GEN.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8,
                     'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                     'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
COL = 3.45  # IEEE column width, inches

SETUPS = [  # (label, grid dir, mode)
    ('Nav2', 'final_grid', 'none'),
    ('Blind', 'final_grid', 'blind'),
    ('CV', 'final_grid', 'cv'),
    ('LSTM', 'final_grid', 'lstm'),
    ('Social', 'final_grid', 'social'),
    ('BC v1', 'final_grid_learned_copycat', 'learned'),
    ('BC v2', 'final_grid', 'learned'),
]
GREY, BLUE, ORANGE, RED = '#8c8c8c', '#2b6cb0', '#dd8452', '#c44e52'
COLORS = [GREY, '#b0b0b0', '#7fa7d6', BLUE, ORANGE, '#e7a0a2', RED]


def runs(grid, mode):
    out = []
    for d in sorted((RES / grid).glob(f't*_b_*_{mode}')):
        if (d / 'metrics.json').exists() and (d / 'motion.json').exists():
            m = json.load(open(d / 'metrics.json'))
            m.update(json.load(open(d / 'motion.json')))
            m['dir'] = d
            out.append(m)
    return out


def boot_ci(x, n=5000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    b = rng.choice(x, (n, len(x))).mean(1)
    return np.percentile(b, [2.5, 97.5])


def wasserstein_norm(a, b):
    """Same definition as experiments/motion_metrics.py: 1-D W1 / IQR of the human samples."""
    from motion_metrics import wasserstein_norm as w  # noqa: PLC0415
    return w(a, b)


# ---------------------------------------------------------------- data
data = {lab: runs(g, m) for lab, g, m in SETUPS}
human = np.load(RES / 'human_reference' / 'samples.npz')
import sys  # noqa: E402
sys.path.insert(0, str(ROOT / 'experiments'))

# per-step motion samples (motion.npz) stay out of git; a pooled copy is kept for the figures
POOLED = GEN / 'pooled_motion.npz'
cache = dict(np.load(POOLED)) if POOLED.exists() else {}
stats = {}
for lab, rs in data.items():
    ok = [r for r in rs if r['goals_reached'] == r['goals_total']]
    rate = [60 * r['people_displaced_s'] / r['duration_s'] for r in rs]
    if all((r['dir'] / 'motion.npz').exists() for r in rs):
        pooled = {k: np.concatenate([np.load(r['dir'] / 'motion.npz')[k] for r in rs]) for k in human.files}
        cache.update({f'{lab}|{k}': v for k, v in pooled.items()})
    else:
        pooled = {k: cache[f'{lab}|{k}'] for k in human.files}
    hl = np.mean([wasserstein_norm(pooled[k], human[k]) for k in
                  ['speed', 'accel', 'jerk', 'turn_rate', 'robot_approach_close', 'pass_clearance']])
    stats[lab] = dict(n=len(rs), success=100 * len(ok) / len(rs),
                      ttg=np.mean([sum(r['time_to_goals_s']) for r in ok]),
                      ttg_ci=boot_ci([sum(r['time_to_goals_s']) for r in ok]),
                      within=np.mean([r['time_within_personal_space_s'] for r in rs]),
                      burden=np.mean([r['people_displaced_s'] for r in rs]),
                      burden_ci=boot_ci([r['people_displaced_s'] for r in rs]),
                      rate=np.mean(rate), rate_ci=boot_ci(rate), hl=hl,
                      speed=np.mean([r['mean_speed'] for r in rs]),
                      stops=np.mean([r['stop_starts_per_min'] for r in rs]),
                      pooled=pooled)
    print(f"{lab:7s} n={len(rs)} succ={stats[lab]['success']:.0f}% ttg={stats[lab]['ttg']:.1f} "
          f"burden={stats[lab]['burden']:.1f} rate={stats[lab]['rate']:.1f} hl={hl:.3f}")

np.savez_compressed(POOLED, **cache)

# ---------------------------------------------------------------- Fig: reaction calibration
cal = json.load(open(RES / 'reaction' / 'reaction_calibration.json'))
occ = json.load(open(RES / 'reaction' / 'reaction_calibration_occupancy_only.json'))
edges = [0.0] + cal['bins_m']
mid = [(a + b) / 2 for a, b in zip(edges[:-1], edges[1:])]
fig, ax = plt.subplots(1, 2, figsize=(COL, 1.75), gridspec_kw={'width_ratios': [1.6, 1]})
ax[0].axhline(1, color='k', lw=0.5, ls=':')
ax[0].plot(mid, cal['target_ratio'], 'o-', color='k', ms=3, label='real (THÖR-MAGNI)')
ax[0].plot(mid, occ['simulated_ratio'], 's--', color=GREY, ms=2.5, label='occupancy fit')
ax[0].plot(mid, cal['simulated_ratio'], '^-', color=BLUE, ms=3, label='occupancy + effort (used)')
ax[0].set_xlabel('distance to robot (m)')
ax[0].set_ylabel('occupancy ratio')
fig.legend(*ax[0].get_legend_handles_labels(), frameon=False, loc='upper center', ncol=3, fontsize=6,
           handlelength=1.6, columnspacing=0.8)
w = 0.35
x = np.arange(3)
ax[1].bar(x - w / 2, cal['target_extra_rms_acc'], w, color='k', label='real')
ax[1].bar(x + w / 2, cal['simulated_extra_rms_acc'], w, color=BLUE, label='model')
ax[1].set_xticks(x, ['<1.5', '1.5–2.5', '>2.5'], fontsize=6)
ax[1].set_xlabel('distance (m)')
ax[1].set_ylabel('extra RMS accel. (m/s$^2$)')
ax[1].legend(frameon=False)
fig.tight_layout(pad=0.3, w_pad=0.8, rect=(0, 0, 1, 0.88))
fig.savefig(FIGS / 'reaction.pdf')

# ---------------------------------------------------------------- Fig: main results
labs = [s[0] for s in SETUPS]
fig, ax = plt.subplots(1, 3, figsize=(2 * COL + 0.25, 1.55))
panels = [('ttg', 'ttg_ci', 'time to goal (s)'), ('rate', 'rate_ci', 'burden rate (person-s/min)'),
          ('hl', None, 'human-likeness distance')]
for a, (k, ci, title) in zip(ax, panels):
    v = [stats[l][k] for l in labs]
    err = None
    if ci:
        err = np.array([[stats[l][k] - stats[l][ci][0], stats[l][ci][1] - stats[l][k]] for l in labs]).T
    a.bar(range(len(labs)), v, color=COLORS, yerr=err, error_kw={'lw': 0.6, 'capsize': 1.5})
    a.set_xticks(range(len(labs)), labs, rotation=35, ha='right')
    a.set_title(title)
    if k == 'hl':
        a.set_ylim(0.6, 0.85)
    if k == 'rate':
        a.set_ylim(6, 16)
fig.tight_layout(pad=0.3, w_pad=1.0)
fig.savefig(FIGS / 'results.pdf')

# ---------------------------------------------------------------- Fig: distributions vs the human driver
fig, ax = plt.subplots(1, 3, figsize=(2 * COL + 0.25, 1.5))
feats = [('speed', 'speed (m/s)', (0, 0.7)), ('turn_rate', 'turn rate (rad/s)', (0, 1)),
         ('pass_clearance', 'passing clearance (m)', (0, 3))]
show = [('Nav2', GREY), ('LSTM', BLUE), ('BC v2', RED)]
for a, (k, xl, rng_) in zip(ax, feats):
    bins = np.linspace(*rng_, 30)
    a.hist(human[k], bins, density=True, color='k', alpha=0.18, label='human driver')
    for lab, c in show:
        a.hist(stats[lab]['pooled'][k], bins, density=True, histtype='step', color=c, lw=1, label=lab)
    a.set_xlabel(xl)
    a.set_yticks([])
fig.legend(*ax[0].get_legend_handles_labels(), frameon=False, loc='upper center', ncol=4)
fig.tight_layout(pad=0.3, w_pad=1.0, rect=(0, 0, 1, 0.86))
fig.savefig(FIGS / 'distributions.pdf')

# ---------------------------------------------------------------- Fig: scene frame
gif = ROOT / 'docs' / 'media' / 'baseline_walking.gif'
# the Gazebo view (top) and the top-down plot (bottom) of frame 60, side by side
subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', str(gif), '-filter_complex',
                'select=eq(n\\,60),split[a][b];[a]crop=560:230:0:40,scale=-1:235[t];[b]crop=510:235:25:325[d];'
                '[t][d]hstack', '-frames:v', '1', str(FIGS / 'scene.png')], check=True)

# top row of the MATLAB predictor examples (results/predictor/examples.png)
subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', str(RES / 'predictor' / 'examples.png'), '-vf',
                'crop=1135:290:0:0', str(FIGS / 'pred_examples.png')], check=True)

# ---------------------------------------------------------------- macros
pred = {r['model']: r for r in csv.DictReader(open(RES / 'predictor' / 'metrics.csv'))}
drv = {r['model']: r for r in csv.DictReader(open(RES / 'driver' / 'metrics.csv'))}
hum = list(csv.DictReader(open(RES / 'human_reference' / 'scalars.csv')))
m_hours = sum(float(r['duration_s']) for r in hum) / 60
with open(GEN / 'macros.tex', 'w') as f:
    def m(name, val):
        f.write(f'\\newcommand{{\\{name}}}{{{val}}}\n')
    for k, s in zip(['Nav', 'Blind', 'CV', 'LSTM', 'Social', 'BCa', 'BCb'], stats.values()):
        m(f'succ{k}', f"{s['success']:.0f}")
        m(f'ttg{k}', f"{s['ttg']:.0f}")
        m(f'within{k}', f"{s['within']:.2f}")
        m(f'burden{k}', f"{s['burden']:.1f}")
        m(f'rate{k}', f"{s['rate']:.1f}")
        m(f'hl{k}', f"{s['hl']:.2f}")
        m(f'speed{k}', f"{s['speed']:.2f}")
        m(f'stops{k}', f"{s['stops']:.1f}")
    for k, key in [('CV', 'CV'), ('Ego', 'LSTM-ego'), ('Soc', 'LSTM-social'), ('Rob', 'LSTM-robot')]:
        m(f'ade{k}', f"{float(pred[key]['ADE_m']):.3f}")
        m(f'fde{k}', f"{float(pred[key]['FDE_m']):.3f}")
        m(f'adenear{k}', f"{float(pred[key]['ADE_near_robot_m']):.3f}")
    for k, key in [('Keep', 'KEEP'), ('Goal', 'GOAL_PACE'), ('Blind', 'BC_blind'), ('Social', 'BC_social'),
                   ('Nocopy', 'BC_nocopy'), ('People', 'BC_people')]:
        m(f'dvel{k}', f"{float(drv[key]['vel_err_ms']):.3f}")
        m(f'dspd{k}', f"{float(drv[key]['speed_err_near_ms']):.3f}")
        m(f'dyield{k}', f"{100 * float(drv[key]['yield_recall']):.0f}")
    href = next(r for r in csv.DictReader(open(RES / 'final_grid' / 'motion_summary.csv')) if r['setup'] == 'Human driver')
    m('humSpeed', f"{float(href['mean_speed']):.2f}")
    m('humStops', f"{float(href['stop_starts_per_min']):.1f}")
    m('humTTC', f"{float(href['time_ttc_under_2s']):.1f}")
    m('humMinDist', f"{np.mean([float(r['min_distance']) for r in hum]):.2f}")
    m('humMinutes', f"{m_hours:.0f}")
    m('humPass', f"{np.median(human['pass_clearance']):.2f}")
print('figures and macros written')
