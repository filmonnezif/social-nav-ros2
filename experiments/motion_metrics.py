"""Motion, comfort and legibility metrics for a robot trajectory among people.

The same pipeline is applied to our simulated robot (from rosbag odometry) and
to the human-driven robot in THÖR-MAGNI (from motion capture), so the two can be
compared directly. Both are resampled to 10 Hz and smoothed with the same
Savitzky-Golay filter (1.1 s window, cubic) before differentiating; only the
translational motion is used, because the recorded robot's rotation is not
reliable in the dataset.

Per-sample distributions (for human-likeness):
  speed, |tangential acceleration|, |jerk|, |turn rate of the travel direction|,
  approach speed towards the nearest person when closer than 1.5 m,
  clearance at each pass (local minima of nearest-person distance below 2 m)
Per-episode scalars (for the H3 comparison):
  RMS jerk, heading change per metre, stop-starts, time with time-to-collision
  under 2 s, robot-caused approach speed, mean speed
"""
import numpy as np
from scipy.signal import savgol_filter

DT = 0.1
WIN = 11
CLOSE = 1.5
TTC_RADIUS = 0.5
TTC_HORIZON = 10.0


def resample(t, x, y):
    grid = np.arange(t[0], t[-1], DT)
    return grid, np.interp(grid, t, x), np.interp(grid, t, y)


def kinematics(t, x, y):
    t, x, y = resample(t, x, y)
    if len(t) < WIN + 2:
        return None
    sm = lambda a, d=0: savgol_filter(a, WIN, 3, deriv=d, delta=DT)  # noqa: E731
    x, y = sm(x), sm(y)
    vx, vy = sm(x, 1), sm(y, 1)
    ax, ay = sm(x, 2), sm(y, 2)
    jx, jy = sm(x, 3), sm(y, 3)
    v = np.hypot(vx, vy)
    safe = np.maximum(v, 1e-6)
    a_t = (vx * ax + vy * ay) / safe                    # tangential acceleration
    turn = (vx * ay - vy * ax) / safe ** 2              # turn rate of the travel direction
    jerk = np.hypot(jx, jy)
    return dict(t=t, x=x, y=y, vx=vx, vy=vy, v=v, a=a_t, jerk=jerk, turn=turn)


def people_at(people, t):
    """people: dict body -> (t, x, y) arrays. Returns positions/velocities [P x T x 2] and a mask."""
    P = len(people)
    pos = np.full((P, len(t), 2), np.nan)
    for i, (pt, px, py) in enumerate(people.values()):
        ok = (t >= pt[0]) & (t <= pt[-1])
        j = np.clip(np.searchsorted(pt, t), 1, len(pt) - 1)
        ok &= (pt[j] - pt[j - 1]) <= 0.5
        pos[i, ok, 0] = np.interp(t[ok], pt, px)
        pos[i, ok, 1] = np.interp(t[ok], pt, py)
    vel = np.gradient(pos, DT, axis=1)
    return pos, vel


def compute(t, x, y, people):
    k = kinematics(np.asarray(t, float), np.asarray(x, float), np.asarray(y, float))
    if k is None:
        return None
    rp = np.stack([k['x'], k['y']], -1)
    rv = np.stack([k['vx'], k['vy']], -1)
    out = {'samples': {}, 'scalars': {}}
    moving = k['v'] > 0.1
    s = out['samples']
    s['speed'] = k['v']
    s['accel'] = np.abs(k['a'])
    s['jerk'] = k['jerk']
    s['turn_rate'] = np.abs(k['turn'][moving])

    sc = out['scalars']
    path = np.sum(k['v']) * DT
    heading = np.unwrap(np.arctan2(k['vy'], k['vx']))
    dheading = np.abs(np.diff(heading))[moving[1:]]
    sc['duration_s'] = float(k['t'][-1] - k['t'][0])
    sc['mean_speed'] = float(k['v'].mean())
    sc['rms_accel'] = float(np.sqrt(np.mean(k['a'] ** 2)))
    sc['rms_jerk'] = float(np.sqrt(np.mean(k['jerk'] ** 2)))
    sc['heading_change_per_m'] = float(dheading.sum() / max(path, 1e-6))
    stop = k['v'] < 0.05
    go = k['v'] > 0.15
    starts, state = 0, None
    for st, g in zip(stop, go):
        if st and state == 'go':
            starts += 1
        state = 'stop' if st else 'go' if g else state
    sc['stop_starts_per_min'] = float(starts / max(sc['duration_s'] / 60, 1e-6))

    if people:
        pos, vel = people_at(people, k['t'])
        rel = pos - rp[None]                               # robot -> person
        dist = np.linalg.norm(rel, axis=-1)
        dist_f = np.where(np.isnan(dist), np.inf, dist)
        near = np.argmin(dist_f, axis=0)
        dmin = dist_f[near, np.arange(len(near))]
        u = rel[near, np.arange(len(near))] / np.maximum(dmin, 1e-6)[:, None]
        robot_approach = np.einsum('ij,ij->i', rv, u)        # robot speed towards nearest person
        closing = robot_approach - np.einsum('ij,ij->i', np.nan_to_num(vel[near, np.arange(len(near))]), u)
        close = dmin < CLOSE
        s['approach_speed_close'] = closing[close]
        s['robot_approach_close'] = robot_approach[close]
        # clearance at each pass: local minima of the nearest-person distance below 2 m
        mins = np.where((dmin[1:-1] < dmin[:-2]) & (dmin[1:-1] <= dmin[2:]) & (dmin[1:-1] < 2.0))[0] + 1
        s['pass_clearance'] = dmin[mins]
        # time to collision with constant velocities
        dv = np.nan_to_num(vel) - rv[None]                  # person velocity relative to robot
        a = np.sum(dv ** 2, -1)
        b = 2 * np.sum(np.nan_to_num(rel) * dv, -1)
        c = np.sum(np.nan_to_num(rel) ** 2, -1) - TTC_RADIUS ** 2
        disc = b ** 2 - 4 * a * c
        ttc = np.where((disc >= 0) & (a > 1e-9), (-b - np.sqrt(np.maximum(disc, 0))) / (2 * np.maximum(a, 1e-9)), np.inf)
        ttc = np.where(np.isnan(dist) | (ttc < 0) | (ttc > TTC_HORIZON), np.inf, ttc)
        ttc_min = ttc.min(axis=0)
        sc['min_ttc_s'] = float(ttc_min.min()) if np.isfinite(ttc_min).any() else TTC_HORIZON
        sc['time_ttc_under_2s'] = float(np.sum(ttc_min < 2.0) * DT)
        sc['mean_robot_approach_close'] = float(robot_approach[close].mean()) if close.any() else 0.0
        sc['min_distance'] = float(dmin.min())
    return out


def wasserstein_norm(a, b):
    """1-Wasserstein distance between samples a and b, in units of b's interquartile range."""
    from scipy.stats import wasserstein_distance
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return np.nan
    iqr = np.subtract(*np.percentile(b, [75, 25]))
    return float(wasserstein_distance(a, b) / max(iqr, 1e-6))
