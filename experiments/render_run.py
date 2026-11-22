"""Turn one recorded run into a demo video and a summary.

Reads the rosbag of a run, draws a top-down view (map, people with trails, robot
path, Nav2 global plan, distance to the closest person) for every frame of the
Gazebo scene video, stacks both views into demo.mp4, and writes metrics.json.

Runs inside the experiment container:
  python3 experiments/render_run.py results/<run> [scene] [--no-video]
--no-video writes metrics.json only (fast; used by the experiment grid).
"""
import json
import math
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402
from rclpy.serialization import deserialize_message  # noqa: E402
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions  # noqa: E402
from rosidl_runtime_py.utilities import get_message  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
PKG = REPO / 'ros2_ws/src/social_nav_sim'
HUES = [(0.85, 0.33, 0.25), (0.25, 0.55, 0.85), (0.3, 0.7, 0.4), (0.9, 0.65, 0.2),
        (0.6, 0.4, 0.8), (0.2, 0.7, 0.75), (0.85, 0.45, 0.65), (0.55, 0.55, 0.3), (0.45, 0.45, 0.5)]
ROBOT = (1.0, 0.45, 0.0)
PERSONAL_SPACE = 0.5  # m, centre-to-centre distance counted as an intrusion (robot + person radii ~0.45 m)


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def read_bag(bag_dir):
    reader = SequentialReader()
    reader.open(StorageOptions(uri=str(bag_dir), storage_id='mcap'), ConverterOptions('', ''))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    odom, map_odom, plans, people, preds = [], [], [], [], []
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic not in ('/odom', '/tf', '/plan', '/people/poses', '/predicted_people'):
            continue
        msg = deserialize_message(data, get_message(types[topic]))
        if topic == '/odom':
            p = msg.pose.pose
            odom.append((stamp(msg.header), p.position.x, p.position.y, yaw_of(p.orientation)))
        elif topic == '/tf':
            for tf in msg.transforms:
                if tf.header.frame_id == 'map' and tf.child_frame_id == 'odom':
                    tr = tf.transform
                    map_odom.append((stamp(tf.header), tr.translation.x, tr.translation.y, yaw_of(tr.rotation)))
        elif topic == '/plan':
            plans.append((stamp(msg.header), np.array([[q.pose.position.x, q.pose.position.y] for q in msg.poses])))
        elif topic == '/predicted_people':
            preds.append((stamp(msg.header), np.array([[q.position.x, q.position.y] for q in msg.poses])))
        else:
            people.append((stamp(msg.header), np.array([[q.position.x, q.position.y] for q in msg.poses])))
    return np.array(odom), np.array(map_odom), plans, people, preds


def stamp(h):
    return h.stamp.sec + h.stamp.nanosec * 1e-9


def robot_in_map(odom, map_odom):
    out = []
    for t, x, y, a in odom:
        i = max(np.searchsorted(map_odom[:, 0], t) - 1, 0) if len(map_odom) else None
        ox, oy, oa = map_odom[i, 1:] if i is not None else (0.0, 0.0, 0.0)
        c, s = math.cos(oa), math.sin(oa)
        out.append((t, ox + c * x - s * y, oy + s * x + c * y, oa + a))
    return np.array(out)


def latest(seq, t):
    best = None
    for ts, v in seq:
        if ts > t:
            break
        best = v
    return best


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    run = Path(args[0]).resolve()
    scene = args[1] if len(args) > 1 else 'thor_sc3'
    odom, map_odom, plans, people_msgs, preds = read_bag(run / 'bag')
    robot = robot_in_map(odom, map_odom)
    frames = np.loadtxt(run / 'run.mp4.frames.csv', delimiter=',', skiprows=1)[:, 1]

    import pandas as pd
    tracks = pd.read_csv(PKG / 'config' / f'{scene}_people.csv')
    bodies = sorted(tracks.body.unique())
    tracks_by_body = {b: g.sort_values('t') for b, g in tracks.groupby('body')}

    # metrics from ground-truth people positions at robot odometry rate
    dmin = []
    for t, x, y, _ in robot:
        pts = []
        for b in bodies:
            g = tracks_by_body[b]
            i = np.searchsorted(g.t.values, t)
            if 0 < i < len(g) and g.t.values[i] - g.t.values[i - 1] <= 0.5:
                pts.append((np.interp(t, g.t.values, g.x.values), np.interp(t, g.t.values, g.y.values)))
        dmin.append(min((math.hypot(px - x, py - y) for px, py in pts), default=np.inf))
    dmin = np.array(dmin)
    episode = json.loads((run / 'episode.json').read_text())
    moving = (robot[:, 0] >= episode['goals'][0]['start_s']) & (robot[:, 0] <= episode['end_s'])
    dt = np.diff(robot[:, 0], prepend=robot[0, 0])
    metrics = {
        'goals_reached': sum(g['result'] == 'SUCCEEDED' for g in episode['goals']),
        'goals_total': len(episode['goals']),
        'time_to_goals_s': [g['duration_s'] for g in episode['goals']],
        'path_length_m': round(float(np.sum(np.hypot(np.diff(robot[moving, 1]), np.diff(robot[moving, 2])))), 2),
        'min_distance_to_person_m': round(float(dmin[moving].min()), 3),
        'time_within_personal_space_s': round(float(dt[moving & (dmin < PERSONAL_SPACE)].sum()), 2),
        'time_within_1m_s': round(float(dt[moving & (dmin < 1.0)].sum()), 2),
        'personal_space_threshold_m': PERSONAL_SPACE,
    }
    (run / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    if '--no-video' in sys.argv:
        return

    # top-down video, one frame per scene-camera frame
    meta = yaml.safe_load((PKG / 'maps' / f'{scene}.yaml').read_text())
    img = np.array(Image.open(PKG / 'maps' / meta['image']))
    h, w = img.shape
    x0, y0 = meta['origin'][:2]
    ext = [x0, x0 + w * meta['resolution'], y0, y0 + h * meta['resolution']]
    W, H = 1280, 600
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = fig.add_axes([0.03, 0.08, 0.94, 0.84])
    ff = subprocess.Popen(['ffmpeg', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgba',
                           '-s', f'{W}x{H}', '-framerate', '15', '-i', '-', '-c:v', 'libx264',
                           '-pix_fmt', 'yuv420p', '-crf', '23', str(run / 'topdown.mp4')], stdin=subprocess.PIPE)
    for t in frames:
        ax.clear()
        ax.imshow(np.where(img < 128, 0.55, 0.97), cmap='gray', vmin=0, vmax=1, extent=ext)
        for k, b in enumerate(bodies):
            g = tracks_by_body[b]
            m = (g.t.values <= t) & (g.t.values >= t - 4)
            if not m.any() or t - g.t.values[m][-1] > 0.3:
                continue
            ax.plot(g.x.values[m], g.y.values[m], '-', color=HUES[k % 9], lw=1.5, alpha=0.6)
            ax.add_patch(plt.Circle((g.x.values[m][-1], g.y.values[m][-1]), 0.22, color=HUES[k % 9]))
        pr = latest(preds, t)
        if pr is not None and len(pr):
            ax.plot(pr[:, 0], pr[:, 1], '.', color=(0.2, 0.45, 0.9), ms=3, alpha=0.5, label='predicted paths')
        plan = latest(plans, t)
        if plan is not None and len(plan):
            ax.plot(plan[:, 0], plan[:, 1], '--', color=(0.2, 0.45, 0.9), lw=1.2, label='Nav2 global plan')
        r = robot[robot[:, 0] <= t]
        if len(r):
            ax.plot(r[:, 1], r[:, 2], '-', color=ROBOT, lw=2, alpha=0.8, label='robot path')
            _, rx, ry, ra = r[-1]
            ax.add_patch(plt.Circle((rx, ry), 0.2, color=ROBOT, zorder=5))
            ax.plot([rx, rx + 0.4 * math.cos(ra)], [ry, ry + 0.4 * math.sin(ra)], color='k', lw=2, zorder=6)
            d = dmin[len(r) - 1]
            dtxt = f'closest person {d:.2f} m' if np.isfinite(d) else 'no person tracked'
        else:
            dtxt = ''
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'Top-down  |  sim time {t:6.1f} s  |  {dtxt}', fontsize=12, loc='left')
        ax.legend(loc='lower left', fontsize=9, framealpha=0.8)
        fig.canvas.draw()
        ff.stdin.write(fig.canvas.buffer_rgba().tobytes())
    ff.stdin.close()
    ff.wait()

    subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-i', str(run / 'run.mp4'), '-i', str(run / 'topdown.mp4'),
                    '-filter_complex', '[0:v][1:v]vstack=inputs=2', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                    '-crf', '24', str(run / 'demo.mp4')], check=True)
    print('wrote', run / 'demo.mp4')


if __name__ == '__main__':
    main()
