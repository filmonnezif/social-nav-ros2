"""Build a Gazebo world, a Nav2 map and a pedestrian track file from one THÖR-MAGNI run.

The room geometry comes from the dataset's floor map (1 px = 1 cm). Each recorded
person replays their real trajectory, either as an animated walking actor that
Gazebo moves itself (--people actors, the default) or as a simple body whose pose
people_publisher drives (--people bodies). The recorded teleoperated robot is
dropped, since our robot takes its place.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

MAP_RES = 0.01                   # m per pixel in the THÖR-MAGNI maps
MAP_ORIGIN = (-8.6, -10.5)       # world position of the map's lower-left corner (fitted to the tracks)
ROOM = (-8.6, 9.4, -4.1, 3.8)    # x0, x1, y0, y1 crop used for the world and the Nav2 map
GRID = 0.05                      # Nav2 map resolution / world box resolution
SKIP_BODIES = ('DARKO_Robot', 'LO1')
HIDDEN = (60.0, 60.0)            # where a person waits while not tracked


def occupancy(map_png):
    img = np.array(Image.open(map_png).convert('L'))
    h, w = img.shape
    x0, x1, y0, y1 = ROOM
    c0, c1 = int((x0 - MAP_ORIGIN[0]) / MAP_RES), int((x1 - MAP_ORIGIN[0]) / MAP_RES)
    r0, r1 = h - int((y1 - MAP_ORIGIN[1]) / MAP_RES), h - int((y0 - MAP_ORIGIN[1]) / MAP_RES)
    crop = img[r0:r1, c0:c1] < 128
    k = int(round(GRID / MAP_RES))
    hh, ww = crop.shape[0] // k, crop.shape[1] // k
    occ = crop[:hh * k, :ww * k].reshape(hh, k, ww, k).mean(axis=(1, 3)) > 0.5
    occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = True   # close the room
    return occ  # row 0 = top (max y)


def rectangles(occ):
    """Greedy merge of occupied cells into axis-aligned rectangles."""
    occ = occ.copy()
    rects = []
    for r in range(occ.shape[0]):
        c = 0
        while c < occ.shape[1]:
            if not occ[r, c]:
                c += 1
                continue
            c2 = c
            while c2 < occ.shape[1] and occ[r, c2]:
                c2 += 1
            r2 = r + 1
            while r2 < occ.shape[0] and occ[r2, c:c2].all():
                r2 += 1
            occ[r:r2, c:c2] = False
            rects.append((r, r2, c, c2))
            c = c2
    return rects


def write_nav2_map(occ, out_dir, name):
    pgm = np.where(occ, 0, 254).astype(np.uint8)
    Image.fromarray(pgm).save(out_dir / f'{name}.pgm')
    x0, _, y0, _ = ROOM
    (out_dir / f'{name}.yaml').write_text(
        f'image: {name}.pgm\nmode: trinary\nresolution: {GRID}\norigin: [{x0}, {y0}, 0.0]\n'
        'negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.25\n')


def box_sdf(occ, rects):
    x0, _, _, y1 = ROOM
    out = []
    for i, (r, r2, c, c2) in enumerate(rects):
        sx, sy = (c2 - c) * GRID, (r2 - r) * GRID
        cx, cy = x0 + (c + c2) / 2 * GRID, y1 - (r + r2) / 2 * GRID
        geo = f'<geometry><box><size>{sx:.3f} {sy:.3f} 0.6</size></box></geometry>'
        out.append(f'''
      <link name="b{i}"><pose>{cx:.3f} {cy:.3f} 0.3 0 0 0</pose>
        <collision name="c">{geo}</collision>
        <visual name="v">{geo}<material><ambient>0.55 0.57 0.62 1</ambient><diffuse>0.55 0.57 0.62 1</diffuse></material></visual>
      </link>''')
    return '<model name="room"><static>true</static>' + ''.join(out) + '\n    </model>'


def person_sdf(name, i):
    """A simple walking-person body. Its pose is driven at runtime by people_publisher."""
    hue = [(0.85, 0.33, 0.25), (0.25, 0.55, 0.85), (0.3, 0.7, 0.4), (0.9, 0.65, 0.2),
           (0.6, 0.4, 0.8), (0.2, 0.7, 0.75), (0.85, 0.45, 0.65), (0.55, 0.55, 0.3), (0.45, 0.45, 0.5)][i % 9]
    col = f'<material><ambient>{hue[0]} {hue[1]} {hue[2]} 1</ambient><diffuse>{hue[0]} {hue[1]} {hue[2]} 1</diffuse></material>'
    skin = '<material><ambient>0.9 0.75 0.6 1</ambient><diffuse>0.9 0.75 0.6 1</diffuse></material>'
    return f'''
    <model name="{name}"><static>false</static><pose>{HIDDEN[0] + i} {HIDDEN[1]} 0 0 0 0</pose>
      <link name="body"><gravity>false</gravity><kinematic>true</kinematic>
        <visual name="torso"><pose>0 0 0.95 0 0 0</pose><geometry><capsule><radius>0.2</radius><length>0.7</length></capsule></geometry>{col}</visual>
        <visual name="legs"><pose>0 0 0.4 0 0 0</pose><geometry><box><size>0.22 0.34 0.8</size></box></geometry><material><ambient>0.2 0.22 0.3 1</ambient><diffuse>0.2 0.22 0.3 1</diffuse></material></visual>
        <visual name="head"><pose>0.02 0 1.62 0 0 0</pose><geometry><sphere><radius>0.12</radius></sphere></geometry>{skin}</visual>
        <visual name="nose"><pose>0.13 0 1.62 0 0 0</pose><geometry><sphere><radius>0.04</radius></sphere></geometry>{skin}</visual>
      </link>
    </model>'''


WALK_MESH = 'https://fuel.gazebosim.org/1.0/Mingfei/models/actor/tip/files/meshes/walk.dae'


def actor_sdf(name, segs):
    """An animated actor following the recorded spans; it waits off-stage while untracked.

    The script must loop: in Gazebo Harmonic a non-looping actor script stalls the
    simulation once rendering sensors are present. A final waypoint far in the future
    keeps the loop from ever restarting during an episode.
    """
    wps = []

    def add(t, x, y, yaw):
        if wps and t <= wps[-1][0] + 1e-3:
            t = wps[-1][0] + 0.01
        wps.append((t, x, y, yaw))

    add(0.0, *HIDDEN, 0.0)
    for seg in segs:
        t, x, y = seg.t.values, seg.x.values, seg.y.values
        vx, vy = np.gradient(x), np.gradient(y)
        yaw = np.arctan2(vy, vx)
        moving = np.hypot(vx, vy) > 0.01
        for i in range(1, len(yaw)):
            if not moving[i]:
                yaw[i] = yaw[i - 1]
        yaw = np.unwrap(yaw)
        add(t[0] - 0.02, *HIDDEN, 0.0)
        for ti, xi, yi, ai in zip(t, x, y, yaw):
            add(ti, xi, yi, ai)
        add(t[-1] + 0.02, *HIDDEN, 0.0)
    add(1e5, *HIDDEN, 0.0)
    pts = ''.join(f'<waypoint><time>{t:.2f}</time><pose>{x:.3f} {y:.3f} 0 0 0 {a:.3f}</pose></waypoint>'
                  for t, x, y, a in wps)
    return f'''
    <actor name="{name}">
      <skin><filename>{WALK_MESH}</filename><scale>1.0</scale></skin>
      <animation name="walk"><filename>{WALK_MESH}</filename><interpolate_x>true</interpolate_x></animation>
      <script><loop>true</loop><delay_start>0</delay_start><auto_start>true</auto_start>
        <trajectory id="0" type="walk">{pts}</trajectory>
      </script>
    </actor>'''


def fill_gaps(df, dt=0.1, smooth_s=0.5):
    """Resample one person onto a uniform time grid over their whole tracked span.

    Motion capture loses helmets for up to ~12 s at a time while the person keeps
    walking; without filling, a replayed person would vanish and reappear metres
    away. Gaps are bridged by linear interpolation, then a short moving average
    removes marker jitter. Bridged samples are flagged with filled=1.
    """
    t = df.t.values
    grid = np.round(np.arange(t[0], t[-1] + 1e-9, dt), 2)
    x, y = np.interp(grid, t, df.x.values), np.interp(grid, t, df.y.values)
    k = max(int(round(smooth_s / dt)), 1)
    if k > 1:
        kernel = np.ones(k) / k
        pad = k // 2
        x = np.convolve(np.pad(x, pad, mode='edge'), kernel, mode='same')[pad:pad + len(grid)]
        y = np.convolve(np.pad(y, pad, mode='edge'), kernel, mode='same')[pad:pad + len(grid)]
    i = np.clip(np.searchsorted(t, grid), 1, len(t) - 1)
    nearest = np.minimum(np.abs(t[i] - grid), np.abs(t[i - 1] - grid))
    out = pd.DataFrame({'t': grid, 'x': x, 'y': y, 'filled': (nearest > 1.5 * dt).astype(int)})
    out['body'], out['role'] = df.body.iloc[0], df.role.iloc[0]
    return out


def segments(df, max_gap=0.5):
    gaps = np.where(np.diff(df.t.values) > max_gap)[0]
    edges = [0, *(gaps + 1), len(df)]
    return [df.iloc[a:b] for a, b in zip(edges[:-1], edges[1:]) if b - a >= 5]


WORLD_HEAD = '''<?xml version="1.0"?>
<sdf version="1.9">
  <world name="thor_magni">
    <physics name="1ms" type="ignored"><max_step_size>0.003</max_step_size><real_time_factor>1.0</real_time_factor></physics>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors"><render_engine>ogre2</render_engine></plugin>
    <plugin filename="gz-sim-imu-system" name="gz::sim::systems::Imu"/>
    <scene><ambient>0.6 0.6 0.6 1</ambient><background>0.85 0.88 0.92 1</background><shadows>false</shadows><grid>false</grid></scene>
    <light type="directional" name="sun"><cast_shadows>false</cast_shadows><pose>0 0 10 0 0 0</pose>
      <diffuse>0.9 0.9 0.9 1</diffuse><specular>0.2 0.2 0.2 1</specular><direction>-0.3 0.2 -0.9</direction></light>
    <model name="ground"><static>true</static><link name="l">
      <collision name="c"><geometry><plane><normal>0 0 1</normal><size>40 40</size></plane></geometry></collision>
      <visual name="v"><geometry><plane><normal>0 0 1</normal><size>40 40</size></plane></geometry>
        <material><ambient>0.93 0.93 0.9 1</ambient><diffuse>0.93 0.93 0.9 1</diffuse></material></visual></link></model>
    <model name="scene_cam"><static>true</static><pose>0.4 -8.2 10.5 0 0.95 1.5708</pose><link name="l">
      <sensor name="cam" type="camera"><topic>scene_cam/image</topic><update_rate>{fps}</update_rate><always_on>true</always_on>
        <camera><horizontal_fov>1.3</horizontal_fov><image><width>1280</width><height>720</height><format>R8G8B8</format></image>
          <clip><near>0.1</near><far>60</far></clip></camera></sensor></link></model>
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('tracks', type=Path, help='output of learning/data/thor_magni.py')
    ap.add_argument('map_png', type=Path)
    ap.add_argument('--name', default='thor_sc3')
    ap.add_argument('--t0', type=float, default=30.0, help='dataset time where the replay starts')
    ap.add_argument('--duration', type=float, default=150.0)
    ap.add_argument('--lead', type=float, default=15.0, help='sim seconds before people start walking')
    ap.add_argument('--fps', type=int, default=15)
    ap.add_argument('--people', choices=['actors', 'bodies'], default='actors')
    ap.add_argument('--out', type=Path, default=Path(__file__).resolve().parents[1] / 'ros2_ws/src/social_nav_sim')
    args = ap.parse_args()

    occ = occupancy(args.map_png)
    write_nav2_map(occ, args.out / 'maps', args.name)

    tracks = pd.read_csv(args.tracks)
    tracks = tracks[~tracks.body.isin(SKIP_BODIES)]
    win = tracks[(tracks.t >= args.t0) & (tracks.t <= args.t0 + args.duration)].copy()
    win['t'] = win.t - args.t0 + args.lead

    bodies, people = [], []
    for body, df in win.groupby('body'):
        df = df.sort_values('t')
        if len(df) < 5:
            continue
        segs = [fill_gaps(df)]
        bodies.append(actor_sdf(body.lower(), segs) if args.people == 'actors'
                      else person_sdf(body.lower(), len(bodies)))
        people.append(pd.concat(segs))
    people = pd.concat(people)
    print('bridged tracking gaps: %.1f%% of samples' % (100 * people.filled.mean()))
    people[['t', 'body', 'role', 'x', 'y', 'filled']].to_csv(args.out / 'config' / f'{args.name}_people.csv',
                                                   index=False, float_format='%.3f')

    world = (WORLD_HEAD.format(fps=args.fps) + '    ' + box_sdf(occ, rectangles(occ))
             + ''.join(bodies) + '\n  </world>\n</sdf>\n')
    (args.out / 'worlds' / f'{args.name}.sdf').write_text(world)
    meta = {'source': args.tracks.name, 'dataset_t0': args.t0, 'duration': args.duration,
            'lead': args.lead, 'people': int(people.body.nunique()), 'people_model': args.people}
    (args.out / 'config' / f'{args.name}_meta.json').write_text(json.dumps(meta, indent=2))
    print(meta, 'boxes:', len(rectangles(occ)))


if __name__ == '__main__':
    main()
