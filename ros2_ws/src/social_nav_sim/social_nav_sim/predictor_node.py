"""Forecast where each person will walk and publish it for the Nav2 prediction layer.

Every 0.4 s (sim time) it samples the tracked people (/people/markers, one id
per person) and the robot (tf map -> base_footprint), keeps 3.6 s of history,
and predicts 12 future steps (4.8 s) for everyone with a full history:
  mode=cv    constant velocity
  mode=lstm  the MATLAB-trained LSTM (own motion + 4 nearest people + robot)
Inputs are built exactly like learning/data/make_dataset.py.

Publishes /predicted_people (PoseArray, map frame; position.z = weight from 1.0
for the next step down to 0.5 at 4.8 s) and /predicted_people/markers for RViz
and the rosbag.
"""
import math
from collections import deque

import numpy as np
import rclpy
from geometry_msgs.msg import Point, Pose, PoseArray
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from social_nav_sim.lstm_numpy import LSTMPredictor

DT, OBS, FUT, K = 0.4, 8, 12, 4


def local_frame(pos):
    """Origin and heading of a track's last observed step, as in make_dataset.py."""
    h = pos[-1] - pos[-3]
    yaw = math.atan2(h[1], h[0]) if np.hypot(*h) > 0.05 else 0.0
    return pos[-1], math.cos(yaw), math.sin(yaw)


def to_local(xy, o, c, s):
    d = xy - o
    return np.stack([c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]], -1)


def rot(v, c, s):
    return np.stack([c * v[..., 0] + s * v[..., 1], -s * v[..., 0] + c * v[..., 1]], -1)


class PredictorNode(Node):
    def __init__(self):
        super().__init__('predictor')
        self.declare_parameter('mode', 'lstm')
        self.declare_parameter('weights', '')
        self.declare_parameter('robot_frame', 'base_footprint')
        self.mode = self.get_parameter('mode').value
        self.net = LSTMPredictor(self.get_parameter('weights').value) if self.mode == 'lstm' else None
        self.robot_frame = self.get_parameter('robot_frame').value
        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self)
        self.latest = {}                      # person id -> (x, y) from the last /people/markers
        self.hist = {}                        # person id -> deque of (x, y) at 2.5 Hz
        self.robot_hist = deque(maxlen=OBS + 1)
        self.create_subscription(MarkerArray, '/people/markers', self.on_people, 10)
        self.pub = self.create_publisher(PoseArray, '/predicted_people', 10)
        self.pub_markers = self.create_publisher(MarkerArray, '/predicted_people/markers', 10)
        self.create_timer(DT, self.tick)
        self.get_logger().info(f'predictor mode={self.mode}')

    def on_people(self, msg):
        self.latest = {m.id: (m.pose.position.x, m.pose.position.y)
                       for m in msg.markers if m.action == Marker.ADD}

    def robot_xy(self):
        try:
            t = self.tf.lookup_transform('map', self.robot_frame, Time())
            return t.transform.translation.x, t.transform.translation.y
        except TransformException:
            return None

    def tick(self):
        for pid in list(self.hist):
            if pid not in self.latest:
                del self.hist[pid]                 # lost: history must be contiguous
        for pid, xy in self.latest.items():
            self.hist.setdefault(pid, deque(maxlen=OBS + 1)).append(xy)
        r = self.robot_xy()
        if r is None:
            self.robot_hist.clear()
        else:
            self.robot_hist.append(r)

        ready = [pid for pid, h in self.hist.items() if len(h) == OBS + 1]
        out = PoseArray()
        out.header.frame_id = 'map'
        out.header.stamp = self.get_clock().now().to_msg()
        markers = MarkerArray()
        markers.markers.append(Marker(action=Marker.DELETEALL))
        if ready:
            futures = self.predict(ready)
            weights = np.linspace(1.0, 0.5, FUT)
            for pid, fut in zip(ready, futures):
                line = Marker()
                line.header = out.header
                line.ns, line.id, line.type = 'prediction', int(pid), Marker.LINE_STRIP
                line.scale.x = 0.06
                line.color.r, line.color.g, line.color.b, line.color.a = 0.2, 0.45, 0.9, 0.9
                line.pose.orientation.w = 1.0
                for (x, y), w in zip(fut, weights):
                    p = Pose()
                    p.position.x, p.position.y, p.position.z = float(x), float(y), float(w)
                    p.orientation.w = 1.0
                    out.poses.append(p)
                    line.points.append(Point(x=float(x), y=float(y), z=0.05))
                markers.markers.append(line)
        self.pub.publish(out)
        self.pub_markers.publish(markers)

    def predict(self, ids):
        tracks = {pid: np.array(self.hist[pid]) for pid in self.hist}
        robot = np.array(self.robot_hist) if len(self.robot_hist) == OBS + 1 else None
        X, frames = [], []
        for pid in ids:
            pos = tracks[pid]                                   # [9 x 2], oldest first
            vel = np.diff(pos, axis=0) / DT                     # [8 x 2]
            pos = pos[1:]
            o, c, s = local_frame(pos)
            frames.append((o, c, s))
            if self.mode == 'cv':
                X.append(rot(vel[-1], c, s))
                continue
            ego = np.concatenate([to_local(pos, o, c, s), rot(vel, c, s)], -1)
            others = sorted((q for q in tracks if q != pid),
                            key=lambda q: np.hypot(*(tracks[q][-1] - pos[-1])))[:K]
            nbr = np.zeros((OBS, K, 5))
            for k, q in enumerate(others):
                tq = tracks[q]
                n = min(len(tq) - 1, OBS)                      # steps with a velocity
                if n <= 0:
                    continue
                qp, qv = tq[-n:], np.diff(tq[-n - 1:], axis=0) / DT
                nbr[OBS - n:, k, :2] = to_local(qp, o, c, s)
                nbr[OBS - n:, k, 2:4] = rot(qv, c, s)
                nbr[OBS - n:, k, 4] = 1
            rob = np.zeros((OBS, 5))
            if robot is not None:
                rob[:, :2] = to_local(robot[1:], o, c, s)
                rob[:, 2:4] = rot(np.diff(robot, axis=0) / DT, c, s)
                rob[:, 4] = 1
            X.append(LSTMPredictor.features(ego, nbr, rob))
        if self.mode == 'cv':
            steps = np.arange(1, FUT + 1)[None, :, None] * DT
            local = np.array(X)[:, None, :] * steps
        else:
            local = self.net.predict(np.array(X))
        world = []
        for (o, c, s), f in zip(frames, local):
            world.append(np.stack([o[0] + c * f[:, 0] - s * f[:, 1], o[1] + s * f[:, 0] + c * f[:, 1]], -1))
        return world


def main():
    rclpy.init()
    node = PredictorNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()
