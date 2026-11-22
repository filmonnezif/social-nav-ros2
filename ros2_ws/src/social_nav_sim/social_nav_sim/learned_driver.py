"""The learned driver: let a policy cloned from the THÖR-MAGNI human driver set the robot's speed.

Every 0.4 s it builds the same input as learning/data/make_bc_dataset.py: the last
1.6 s of the robot's own velocity, a subgoal (the point on Nav2's global plan
about 1 m ahead, matching the driver's typical 4 s progress), and the 6 nearest
people's positions and velocities, all in a frame pointing at the subgoal. The
MATLAB-trained LSTM returns the velocity the human driver would have chosen; its
magnitude becomes Nav2's speed limit (/speed_limit, m/s), so the learned policy
decides when to slow, wait and go while Nav2 + the LSTM prediction layer steer.
The limit can fall immediately but rises at most `rise_rate` per second.
"""
import math
from collections import deque

import numpy as np
import rclpy
from nav2_msgs.msg import SpeedLimit
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from social_nav_sim.lstm_numpy import LSTMPredictor

DT, HIST, K = 0.4, 4, 6


def rot(v, c, s):
    return np.stack([c * v[..., 0] + s * v[..., 1], -s * v[..., 0] + c * v[..., 1]], -1)


class LearnedDriver(Node):
    def __init__(self):
        super().__init__('learned_driver')
        self.declare_parameter('weights', '')
        self.declare_parameter('subgoal_dist', 1.0)
        self.declare_parameter('v_min', 0.05)
        self.declare_parameter('v_max', 0.6)
        self.declare_parameter('rise_rate', 0.25)
        self.net = LSTMPredictor(self.get_parameter('weights').value)
        self.subgoal_dist = self.get_parameter('subgoal_dist').value
        self.v_min, self.v_max = self.get_parameter('v_min').value, self.get_parameter('v_max').value
        self.rise_rate = self.get_parameter('rise_rate').value
        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self)
        self.plan = None
        self.latest = {}
        self.people = {}
        self.robot = deque(maxlen=HIST + 1)
        self.target = self.v_max
        self.limit = self.v_max
        self.create_subscription(Path, '/plan', lambda m: setattr(self, 'plan', m), 10)
        self.create_subscription(MarkerArray, '/people/markers', self.on_people, 10)
        self.pub = self.create_publisher(SpeedLimit, '/speed_limit', 10)
        self.create_timer(DT, self.decide)
        self.create_timer(0.1, self.publish)

    def on_people(self, msg):
        self.latest = {m.id: (m.pose.position.x, m.pose.position.y) for m in msg.markers if m.action == Marker.ADD}

    def robot_xy(self):
        try:
            tr = self.tf.lookup_transform('map', 'base_footprint', Time()).transform.translation
            return np.array([tr.x, tr.y])
        except TransformException:
            return None

    def subgoal(self, pos):
        pts = np.array([[p.pose.position.x, p.pose.position.y] for p in self.plan.poses])
        i = int(np.argmin(np.linalg.norm(pts - pos, axis=1)))
        pts = pts[i:]
        s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
        d = min(self.subgoal_dist, s[-1])
        return np.array([np.interp(d, s, pts[:, 0]), np.interp(d, s, pts[:, 1])])

    def decide(self):
        for pid in list(self.people):
            if pid not in self.latest:
                del self.people[pid]
        for pid, xy in self.latest.items():
            self.people.setdefault(pid, deque(maxlen=HIST + 1)).append(np.array(xy))
        pos = self.robot_xy()
        if pos is None:
            self.robot.clear()
            return
        self.robot.append(pos)
        if len(self.robot) < HIST + 1 or self.plan is None or len(self.plan.poses) < 2:
            return
        goal = self.subgoal(pos) - pos
        if np.hypot(*goal) < 0.05:
            return
        yaw = math.atan2(goal[1], goal[0])
        c, s = math.cos(yaw), math.sin(yaw)
        hist = np.array(self.robot)
        own = rot(np.diff(hist, axis=0) / DT, c, s)
        sub = np.repeat(rot(goal, c, s)[None], HIST, 0)
        near = sorted(self.people, key=lambda p: np.hypot(*(self.people[p][-1] - pos)))[:K]
        ppl = np.zeros((HIST, K, 5))
        for j, pid in enumerate(near):
            h = np.array(self.people[pid])
            n = min(len(h) - 1, HIST)
            if n <= 0:
                continue
            ppl[HIST - n:, j, :2] = rot(h[-n:] - pos, c, s)
            ppl[HIST - n:, j, 2:4] = rot(np.diff(h[-n - 1:], axis=0) / DT, c, s)
            ppl[HIST - n:, j, 4] = 1
        x = np.concatenate([own, sub, np.transpose(ppl, (0, 2, 1)).reshape(HIST, -1)], 1)
        v = self.net.forward(x)[0]
        self.target = float(np.clip(np.hypot(*v), self.v_min, self.v_max))

    def publish(self):
        if self.target < self.limit:
            self.limit = self.target
        else:
            self.limit = min(self.target, self.limit + self.rise_rate * 0.1)
        msg = SpeedLimit()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.percentage = False
        msg.speed_limit = float(self.limit)
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = LearnedDriver()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()
