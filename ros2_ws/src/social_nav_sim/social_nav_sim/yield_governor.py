"""Human-like yielding: slow down smoothly when a forecast path will cross ours.

Every 0.1 s it rolls the robot forward along its current global plan at its
current speed limit, compares that with every person's forecast
(/predicted_people/markers, one 4.8 s line per person, 0.4 s per point), and
finds the earliest predicted conflict (closer than `conflict_dist`). The speed
limit then scales with the time left before that conflict, and is capped to
`approach_cap` while a person is within `approach_radius` inside a cone in front
of the robot. Profiles live in config/governor_*.yaml. The limit
drops as fast as needed but rises at most `rise_rate` per second, so the robot
does not lurch. Thresholds default to statistics of the human driver in
THÖR-MAGNI (see experiments/human_reference.py). Publishes nav2_msgs/SpeedLimit
on /speed_limit (absolute m/s).
"""
import math

import numpy as np
import rclpy
from nav2_msgs.msg import SpeedLimit
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

STEP = 0.4


class YieldGovernor(Node):
    def __init__(self):
        super().__init__('yield_governor')
        self.declare_parameter('v_max', 0.47)          # human driver p95 speed
        self.declare_parameter('v_min', 0.08)
        self.declare_parameter('conflict_dist', 1.0)    # m, below the human's typical 1.4 m passing clearance
        self.declare_parameter('t_safe', 1.2)           # s: conflicts nearer than this -> v_min
        self.declare_parameter('t_free', 4.0)           # s: conflicts further than this -> v_max
        self.declare_parameter('approach_cap', 0.27)    # human p95 speed towards a person within 1.5 m
        self.declare_parameter('approach_radius', 1.5)  # m
        self.declare_parameter('approach_cone_deg', 90.0)  # half-angle of "in front"
        self.declare_parameter('rise_rate', 0.15)       # m/s per s
        p = {n: self.get_parameter(n).value for n in
             ('v_max', 'v_min', 'conflict_dist', 't_safe', 't_free', 'approach_cap',
              'approach_radius', 'approach_cone_deg', 'rise_rate')}
        self.__dict__.update(p)
        self.plan = None
        self.forecasts = []
        self.limit = self.v_max
        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self)
        self.create_subscription(Path, '/plan', lambda m: setattr(self, 'plan', m), 10)
        self.create_subscription(MarkerArray, '/predicted_people/markers', self.on_forecasts, 10)
        self.pub = self.create_publisher(SpeedLimit, '/speed_limit', 10)
        self.create_timer(0.1, self.tick)

    def on_forecasts(self, msg):
        self.forecasts = [np.array([[p.x, p.y] for p in m.points]) for m in msg.markers
                          if m.type == Marker.LINE_STRIP and len(m.points)]

    def robot(self):
        try:
            t = self.tf.lookup_transform('map', 'base_footprint', Time())
        except TransformException:
            return None
        q = t.transform.rotation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        return np.array([t.transform.translation.x, t.transform.translation.y]), yaw

    def rollout(self, pos, speed, n):
        """Robot positions at 0.4 s steps along the plan, starting from the closest plan point."""
        pts = np.array([[ps.pose.position.x, ps.pose.position.y] for ps in self.plan.poses])
        i = int(np.argmin(np.linalg.norm(pts - pos, axis=1)))
        pts = pts[i:]
        if len(pts) < 2:
            return np.repeat(pos[None], n, 0)
        s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
        d = np.minimum(np.arange(1, n + 1) * STEP * speed, s[-1])
        return np.stack([np.interp(d, s, pts[:, 0]), np.interp(d, s, pts[:, 1])], -1)

    def tick(self):
        r = self.robot()
        target = self.v_max
        if r is not None and self.plan is not None and len(self.plan.poses) > 1 and self.forecasts:
            pos, yaw = r
            heading = np.array([math.cos(yaw), math.sin(yaw)])
            n = max(len(f) for f in self.forecasts)
            path = self.rollout(pos, max(self.limit, 0.2), n)
            t_conflict = math.inf
            for f in self.forecasts:
                k = min(len(f), len(path))
                hit = np.where(np.linalg.norm(f[:k] - path[:k], axis=1) < self.conflict_dist)[0]
                if len(hit):
                    t_conflict = min(t_conflict, (hit[0] + 1) * STEP)
                rel = f[0] - pos                                   # where the person is about now
                dist = np.linalg.norm(rel)
                cos_cone = math.cos(math.radians(self.approach_cone_deg))
                if dist < self.approach_radius and rel @ heading > cos_cone * dist:
                    target = min(target, self.approach_cap)
            if math.isfinite(t_conflict):
                a = (t_conflict - self.t_safe) / (self.t_free - self.t_safe)
                target = min(target, self.v_min + (self.v_max - self.v_min) * min(max(a, 0.0), 1.0))
        self.limit = target if target < self.limit else min(target, self.limit + self.rise_rate * 0.1)
        msg = SpeedLimit()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.percentage = False
        msg.speed_limit = float(self.limit)
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = YieldGovernor()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()
