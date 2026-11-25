"""Replay THÖR-MAGNI pedestrians in Gazebo and publish them on ROS topics, in sync with sim time.

When the scene uses simple bodies (named after their helmet, e.g. `helmet_3`), this
node moves them along the recorded trajectories through Gazebo's set_pose_vector
service. Animated actors move by themselves. Either way it publishes the same
ground-truth positions:
  /people/poses    geometry_msgs/PoseArray (map frame)
  /people/markers  visualization_msgs/MarkerArray (for RViz and the rosbag)

With `reaction_model` set (results/reaction/reaction_calibration.json, simple
bodies only), people react to our robot the way THÖR-MAGNI participants reacted
to theirs: each keeps their recorded path plus an offset o driven by
o'' = A exp(-(d - R)/B) n - k o - c o' (repulsion from the robot, damped spring
back to the path). The published positions then include that offset.
"""
import json
import math

import numpy as np
import pandas as pd
import rclpy
from geometry_msgs.msg import Pose, PoseArray
from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.transport13 import Node as GzNode
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

HIDDEN = (60.0, 60.0)


class PeoplePublisher(Node):
    def __init__(self):
        super().__init__('people_publisher')
        self.declare_parameter('people_csv', '')
        self.declare_parameter('world', 'thor_magni')
        self.declare_parameter('rate', 20.0)
        self.declare_parameter('scene_meta', '')
        self.declare_parameter('reaction_model', '')
        meta = self.get_parameter('scene_meta').value
        self.drive_gazebo = not meta or json.load(open(meta)).get('people_model', 'bodies') == 'bodies'
        model = self.get_parameter('reaction_model').value
        self.reaction = json.load(open(model)) if model and self.drive_gazebo else None
        if model and not self.drive_gazebo:
            self.get_logger().warn('reaction_model needs a scene with simple bodies; replaying without reactions')
        df = pd.read_csv(self.get_parameter('people_csv').value)
        self.tracks = {b: (g.t.values, g.x.values, g.y.values) for b, g in df.groupby('body')}
        self.ids = {b: i for i, b in enumerate(sorted(self.tracks))}
        self.yaw = {b: 0.0 for b in self.tracks}
        self.offset = {b: np.zeros(2) for b in self.tracks}
        self.offset_v = {b: np.zeros(2) for b in self.tracks}
        self.dt = 1.0 / self.get_parameter('rate').value
        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self)
        self.gz = GzNode()
        self.service = f'/world/{self.get_parameter("world").value}/set_pose_vector'
        self.pub_poses = self.create_publisher(PoseArray, '/people/poses', 10)
        self.pub_markers = self.create_publisher(MarkerArray, '/people/markers', 10)
        self.create_timer(1.0 / self.get_parameter('rate').value, self.tick)

    def state(self, body, t):
        """Interpolated (x, y, vx, vy) at time t, or None while the person is not tracked."""
        ts, xs, ys = self.tracks[body]
        i = np.searchsorted(ts, t)
        if i == 0 or i >= len(ts) or ts[i] - ts[i - 1] > 0.5:
            return None
        dt = ts[i] - ts[i - 1]
        a = (t - ts[i - 1]) / dt
        vx, vy = (xs[i] - xs[i - 1]) / dt, (ys[i] - ys[i - 1]) / dt
        return xs[i - 1] + a * vx * dt, ys[i - 1] + a * vy * dt, vx, vy

    def robot_xy(self):
        try:
            tr = self.tf.lookup_transform('map', 'base_footprint', Time()).transform.translation
            return np.array([tr.x, tr.y])
        except TransformException:
            return None

    def react(self, body, x, y, robot):
        """Advance the person's reaction offset by one tick and return the displaced position."""
        m, o, v = self.reaction, self.offset[body], self.offset_v[body]
        f = np.zeros(2)
        if robot is not None:
            rel = np.array([x, y]) + o - robot
            d = max(np.linalg.norm(rel), 1e-6)
            f = m['A'] * math.exp(-(d - m['R']) / m['B']) * rel / d
        v += (f - m['k'] * o - m['c'] * v) * self.dt
        o += v * self.dt
        return x + o[0], y + o[1]

    def tick(self):
        now = self.get_clock().now()
        t = now.nanoseconds * 1e-9
        header_stamp = now.to_msg()
        poses = PoseArray()
        poses.header.frame_id = 'map'
        poses.header.stamp = header_stamp
        markers = MarkerArray()
        markers.markers.append(Marker(action=Marker.DELETEALL))
        req = Pose_V()
        robot = self.robot_xy() if self.reaction else None
        for body in self.tracks:
            s = self.state(body, t)
            gp = req.pose.add()
            gp.name = body.lower()
            if s is None:
                self.offset[body][:] = 0
                self.offset_v[body][:] = 0
                gp.position.x, gp.position.y = HIDDEN[0] + self.ids[body], HIDDEN[1]
                gp.orientation.w = 1.0
                continue
            x, y, vx, vy = s
            if self.reaction:
                x, y = self.react(body, x, y, robot)
            if math.hypot(vx, vy) > 0.15:
                self.yaw[body] = math.atan2(vy, vx)
            yaw = self.yaw[body]
            gp.position.x, gp.position.y = x, y
            gp.orientation.z, gp.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)

            p = Pose()
            p.position.x, p.position.y = float(x), float(y)
            p.orientation.z, p.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
            poses.poses.append(p)
            m = Marker()
            m.header = poses.header
            m.ns, m.id, m.type = 'people', self.ids[body], Marker.CYLINDER
            m.pose.position.x, m.pose.position.y, m.pose.position.z = p.position.x, p.position.y, 0.9
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = 0.45
            m.scale.z = 1.8
            m.color.r, m.color.g, m.color.b, m.color.a = 0.9, 0.3, 0.2, 0.8
            markers.markers.append(m)
        if self.drive_gazebo:
            self.gz.request(self.service, req, Pose_V, Boolean, 200)
        self.pub_poses.publish(poses)
        self.pub_markers.publish(markers)


def main():
    rclpy.init()
    node = PeoplePublisher()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()
