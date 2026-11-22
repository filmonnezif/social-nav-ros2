"""Replay THÖR-MAGNI pedestrians in Gazebo and publish them on ROS topics, in sync with sim time.

When the scene uses simple bodies (named after their helmet, e.g. `helmet_3`), this
node moves them along the recorded trajectories through Gazebo's set_pose_vector
service. Animated actors move by themselves. Either way it publishes the same
ground-truth positions:
  /people/poses    geometry_msgs/PoseArray (map frame)
  /people/markers  visualization_msgs/MarkerArray (for RViz and the rosbag)
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
from visualization_msgs.msg import Marker, MarkerArray

HIDDEN = (60.0, 60.0)


class PeoplePublisher(Node):
    def __init__(self):
        super().__init__('people_publisher')
        self.declare_parameter('people_csv', '')
        self.declare_parameter('world', 'thor_magni')
        self.declare_parameter('rate', 20.0)
        self.declare_parameter('scene_meta', '')
        meta = self.get_parameter('scene_meta').value
        self.drive_gazebo = not meta or json.load(open(meta)).get('people_model', 'bodies') == 'bodies'
        df = pd.read_csv(self.get_parameter('people_csv').value)
        self.tracks = {b: (g.t.values, g.x.values, g.y.values) for b, g in df.groupby('body')}
        self.ids = {b: i for i, b in enumerate(sorted(self.tracks))}
        self.yaw = {b: 0.0 for b in self.tracks}
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
        for body in self.tracks:
            s = self.state(body, t)
            gp = req.pose.add()
            gp.name = body.lower()
            if s is None:
                gp.position.x, gp.position.y = HIDDEN[0] + self.ids[body], HIDDEN[1]
                gp.orientation.w = 1.0
                continue
            x, y, vx, vy = s
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
