"""Drive one episode: wait for the people to start walking, then send Nav2 through the goals.

Writes a JSON summary (reached goals, times) next to the recording.
"""
import json
import math
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult


def pose(nav, x, y, yaw):
    p = PoseStamped()
    p.header.frame_id = 'map'
    p.header.stamp = nav.get_clock().now().to_msg()
    p.pose.position.x, p.pose.position.y = x, y
    p.pose.orientation.z, p.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
    return p


def sim_time(nav):
    return nav.get_clock().now().nanoseconds * 1e-9


def main():
    rclpy.init()
    nav = BasicNavigator()
    nav.declare_parameter('start', [-7.0, -1.0, 0.0])
    nav.declare_parameter('goals', [7.5, 1.0, 0.0, -7.0, -1.0, 3.14159])
    nav.declare_parameter('start_time', 15.0)
    nav.declare_parameter('goal_timeout', 120.0)
    nav.declare_parameter('summary', '/ws/results/episode.json')
    start = nav.get_parameter('start').value
    flat = nav.get_parameter('goals').value
    goals = [flat[i:i + 3] for i in range(0, len(flat), 3)]

    nav.setInitialPose(pose(nav, *start))
    nav.waitUntilNav2Active()
    while sim_time(nav) < nav.get_parameter('start_time').value:
        rclpy.spin_once(nav, timeout_sec=0.1)

    results = []
    for g in goals:
        t0 = sim_time(nav)
        nav.goToPose(pose(nav, *g))
        while not nav.isTaskComplete():
            rclpy.spin_once(nav, timeout_sec=0.1)
            if sim_time(nav) - t0 > nav.get_parameter('goal_timeout').value:
                nav.cancelTask()
        res = nav.getResult()
        results.append({'goal': g, 'result': res.name, 'start_s': round(t0, 2),
                        'duration_s': round(sim_time(nav) - t0, 2)})
        nav.get_logger().info(f'goal {g}: {res.name} in {results[-1]["duration_s"]} s')
        if res != TaskResult.SUCCEEDED:
            break

    with open(nav.get_parameter('summary').value, 'w') as f:
        json.dump({'goals': results, 'end_s': round(sim_time(nav), 2)}, f, indent=2)
    time.sleep(1.0)
    rclpy.shutdown()
