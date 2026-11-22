"""Write frames from the off-screen Gazebo scene camera to an MP4 with ffmpeg.

The sim-time stamp of every frame goes to <output>.frames.csv, so other views
(e.g. the top-down plot) can be rendered frame-for-frame in sync.
"""
import subprocess

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


class VideoRecorder(Node):
    def __init__(self):
        super().__init__('video_recorder')
        self.declare_parameter('topic', '/scene_cam/image')
        self.declare_parameter('output', '/ws/results/run.mp4')
        self.declare_parameter('fps', 15)
        self.output = self.get_parameter('output').value
        self.fps = self.get_parameter('fps').value
        self.ffmpeg = None
        self.frames = 0
        self.stamps = open(self.output + '.frames.csv', 'w')
        self.stamps.write('frame,t\n')
        self.create_subscription(Image, self.get_parameter('topic').value, self.on_image, 10)

    def on_image(self, msg: Image):
        if self.ffmpeg is None:
            pix = {'rgb8': 'rgb24', 'bgr8': 'bgr24', 'rgba8': 'rgba', 'bgra8': 'bgra'}[msg.encoding]
            self.ffmpeg = subprocess.Popen(
                ['ffmpeg', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', pix,
                 '-s', f'{msg.width}x{msg.height}', '-framerate', str(self.fps), '-i', '-',
                 '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '23', self.output],
                stdin=subprocess.PIPE)
            self.get_logger().info(f'Recording {msg.width}x{msg.height} to {self.output}')
        self.ffmpeg.stdin.write(bytes(msg.data))
        self.stamps.write(f'{self.frames},{msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9:.3f}\n')
        self.frames += 1

    def close(self):
        self.stamps.close()
        if self.ffmpeg:
            self.ffmpeg.stdin.close()
            self.ffmpeg.wait()
            self.get_logger().info(f'Wrote {self.frames} frames to {self.output}')


def main():
    rclpy.init()
    node = VideoRecorder()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.close()
        if rclpy.ok():
            rclpy.shutdown()
