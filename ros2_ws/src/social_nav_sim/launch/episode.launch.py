"""One recorded episode: Gazebo (headless) + TurtleBot3 + Nav2 + replayed THÖR-MAGNI people.

Everything shuts down when the episode node finishes, leaving in `out_dir`:
  bag/          rosbag2 (mcap) of robot, people, Nav2 and sensor topics
  run.mp4       video from the off-screen scene camera
  episode.json  goals reached and timings

mode:=none|cv|lstm picks the H3 set-up: plain Nav2, or Nav2 with the prediction
costmap layer fed by constant-velocity or LSTM forecasts. All modes share the
same Nav2 parameters apart from that layer being enabled.
"""
import os
import tempfile

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, EmitEvent, ExecuteProcess,
                            IncludeLaunchDescription, RegisterEventHandler, TimerAction)
from launch.conditions import UnlessCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node

# Visual-only markers so the small TurtleBot stands out in the scene video: an orange
# disc on the floor (below the lidar plane) and a beacon above the lidar.
ROBOT_MARKERS = """
        <visual name="video_halo"><pose>-0.064 0 0.004 0 0 0</pose>
          <geometry><cylinder><radius>0.33</radius><length>0.008</length></cylinder></geometry>
          <material><ambient>1 0.45 0 1</ambient><diffuse>1 0.45 0 1</diffuse><emissive>0.6 0.25 0 1</emissive></material></visual>
        <visual name="video_mast"><pose>-0.064 0 0.42 0 0 0</pose>
          <geometry><cylinder><radius>0.015</radius><length>0.36</length></cylinder></geometry>
          <material><ambient>0.2 0.2 0.2 1</ambient><diffuse>0.2 0.2 0.2 1</diffuse></material></visual>
        <visual name="video_beacon"><pose>-0.064 0 0.64 0 0 0</pose>
          <geometry><sphere><radius>0.09</radius></sphere></geometry>
          <material><ambient>1 0.45 0 1</ambient><diffuse>1 0.45 0 1</diffuse><emissive>0.8 0.35 0 1</emissive></material></visual>
"""


def marked_robot_sdf(tb3_share):
    with open(os.path.join(tb3_share, 'urdf', 'gz_waffle.sdf.xacro')) as f:
        sdf = f.read()
    anchor = '</visual>\n      </link>\n\n      <link name="imu_link">'
    assert anchor in sdf, 'unexpected gz_waffle.sdf.xacro layout'
    sdf = sdf.replace(anchor, '</visual>' + ROBOT_MARKERS + '      </link>\n\n      <link name="imu_link">', 1)
    path = os.path.join(tempfile.gettempdir(), 'gz_waffle_marked.sdf.xacro')
    with open(path, 'w') as f:
        f.write(sdf)
    return path


BAG_TOPICS = ['/clock', '/tf', '/tf_static', '/odom', '/scan', '/cmd_vel', '/plan',
              '/local_plan', '/amcl_pose', '/people/poses', '/people/markers',
              '/local_costmap/costmap', '/global_costmap/costmap', '/goal_pose',
              '/predicted_people', '/predicted_people/markers']


def generate_launch_description():
    pkg = get_package_share_directory('social_nav_sim')
    bringup = get_package_share_directory('nav2_bringup')
    tb3 = get_package_share_directory('nav2_minimal_tb3_sim')
    scene = LaunchConfiguration('scene')
    out_dir = LaunchConfiguration('out_dir')
    mode = LaunchConfiguration('mode')
    params = PythonExpression(["'", os.path.join(pkg, 'config', ''), "' + ('nav2_params_baseline.yaml' if '",
                               mode, "' == 'none' else 'nav2_params_prediction.yaml')"])

    args = [
        DeclareLaunchArgument('scene', default_value='thor_sc3'),
        DeclareLaunchArgument('out_dir', default_value='/ws/results/latest'),
        DeclareLaunchArgument('mode', default_value='none', description='none | cv | lstm'),
        DeclareLaunchArgument('weights', default_value=os.path.join(pkg, 'models', 'weights_LSTM_robot.mat')),
        DeclareLaunchArgument('x', default_value='-7.0'),
        DeclareLaunchArgument('y', default_value='-1.0'),
        DeclareLaunchArgument('yaw', default_value='0.0'),
        DeclareLaunchArgument('goals', default_value='7.5,1.0,0.0;-7.0,-1.0,3.14159',
                              description='x,y,yaw;x,y,yaw;... visited in order'),
    ]

    gazebo = ExecuteProcess(
        cmd=['gz', 'sim', '-r', '-s', '--headless-rendering',
             PathJoinSubstitution([pkg, 'worlds', [scene, '.sdf']])],
        output='screen')

    robot = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(tb3, 'launch', 'spawn_tb3.launch.py')),
        launch_arguments={'use_sim_time': 'True', 'robot_sdf': marked_robot_sdf(tb3), 'x_pose': LaunchConfiguration('x'),
                          'y_pose': LaunchConfiguration('y'), 'z_pose': '0.01',
                          'yaw': LaunchConfiguration('yaw')}.items())

    with open(os.path.join(tb3, 'urdf', 'turtlebot3_waffle.urdf')) as f:
        urdf = f.read()
    state_pub = Node(package='robot_state_publisher', executable='robot_state_publisher',
                     parameters=[{'use_sim_time': True, 'robot_description': urdf}])

    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(bringup, 'launch', 'bringup_launch.py')),
        launch_arguments={'map': PathJoinSubstitution([pkg, 'maps', [scene, '.yaml']]),
                          'use_sim_time': 'True', 'params_file': params,
                          'autostart': 'True', 'use_composition': 'True'}.items())

    cam_bridge = Node(package='ros_gz_bridge', executable='parameter_bridge',
                      arguments=['/scene_cam/image@sensor_msgs/msg/Image[gz.msgs.Image'],
                      parameters=[{'use_sim_time': True}])

    people = Node(package='social_nav_sim', executable='people_publisher',
                  parameters=[{'use_sim_time': True,
                               'people_csv': PathJoinSubstitution([pkg, 'config', [scene, '_people.csv']]),
                               'scene_meta': PathJoinSubstitution([pkg, 'config', [scene, '_meta.json']])}])

    predictor = Node(package='social_nav_sim', executable='predictor', condition=UnlessCondition(
                         PythonExpression(["'", mode, "' == 'none'"])),
                     parameters=[{'use_sim_time': True, 'mode': mode,
                                  'weights': LaunchConfiguration('weights')}])

    recorder = Node(package='social_nav_sim', executable='video_recorder',
                    parameters=[{'use_sim_time': True, 'output': PathJoinSubstitution([out_dir, 'run.mp4'])}])

    bag = ExecuteProcess(cmd=['ros2', 'bag', 'record', '-s', 'mcap', '--use-sim-time',
                              '-o', PathJoinSubstitution([out_dir, 'bag']), *BAG_TOPICS],
                         output='log')

    episode = Node(package='social_nav_sim', executable='run_episode', output='screen',
                   parameters=[{'use_sim_time': True,
                                'summary': PathJoinSubstitution([out_dir, 'episode.json']),
                                'start': [LaunchConfiguration('x'), ',', LaunchConfiguration('y'), ',',
                                          LaunchConfiguration('yaw')],
                                'goals': LaunchConfiguration('goals')}])
    stop_when_done = RegisterEventHandler(OnProcessExit(
        target_action=episode, on_exit=[EmitEvent(event=Shutdown(reason='episode finished'))]))

    return LaunchDescription(args + [
        gazebo, robot, state_pub, cam_bridge, people, predictor,
        TimerAction(period=3.0, actions=[nav2]),
        TimerAction(period=5.0, actions=[recorder, bag, episode]),
        stop_when_done,
    ])
