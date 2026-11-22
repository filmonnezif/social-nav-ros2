from glob import glob

from setuptools import setup

package_name = 'social_nav_sim'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/worlds', glob('worlds/*.sdf')),
        ('share/' + package_name + '/maps', glob('maps/*')),
        ('share/' + package_name + '/config', glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Filmon Nezif',
    maintainer_email='139163041+filmonnezif@users.noreply.github.com',
    description='Gazebo scenes that replay THÖR-MAGNI pedestrians around a Nav2 robot, with recording.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'people_publisher = social_nav_sim.people_publisher:main',
            'video_recorder = social_nav_sim.video_recorder:main',
            'run_episode = social_nav_sim.run_episode:main',
        ],
    },
)
