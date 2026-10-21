import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'slam_robot'

# List every file under  the directory.
def package_files(directory):
    entries = []
    for root, _, files in os.walk(directory):
        if files:
            entries.append((os.path.join('share', package_name, root),
                            [os.path.join(root, f) for f in files]))
    return entries

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
        data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
        (os.path.join('share', package_name, 'maps'), glob('maps/*')),
    ] + package_files('models'),
    
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='tausif',
    maintainer_email='tausif3003@gmail.com',
    description='',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'navigation_node = navigation.navigation_node:main',
            'navigation_manager = navigation.navigation_manager:main',
            'detector_node = detector.detector_node:main',
        ],
    },
)