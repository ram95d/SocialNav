from setuptools import find_packages, setup

package_name = 'rl_detect'

setup(
    name=package_name,
    version='0.0.0',
    # find_packages() automatically grabs rl_detect, scripts, rl_detect.model, rl_detect.model.datasets, etc.
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'image_geometry'],
    zip_safe=True,
    maintainer='Muhammad Rameez Ur Rahman',
    maintainer_email='rameezrehman83@gmail.com',
    description='Implementation of SocialNav: Integrated Perception System for Smart Walker',
    license='MIT License',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'intel_publisher_yolo_3dbbox_node2 = scripts.intel_yolo_3dbbox_node2:main',
            'group_detection_node = scripts.group_detection_node:main',
            'forecasting_node = scripts.forecasting_node:main',
            'face_recognition_node = scripts.face_recognition_node:main',
        ],
    },
)
