from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os

def generate_launch_description():
    checkpoint_path = os.path.expanduser('~/SocialNav/src/rl_detect/rl_detect/checkpoints/m1.ckpt')# <--- UPDATE THIS PATH
    det_checkpoint_path = os.path.expanduser('~/SocialNav/src/rl_detect/rl_detect/weights/det_10g.onnx')# <--- UPDATE THIS PATH
    rec_checkpoint_path = os.path.expanduser('~/SocialNav/src/rl_detect/rl_detect/weights/w600k_mbf.onnx')# <--- UPDATE THIS PATH
    faces_dir = os.path.expanduser('~/SocialNav/src/rl_detect/rl_detect/faces2')# <--- UPDATE THIS PATH
    db_path = os.path.expanduser('~/SocialNav/src/rl_detect/rl_detect/database/face_database')# <--- UPDATE THIS PATH
    camera_info_sub_topic = '/oak/stereo/camera_info' # <--- UPDATE THIS IF YOUR CAMERA INFO TOPIC IS DIFFERENT
    color_image_sub_topic = '/oak/rgb/image_raw' # <--- UPDATE THIS IF YOUR COLOR IMAGE TOPIC IS DIFFERENT
    depth_image_sub_topic = '/oak/stereo/image_raw' # <--- UPDATE THIS IF YOUR DEPTH IMAGE TOPIC IS DIFFERENT
    # camera_info_sub_topic = '/camera/camera/color/camera_info' # for realsense d435i, update if using different camera or topic names
    # color_image_sub_topic = '/camera/camera/color/image_raw' # for realsense d435i, update if using different camera or topic names
    # depth_image_sub_topic = '/camera/camera/aligned_depth_to_color/image_raw' # for realsense d435i, update if using different camera or topic names
    start_yolo = LaunchConfiguration('start_yolo')
    start_group = LaunchConfiguration('start_group')
    start_forecasting = LaunchConfiguration('start_forecasting')
    start_face = LaunchConfiguration('start_face')
    return LaunchDescription([
        DeclareLaunchArgument(
            'start_yolo',
            default_value='true',
            description='Start main YOLO detection node'
        ),
        DeclareLaunchArgument(
            'start_group',
            default_value='true',
            description='Start group detection node'
        ),
        DeclareLaunchArgument(
            'start_forecasting',
            default_value='true',
            description='Start trajectory forecasting node'
        ),
        DeclareLaunchArgument(
            'start_face',
            default_value='true',
            description='Start face recognition node'
        ),

        # Main detection + tracking node
        Node(
            package='rl_detect',
            executable='intel_publisher_yolo_3dbbox_node2',
            name='yolo_detector_node',
            output='screen',
            condition=IfCondition(start_yolo),
            parameters=[
                {'model':              'yoloe-26m-seg.pt'},
                {'classes':            'person,chair'},
                {'output_markers':     'True'},
                {'output_pointcloud':  'False'},
                {'output_detection_3d':'True'},
                {'enable_tracking':    'True'}, 
                {'sync_slop':          0.05},
                {'camera_info_sub_topic': camera_info_sub_topic},
                {'color_image_sub_topic': color_image_sub_topic},
                {'depth_image_sub_topic': depth_image_sub_topic},
            ]
        ),

        #Group detection node 
        Node(
            package='rl_detect',
            executable='group_detection_node',
            name='group_detection_node',
            output='screen',
            condition=IfCondition(start_group),
            parameters=[
                {'target_class':              'person'},
                {'eps':    1.5},
                {'min_samples':    2},
                {'cluster_axes': 'xz'},
                {'bbox_padding': 0.1}, 
            ]
        ),
        #Trajectory Forecasting node
        Node(
            package='rl_detect',
            executable='forecasting_node',
            name='forecasting_node',
            output='screen',
            condition=IfCondition(start_forecasting),
            parameters=[
                {'traj_model_arch': 'model1'},  # Options: 'lstm', 'model1', 'model2', 'model3'
                {'traj_model_ckpt': checkpoint_path}, 
                {'obs_len':         8},
                {'pred_len':        12},
                {'obs_interval':    0.4},
                {'video_fps':       30.0},
                {'traj_samples':    1},
                {'noise_type':      'global'},
                {'fixed_noise':     False},
            ]
        ),
        Node(
            package='rl_detect',
            executable='face_recognition_node',
            name='face_recognition_node',
            output='screen',
            condition=IfCondition(start_face),
            parameters=[{
                'det_weight':        det_checkpoint_path,
                'rec_weight':        rec_checkpoint_path,
                'faces_dir':         faces_dir,
                'db_path':           db_path,
                'similarity_thresh': 0.35,
                'confidence_thresh': 0.3,
                'update_db':         'False',
                'publish_annotated': 'True',
                'sync_slop':         0.05,
                'camera_info_sub_topic': camera_info_sub_topic,
                'color_image_sub_topic': color_image_sub_topic,
                'depth_image_sub_topic': depth_image_sub_topic,
            }],
        )
    ])
    
#white box,red box,bottle,whiteboard,headphones,dartboard
