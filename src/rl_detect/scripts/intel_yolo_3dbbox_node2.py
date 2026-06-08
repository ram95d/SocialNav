import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer
from image_geometry import PinholeCameraModel
from ultralytics import YOLO
import open3d as o3d
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose
from rl_detect.DetectionDrawer import DetectionDrawer
class YOLODetectorNode(Node):
    def __init__(self):
        super().__init__('yolo_detector_node')
        self.declare_parameters(
            namespace='',
            parameters=[
                ('model', ''),
                ('classes', ''),
                ('output_markers', 'True'),
                ('output_pointcloud', 'False'),
                ('output_detection_3d', 'True'),
                ('enable_tracking', 'True'),
                ('sync_slop', 0.05),        # seconds — tighten if hardware allows
                ('forecast_horizon', 30),   # steps for trajectory node
                ('camera_info_sub_topic', '/camera/color/camera_info'),
                ('color_image_sub_topic', '/camera/color/image_raw'),
                ('depth_image_sub_topic', '/camera/aligned_depth_to_color/image_raw'),
            ]
        )

        classes_param = self.get_parameter('classes').get_parameter_value().string_value
        self.classes = [c.strip() for c in classes_param.split(',') if c.strip()]
        self.enable_tracking = self.get_parameter('enable_tracking').value == 'True'

        self.yolo_model = YOLO(self.get_parameter('model').value)
        self.yolo_model.to('cuda')
        self.yolo_model.set_classes(self.classes)
        self.get_logger().info(f"Detecting classes: {self.classes}")

        self.bridge = CvBridge()
        self.cam_model = None
        self.drawer = None

        '''oak topics
        '/oak/stereo/camera_info',
        '/oak/rgb/image_raw',
        '/oak/stereo/image_raw',    
        '''
        '''realsense topics
        '/camera/camera/color/camera_info',
        '/camera/camera/color/image_raw',
        '/camera/camera/aligned_depth_to_color/image_raw',
        '''

        # Camera info
        # self.camera_info_sub = self.create_subscription(
        #     CameraInfo, '/camera/camera/color/camera_info',
        #     self.camera_info_callback, 1)
        self.camera_info_sub = self.create_subscription(
            CameraInfo, self.get_parameter('camera_info_sub_topic').value,
            self.camera_info_callback, 1)
        #synchronized color + depth
        # color_sub = Subscriber(self, Image, '/camera/camera/color/image_raw')
        color_sub = Subscriber(self, Image, self.get_parameter('color_image_sub_topic').value)
        depth_sub = Subscriber(self, Image, self.get_parameter('depth_image_sub_topic').value)
        # depth_sub = Subscriber(self, Image, '/camera/camera/aligned_depth_to_color/image_raw')
        slop = self.get_parameter('sync_slop').value
        self.ts = ApproximateTimeSynchronizer(
            [color_sub, depth_sub], queue_size=10, slop=slop)
        self.ts.registerCallback(self.synced_callback)

        #Publishers
        self.pub_markers   = self.create_publisher(MarkerArray,      '/outputs/yolo_detected_objects_marker_array', 10)
        self.pub_pcd       = self.create_publisher(PointCloud2,      'outputs/segment', 10)
        self.pub_det3d     = self.create_publisher(Detection3DArray, '/outputs/detection_3d', 10)
        self.pub_tracking  = self.create_publisher(Detection3DArray, '/outputs/tracking_3d', 10)

    #Callbacks

    def camera_info_callback(self, msg):
        if self.cam_model is not None:
            return

        self.cam_model = PinholeCameraModel()
        self.cam_model.fromCameraInfo(msg)

        fx = self.cam_model.fx()
        fy = self.cam_model.fy()
        cx = self.cam_model.cx()
        cy = self.cam_model.cy()

        cam_intr = o3d.camera.PinholeCameraIntrinsic(
            msg.width,
            msg.height,
            fx,
            fy,
            cx,
            cy
        )

        extrinsic = [
            [1, 0, 0, 0],
            [0,-1, 0, 0],
            [0, 0,-1, 0],
            [0, 0, 0, 1]
        ]

        self.drawer = DetectionDrawer(
            self.classes,
            cam_intr,
            extrinsic,
            self.cam_model
        )

        self.get_logger().info('Camera model initialized')

    def synced_callback(self, color_msg: Image, depth_msg: Image):
        if self.drawer is None:
            self.get_logger().warn('Waiting for camera info…')
            return

        color_image = self.bridge.imgmsg_to_cv2(color_msg, 'bgr8')
        depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

        # Resize color to match depth resolution
        # h, w = depth_image.shape
        # color_image = cv2.resize(color_image, (w, h), interpolation=cv2.INTER_LINEAR)

        header = Header()
        header.stamp    = color_msg.header.stamp
        header.frame_id = depth_msg.header.frame_id

        try:
            if self.enable_tracking:
                results = self.yolo_model.track(
                    color_image, conf=0.5, iou=0.5, persist=True, verbose=False)
            else:
                results = self.yolo_model(
                    color_image, conf=0.5, iou=0.5, verbose=False)

            result = results[0]
            if result.boxes is None or len(result.boxes) == 0:
                return
            if result.masks is None:
                self.get_logger().warn('No masks — use a *-seg.pt model.')
                return

            scores    = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            masks     = result.masks.data.cpu().numpy()   # [N, H, W]
            track_ids = (result.boxes.id.cpu().numpy().astype(int)
                         if self.enable_tracking and result.boxes.id is not None
                         else np.arange(len(scores)))

            bbox3d, pcd = self.drawer(depth_image, masks, scores, class_ids)

            if self.get_parameter('output_pointcloud').value == 'True':
                self.publish_pointcloud(header, pcd)
            if self.get_parameter('output_markers').value == 'True':
                self.publish_markers(header, bbox3d, class_ids, track_ids)
            if self.get_parameter('output_detection_3d').value == 'True':
                self.publish_3d_detections(header, bbox3d, scores, class_ids)
            if self.enable_tracking:
                self.publish_tracking(header, bbox3d, scores, class_ids, track_ids)

        except Exception as e:
            import traceback
            self.get_logger().error(f'Processing error: {e}\n{traceback.format_exc()}')

    #Publishers

    def publish_tracking(self, header, bbox3d, scores, class_ids, track_ids):
        msg = Detection3DArray()
        msg.header = header
        for i, bbox in enumerate(bbox3d):
            det = Detection3D()
            det.id = str(track_ids[i])

            cx, cy, cz = [float(v) for v in bbox.get_center()]
            ex, ey, ez = [float(v) for v in bbox.get_extent()]

            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = self.classes[class_ids[i]]
            hyp.hypothesis.score    = float(scores[i])
            hyp.pose.pose.position.x = cx
            hyp.pose.pose.position.y = cy
            hyp.pose.pose.position.z = cz
            hyp.pose.pose.orientation.w = 1.0
            det.results.append(hyp)

            det.bbox.center.position.x = cx
            det.bbox.center.position.y = cy
            det.bbox.center.position.z = cz
            det.bbox.center.orientation.w = 1.0
            det.bbox.size.x = ex
            det.bbox.size.y = ey
            det.bbox.size.z = ez
            msg.detections.append(det)

        self.pub_tracking.publish(msg)

    def publish_3d_detections(self, header, bbox3d, scores, class_ids):
        msg = Detection3DArray()
        msg.header = header
        for i, bbox in enumerate(bbox3d):
            det = Detection3D()
            cx, cy, cz = [float(v) for v in bbox.get_center()]
            ex, ey, ez = [float(v) for v in bbox.get_extent()]

            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = self.classes[class_ids[i]]
            hyp.hypothesis.score    = float(scores[i])
            hyp.pose.pose.position.x = cx
            hyp.pose.pose.position.y = cy
            hyp.pose.pose.position.z = cz
            hyp.pose.pose.orientation.w = 1.0
            det.results.append(hyp)

            det.bbox.center.position.x = cx
            det.bbox.center.position.y = cy
            det.bbox.center.position.z = cz
            det.bbox.center.orientation.w = 1.0
            det.bbox.size.x = ex; det.bbox.size.y = ey; det.bbox.size.z = ez
            msg.detections.append(det)
        self.pub_det3d.publish(msg)

    def publish_markers(self, header, bbox3d, class_ids, track_ids=None):
        marker_array = MarkerArray()
        for i, bbox in enumerate(bbox3d):
            m = Marker()
            m.header = header
            m.ns     = 'yolo_detected_objects'
            m.id     = int(track_ids[i]) if track_ids is not None else i
            m.type   = Marker.CUBE
            m.action = Marker.ADD
            cx, cy, cz = bbox.get_center()
            ex, ey, ez = bbox.get_extent()
            m.pose.position.x = float(cx)
            m.pose.position.y = float(cy)
            m.pose.position.z = float(cz)
            m.pose.orientation.w = 1.0
            m.scale.x = float(ex); m.scale.y = float(ey); m.scale.z = float(ez)
            r, g, b = self.get_color_for_class(class_ids[i])
            m.color.r = r; m.color.g = g; m.color.b = b; m.color.a = 0.6
            marker_array.markers.append(m)
        self.pub_markers.publish(marker_array)

    def publish_pointcloud(self, header, pcd):
        fields = [
            PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        rgb    = (np.floor(colors * 255).astype(np.uint8))
        rgb_int = ((rgb[:,0].astype(np.uint32) << 16) |
                   (rgb[:,1].astype(np.uint32) << 8)  |
                    rgb[:,2].astype(np.uint32))
        buf = np.zeros((len(points), 4), dtype=np.float32)
        buf[:, :3] = points
        buf[:,  3] = rgb_int.view(np.float32)

        msg = PointCloud2()
        msg.header    = header
        msg.fields    = fields
        msg.height    = 1
        msg.width     = len(points)
        msg.point_step = 16
        msg.row_step   = msg.point_step * msg.width
        msg.is_dense   = bool(np.isfinite(points).all())
        msg.data       = buf.tobytes()
        self.pub_pcd.publish(msg)

    def get_color_for_class(self, class_id):
        palette = [
            (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
            (1.0, 1.0, 0.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0),
        ]
        return palette[class_id % len(palette)]


def main(args=None):
    rclpy.init(args=args)
    node = YOLODetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()