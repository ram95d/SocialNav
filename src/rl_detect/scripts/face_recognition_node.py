import os
import cv2
import random
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer
from std_msgs.msg import Header
# from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from vision_msgs.msg import Detection3DArray, Detection3D, ObjectHypothesisWithPose
from rl_detect.database import FaceDatabase
from rl_detect.models import SCRFD, ArcFace
from rl_detect.helpers import compute_similarity, draw_bbox_info, draw_bbox
# import pyrealsense2 as rs
from image_geometry import PinholeCameraModel

class FaceRecognitionNode(Node):
    def __init__(self):
        super().__init__('face_recognition_node')

        #Parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('det_weight',        'weights/det_10g.onnx'),
                ('rec_weight',        'weights/w600k_mbf.onnx'),
                ('faces_dir',         'assets/faces'),
                ('db_path',           'database/face_database'),
                ('similarity_thresh', 0.3),
                ('confidence_thresh', 0.3),
                ('max_num',           1),
                ('update_db',         'False'),
                ('sync_slop',         0.05),
                ('depth_patch_size',  5),    
                ('min_depth_m',       0.3),     
                ('max_depth_m',       5.0),     
                ('publish_annotated', 'True'),
                ('camera_info_sub_topic', '/oak/stereo/camera_info'),
                ('color_image_sub_topic', '/oak/rgb/image_raw'),
                ('depth_image_sub_topic', '/oak/stereo/image_raw')
            ]
        )

        det_weight        = self.get_parameter('det_weight').value
        rec_weight        = self.get_parameter('rec_weight').value
        faces_dir         = self.get_parameter('faces_dir').value
        db_path           = self.get_parameter('db_path').value
        self.sim_thresh   = self.get_parameter('similarity_thresh').value
        self.conf_thresh  = self.get_parameter('confidence_thresh').value
        self.max_num      = self.get_parameter('max_num').value
        force_update      = self.get_parameter('update_db').value == 'True'
        self.patch_r      = self.get_parameter('depth_patch_size').value
        self.min_depth    = self.get_parameter('min_depth_m').value
        self.max_depth    = self.get_parameter('max_depth_m').value

        try:
            self.detector   = SCRFD(det_weight, input_size=(640, 640),
                                    conf_thres=self.conf_thresh)
            self.recognizer = ArcFace(rec_weight)
        except Exception as e:
            self.get_logger().fatal(f'Failed to load models: {e}')
            raise

        self.face_db = self._build_face_database(faces_dir, db_path, force_update)
        self.colors: dict[str, tuple] = {}

        self.bridge = CvBridge()
        # self.intr: rs.intrinsics | None = None   # set once in _camera_info_cb
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
        self.cam_model: PinholeCameraModel | None = None
        # self.camera_info_sub = self.create_subscription(
        #     CameraInfo, '/camera/camera/color/camera_info',
        #     self._camera_info_cb, 1)

        self.camera_info_sub = self.create_subscription(
            CameraInfo, self.get_parameter('camera_info_sub_topic').value,
            self._camera_info_cb, 1)
        slop      = self.get_parameter('sync_slop').value
        # color_sub = Subscriber(self, Image, '/camera/camera/color/image_raw')
        # depth_sub = Subscriber(self, Image, '/camera/camera/aligned_depth_to_color/image_raw')
        color_sub = Subscriber(self, Image, self.get_parameter('color_image_sub_topic').value)
        depth_sub = Subscriber(self, Image, self.get_parameter('depth_image_sub_topic').value)
        self.ts   = ApproximateTimeSynchronizer(
            [color_sub, depth_sub], queue_size=10, slop=slop)
        self.ts.registerCallback(self._synced_cb)

        self.pub_det3d     = self.create_publisher(
            Detection3DArray, '/outputs/face_rec_det_3d', 10)
        self.pub_annotated = self.create_publisher(
            Image, '/outputs/face_rec_image_annotated', 10)

        self.get_logger().info('FaceRecognitionNode ready.')

    def _camera_info_cb(self, msg: CameraInfo):

        if self.cam_model is not None:
            return

        self.cam_model = PinholeCameraModel()
        self.cam_model.fromCameraInfo(msg)

        self.get_logger().info(
            f'Camera model initialized: '
            f'{msg.width}×{msg.height}  '
            f'fx={self.cam_model.fx():.1f}  '
            f'fy={self.cam_model.fy():.1f}'
        )
    def _synced_cb(self, color_msg: Image, depth_msg: Image):
        if self.cam_model is None:
            self.get_logger().warn('Waiting for camera intrinsics…')
            return

        color_image = self.bridge.imgmsg_to_cv2(color_msg, 'bgr8')
        depth_image = self.bridge.imgmsg_to_cv2(
            depth_msg, desired_encoding='passthrough')

        # h, w = depth_image.shape[:2]
        # color_image = cv2.resize(color_image, (w, h), interpolation=cv2.INTER_LINEAR)

        header            = Header()
        header.stamp      = color_msg.header.stamp
        header.frame_id   = depth_msg.header.frame_id 

        try:
            annotated, detections = self._process_frame(color_image, depth_image)

            self._publish_detections(header, detections)

            if self.get_parameter('publish_annotated').value == 'True':
                ann_msg        = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
                ann_msg.header = header
                self.pub_annotated.publish(ann_msg)

        except Exception as e:
            import traceback
            self.get_logger().error(
                f'Processing error: {e}\n{traceback.format_exc()}')

    def pixel_to_3d(self, u, v, depth_m):
        """
        Convert pixel coordinates + depth to 3D point.

        Parameters
        ----------
        u, v     : pixel coordinates
        depth_m  : depth in meters
        """

        ray = self.cam_model.projectPixelTo3dRay((u, v))

        ray = np.array(ray, dtype=np.float32)

        # normalize so z = 1
        ray = ray / ray[2]

        point = ray * depth_m

        return point.tolist()
    def _process_frame(self, frame: np.ndarray, depth_image: np.ndarray):
        """
        Run SCRFD + ArcFace on one BGR frame, then deproject each face
        center into 3-D camera-frame coordinates using the aligned depth image.

        Returns
        -------
        annotated  : np.ndarray  – BGR frame with bboxes / names drawn
        detections : list[dict]  – one entry per face:
            bbox_xyxy  (x1,y1,x2,y2)  pixel bounding box
            name       str            recognised identity or 'Unknown'
            score      float          cosine similarity  [0, 1]
            conf       float          SCRFD detection confidence
            point_3d   list|None      [X, Y, Z] metres in camera frame,
                                      or None if depth was invalid
            bbox_3d_size list|None    [w_m, h_m, d_m] metric bbox extents
        """
        annotated  = frame.copy()
        detections = []

        bboxes, kpss = self.detector.detect(frame, 0)

        for bbox, kps in zip(bboxes, kpss):
            *xyxy, conf_score = bbox.astype(np.int32)
            x1, y1, x2, y2   = xyxy

            embedding        = self.recognizer.get_embedding(frame, kps)
            name, similarity = self.face_db.search(embedding, self.sim_thresh)

            point_3d    = None
            bbox_3d_size = None

            cx_px = int(np.clip((x1 + x2) / 2, 0, depth_image.shape[1] - 1))
            cy_px = int(np.clip((y1 + y2) / 2, 0, depth_image.shape[0] - 1))

            r   = self.patch_r
            patch = depth_image[
                max(0, cy_px - r) : min(depth_image.shape[0], cy_px + r + 1),
                max(0, cx_px - r) : min(depth_image.shape[1], cx_px + r + 1),
            ]
            valid = patch[patch > 0]

            if len(valid) > 0:
                depth_m = float(np.median(valid)) * 0.001   # mm → m

                if self.min_depth <= depth_m <= self.max_depth:
                    point_3d = self.pixel_to_3d(
                        cx_px,
                        cy_px,
                        depth_m
                    )
                face_w_m = (
                    (x2 - x1) * depth_m / self.cam_model.fx()
                )

                face_h_m = (
                    (y2 - y1) * depth_m / self.cam_model.fy()
                )
                bbox_3d_size = [face_w_m, face_h_m, face_w_m]

            if name != 'Unknown':
                if name not in self.colors:
                    self.colors[name] = (
                        random.randint(0, 255),
                        random.randint(0, 255),
                        random.randint(0, 255),
                    )
                draw_bbox_info(annotated, xyxy,
                               similarity=similarity,
                               name=name,
                               color=self.colors[name])

                if point_3d is not None:
                    px, py, pz = point_3d
                    label = f'{pz:.2f}m'
                    cv2.putText(annotated, label,
                                (x1, y2 + 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                self.colors[name], 2, cv2.LINE_AA)
            else:
                draw_bbox(annotated, xyxy, (255, 0, 0))

            detections.append({
                'bbox_xyxy':   (x1, y1, x2, y2),
                'name':        name,
                'score':       float(similarity),
                'conf':        float(conf_score),
                'point_3d':    point_3d,      
                'bbox_3d_size': bbox_3d_size,  
            })

        return annotated, detections


    def _publish_detections(self, header: Header, detections: list):
        """
        Publishes Detection3DArray

        Each Detection3D:
          - id               → person name
          - bbox.center      → 3-D face position in camera optical frame [metres]
          - bbox.size        → metric extents [metres]
          - results[0]
              class_id       → person name
              score          → cosine similarity
              pose.position  → same 3-D face position (redundant, for compatibility)

        Detections with no valid depth are silently skipped.
        """
        msg        = Detection3DArray()
        msg.header = header

        for det in detections:
            if det['point_3d'] is None:
                continue 

            px, py, pz     = [float(v) for v in det['point_3d']]
            bw, bh, bd     = [float(v) for v in det['bbox_3d_size']]

            d        = Detection3D()
            d.header = header
            d.id     = det['name']

            hyp                          = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id      = det['name']
            hyp.hypothesis.score         = det['score']
            hyp.pose.pose.position.x     = px
            hyp.pose.pose.position.y     = py
            hyp.pose.pose.position.z     = pz
            hyp.pose.pose.orientation.w  = 1.0
            d.results.append(hyp)

            d.bbox.center.position.x    = px
            d.bbox.center.position.y    = py
            d.bbox.center.position.z    = pz
            d.bbox.center.orientation.w = 1.0
            d.bbox.size.x               = bw
            d.bbox.size.y               = bh
            d.bbox.size.z               = bd

            msg.detections.append(d)

        self.pub_det3d.publish(msg)

    def _build_face_database(self, faces_dir: str,
                             db_path: str,
                             force_update: bool) -> FaceDatabase:
        face_db = FaceDatabase(db_path=db_path)

        if not force_update and face_db.load():
            self.get_logger().info('Loaded face database from disk.')
            return face_db

        self.get_logger().info('Building face database from images…')

        if not os.path.isdir(faces_dir):
            self.get_logger().error(f'faces_dir not found: {faces_dir}')
            return face_db

        for person_name in os.listdir(faces_dir):
            person_dir = os.path.join(faces_dir, person_name)
            if not os.path.isdir(person_dir):
                continue

            embeddings = []
            for filename in os.listdir(person_dir):
                if not filename.lower().endswith(('.jpg', '.png')):
                    continue

                image = cv2.imread(os.path.join(person_dir, filename))
                if image is None:
                    continue

                bboxes, kpss = self.detector.detect(image, max_num=1)
                if len(kpss) == 0:
                    self.get_logger().warn(
                        f'No face detected in {filename} — skipping.')
                    continue

                embeddings.append(
                    self.recognizer.get_embedding(image, kpss[0]))
                self.get_logger().debug(
                    f'Enrolled {filename} → {person_name}')

            if embeddings:
                face_db.add_face(np.mean(embeddings, axis=0), person_name)
                self.get_logger().info(
                    f'Enrolled "{person_name}"  ({len(embeddings)} images)')

        face_db.save()
        return face_db



def main(args=None):
    rclpy.init(args=args)
    node = FaceRecognitionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()