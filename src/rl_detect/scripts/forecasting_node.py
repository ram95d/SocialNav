#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection3DArray
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
import torch
import numpy as np
from collections import defaultdict
import math
from geometry_msgs.msg import Point
from rl_detect import forecast
from rl_detect.model.rnn_model import RNNLitModule
from rl_detect.model.model1 import Model1LitModule
from rl_detect.model.model2 import Model2LitModule
from rl_detect.model.model3 import Model3LitModule
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from builtin_interfaces.msg import Duration
COLORS = [
    (0, 102, 255), (0, 183, 255), (0, 255, 255), (0, 255, 183),
    (0, 255, 0), (255, 255, 0), (255, 174, 0), (255, 85, 0),
    (255, 0, 89), (255, 0, 174), (255, 0, 255)
]

class TrajectoryForecastingNode(Node):
    def __init__(self):
        super().__init__('trajectory_forecasting_node')
        
        # Parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('target_class', 'person'),
                ('traj_model_arch', 'lstm'),
                ('traj_model_ckpt', ''),
                ('obs_len', 8),
                ('pred_len', 12),
                ('obs_interval', 0.4),      # seconds
                ('video_fps', 30.0),        # Tracker FPS
                ('traj_samples', 1),
                ('noise_type', 'global'),
                ('fixed_noise', False)
            ]
        )

        self.obs_len = self.get_parameter('obs_len').value
        self.pred_len = self.get_parameter('pred_len').value
        self.obs_interval = self.get_parameter('obs_interval').value
        self.video_fps = self.get_parameter('video_fps').value
        self.target_class = self.get_parameter('target_class').value        
        self.max_track_len = math.ceil(self.obs_len * self.obs_interval * self.video_fps)

        self.track_hist = defaultdict(list)
        self.last_seen = defaultdict(int)
        self.last_y_heights = {} 

        # Load Model
        self.model = self.load_traj_model()
        if self.model is None:
            self.get_logger().error("Failed to load trajectory model. Check architecture/checkpoint.")
            raise RuntimeError("Model initialization failed")
            
        self.model.eval()

        # Subscribers & Publishers
        self.sub_tracking = self.create_subscription(
            Detection3DArray, '/outputs/tracking_3d', self.tracking_callback, 10)
        self.pub_forecast = self.create_publisher(
            MarkerArray, '/outputs/forecasted_trajectories', 10)
        self.get_logger().info("Trajectory Forecasting Node Initialized.")
        self.path_publishers = {}
    def _get_path_publisher(self, track_id: int):
        if track_id not in self.path_publishers:
            topic = f'/outputs/forecasted_path/agent_{track_id}'
            self.path_publishers[track_id] = self.create_publisher(
                Path, topic, 10
            )
            self.get_logger().info(f"Created path publisher for agent {track_id}")
        return self.path_publishers[track_id]
    def load_traj_model(self):
        arch = self.get_parameter('traj_model_arch').value
        ckpt = self.get_parameter('traj_model_ckpt').value
        
        if not arch or not ckpt:
            return None

        model_dict = {
            'lstm': RNNLitModule,
            'model1': Model1LitModule,
            'model2': Model2LitModule,
            'model3': Model3LitModule
        }
        
        if arch not in model_dict:
            raise ValueError(f'Unknown trajectory model architecture: {arch}')

        ModelClass = model_dict[arch]
        self.get_logger().info(f"Loading {arch} from {ckpt}...")
        return ModelClass.load_from_checkpoint(ckpt)


    def tracking_callback(self, msg: Detection3DArray):
        if not msg.detections:
            return

        new_track_ids = []
        new_floor_coords = []

        for det in msg.detections:
            if not det.results:
                continue
                
            if det.results[0].hypothesis.class_id != self.target_class:
                continue

            try:
                track_id = int(det.id)
            except ValueError:
                continue
                
            pos = det.bbox.center.position
            new_track_ids.append(track_id)
            new_floor_coords.append(torch.tensor([pos.x, pos.z], dtype=torch.float32))
            
            # self.last_y_heights[track_id] = pos.y
            self.last_y_heights[track_id] = pos.y + det.bbox.size.y / 2.0
        # If no people were found after filtering, just return
        if not new_track_ids:
            return

        self.update_tracks(new_track_ids, new_floor_coords)

        # Need at least one track to build a scene
        if len(self.track_hist) == 0:
            return

        # scene shape: (n, seq_len, 2)
        scene, agent_ids = self.build_scene()

        # Generate dummy BEV image tensor (CHW) if your model expects it.
        # Note: If model actually extracts CNN features from the map, you will need 
        # to subscribe to an occupancy grid or camera frame here instead.
        dummy_bev_img = torch.zeros((3, 256, 256), dtype=torch.uint8)

        # Forecast
        res = self.trajectory_forecast(scene, dummy_bev_img, agent_ids)
        
        if res is not None:
            preds_scene_KSP2, used_obs_SO2 = res
            self.publish_forecast_markers(preds_scene_KSP2, agent_ids.tolist(), msg.header)


    def update_tracks(self, new_track_ids, new_floor_coords):
        # Update seen tracks
        for track_id, xz in zip(new_track_ids, new_floor_coords):
            self.track_hist[track_id].append(xz)
            self.last_seen[track_id] = 0

        for track_id in list(self.track_hist.keys()):
            self.last_seen[track_id] += 1
            if self.last_seen[track_id] > 1:
                self.track_hist[track_id].append(torch.tensor([np.nan, np.nan]))
            
            if len(self.track_hist[track_id]) > self.max_track_len:
                self.track_hist[track_id].pop(0)

        for track_id, last_seen_count in list(self.last_seen.items()):
            if last_seen_count > self.max_track_len:
                self.track_hist.pop(track_id)
                self.last_seen.pop(track_id)
                self.last_y_heights.pop(track_id, None)


    def build_scene(self):
        scene = torch.nn.utils.rnn.pad_sequence(
            [torch.stack(self.track_hist[track_id][::-1]) for track_id in self.track_hist],
            batch_first=True,
            padding_value=np.nan
        ).flip([1])

        agent_ids = [int(tid) for tid in self.track_hist.keys()]
        return scene, torch.tensor(agent_ids, dtype=torch.long)


    def trajectory_forecast(self, scene_bev_meters, bev_frame_CHW, agent_ids):
        # Detections are already in meters, pass directly to predict
        try:
            res = forecast.predict(
                self.model,
                scene_bev_meters,
                bev_frame_CHW=bev_frame_CHW,
                agent_ids=agent_ids,
                scene_fps=self.video_fps,
                obs_len=self.obs_len,
                pred_len=self.pred_len,
                num_samples=self.get_parameter('traj_samples').value,
                noise_type=self.get_parameter('noise_type').value,
                fixed_noise=self.get_parameter('fixed_noise').value
            )
            return res
        except Exception as e:
            self.get_logger().error(f"Forecasting error: {e}")
            return None


    def publish_forecast_markers(self, preds_scene_KSP2, agent_ids, header):
        """
        Publishes the forecasted paths as LineStrips in RViz.
        preds_scene_KSP2 shape: (num_samples, num_agents, pred_len, 2)
        """
        marker_array = MarkerArray()
        
        num_samples = preds_scene_KSP2.shape[0]
        num_agents = preds_scene_KSP2.shape[1]

        for a_idx in range(num_agents):
            track_id = agent_ids[a_idx]
            
            if self.last_seen[track_id] > 1:
                continue

            r, g, b = COLORS[track_id % len(COLORS)]
            color = ColorRGBA(r=b/255.0, g=g/255.0, b=r/255.0, a=0.8)

            y_height = self.last_y_heights.get(track_id, 0.0)

            for s_idx in range(num_samples):
                # Shape: (pred_len, 2)
                pred_traj = preds_scene_KSP2[s_idx, a_idx]
                
                # Check for NaNs
                if torch.isnan(pred_traj).any():
                    continue

                marker = Marker()
                marker.lifetime = Duration(sec=2, nanosec=0)
                marker.header = header
                marker.ns = 'trajectory_forecast'
                # Ensure unique ID per agent per sample
                marker.id = int(track_id * 1000 + s_idx)
                marker.type = Marker.LINE_STRIP
                marker.action = Marker.ADD
                
                marker.scale.x = 0.05  
                marker.color = color
                path_msg = Path()
                path_msg.header = header
                for step in range(pred_traj.shape[0]):
                    pt = pred_traj[step]
                    p = Point()
                    p.x = float(pt[0])
                    p.y = float(y_height) # Draw the line at the object's vertical center
                    p.z = float(pt[1])
                    marker.points.append(p)
                    ps = PoseStamped()
                    ps.header = header
                    ps.pose.position.x = float(pred_traj[step, 0])
                    ps.pose.position.y = float(y_height)
                    ps.pose.position.z = float(pred_traj[step, 1])
                    path_msg.poses.append(ps)
                marker_array.markers.append(marker)
                self._get_path_publisher(track_id).publish(path_msg)
        self.pub_forecast.publish(marker_array)
        # self.pub_path.publish(path_msg)
    def _remove_track(self, track_id: int):
        self.track_hist.pop(track_id, None)
        self.last_seen.pop(track_id, None)
        self.last_y_heights.pop(track_id, None)

        if track_id in self.path_publishers:
            self.destroy_publisher(self.path_publishers.pop(track_id))
            self.get_logger().info(f"Destroyed path publisher for agent {track_id}")
def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryForecastingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()