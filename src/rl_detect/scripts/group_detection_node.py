import json
import numpy as np
from sklearn.cluster import DBSCAN
import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection3DArray
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import String, Header
from geometry_msgs.msg import Point
from builtin_interfaces.msg import Duration

class GroupDetectionNode(Node):
    def __init__(self):
        super().__init__('group_detection_node')
        self.declare_parameters(
            namespace='',
            parameters=[
                ('target_class',   'person'),
                # DBSCAN parameters
                ('eps',            1.5),   # max distance (meters) between two people to be in same group
                ('min_samples',    1),     # min people to form a core point (1 = no noise points)
                # Which axes to cluster on: 'xz' = ground plane only, 'xyz' = full 3D
                ('cluster_axes',   'xz'),
                ('bbox_padding',   0.1),
            ]
        )

        self.target_class  = self.get_parameter('target_class').value
        self.eps           = self.get_parameter('eps').value
        self.min_samples   = self.get_parameter('min_samples').value
        self.cluster_axes  = self.get_parameter('cluster_axes').value
        self.bbox_padding  = self.get_parameter('bbox_padding').value

        self._color_cache: dict = {}

        self.sub = self.create_subscription(
            Detection3DArray, '/outputs/tracking_3d',
            self.tracking_callback, 10)

        self.pub_markers = self.create_publisher(
            MarkerArray, '/outputs/group_det_markers', 10)
        self.pub_data = self.create_publisher(
            String, '/outputs/group_det_data', 10)

        self.get_logger().info(
            f'GroupDetectionNode ready | eps={self.eps}m '
            f'min_samples={self.min_samples} axes={self.cluster_axes}')


    def _group_color(self, group_id: int):
        """Deterministic color per group id."""
        if group_id not in self._color_cache:
            rng = np.random.default_rng(seed=group_id + 42)
            self._color_cache[group_id] = tuple(rng.random(3).tolist())
        return self._color_cache[group_id]


    def tracking_callback(self, msg: Detection3DArray):
        # Extract persons
        people = []
        for det in msg.detections:
            if not det.results:
                continue
            if det.results[0].hypothesis.class_id != self.target_class:
                continue
            people.append({
                'track_id': det.id,
                'center':   np.array([
                    det.bbox.center.position.x,
                    det.bbox.center.position.y,
                    det.bbox.center.position.z,
                ]),
                'size': np.array([
                    det.bbox.size.x,
                    det.bbox.size.y,
                    det.bbox.size.z,
                ]),
                'score': det.results[0].hypothesis.score,
            })

        marker_array = MarkerArray()

        clear = Marker()
        clear.header = msg.header
        clear.action = Marker.DELETEALL
        marker_array.markers.append(clear)

        if not people:
            self.pub_markers.publish(marker_array)
            empty = String()
            empty.data = json.dumps({'groups': {}, 'num_groups': 0, 'noise': []})
            self.pub_data.publish(empty)
            return

        # DBSCAN
        centers = np.array([p['center'] for p in people])

        if self.cluster_axes == 'xz':
            features = centers[:, [0, 2]]   # ground plane (x, depth)
        elif self.cluster_axes == 'xy':
            features = centers[:, [0, 1]]
        else:
            features = centers              # full 3D

        labels = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric='euclidean'
        ).fit_predict(features)
        # labels == -1 → noise (lone individuals when min_samples > 1)
        group_centroids = {}
        for idx, label in enumerate(labels):
            if label == -1:
                continue
            group_centroids.setdefault(label, []).append(features[idx])

        sorted_ids = sorted(group_centroids.keys(),
                            key=lambda g: np.mean(group_centroids[g], axis=0)[0])
        label_remap = {old: new for new, old in enumerate(sorted_ids)}

        labels = np.array([
            label_remap[l] if l != -1 else -1
            for l in labels
        ])
        groups: dict = {} 
        noise:  list = []

        for idx, label in enumerate(labels):
            if label == -1:
                noise.append(people[idx])
            else:
                groups.setdefault(label, []).append(people[idx])

        self._build_markers(msg.header, groups, noise, marker_array)
        self.pub_markers.publish(marker_array)

        output = {
            'num_groups': len(groups),
            'groups': {
                str(gid): {
                    'size':    len(members),
                    'members': [
                        {
                            'track_id': p['track_id'],
                            'center':   p['center'].tolist(),
                            'size':     p['size'].tolist(),
                        }
                        for p in members
                    ],
                    'group_bbox': self._compute_group_bbox(members),
                }
                for gid, members in groups.items()
            },
            'noise': [
                {'track_id': p['track_id'], 'center': p['center'].tolist()}
                for p in noise
            ],
        }
        out_msg = String()
        out_msg.data = json.dumps(output)
        self.pub_data.publish(out_msg)

    def _compute_group_bbox(self, members: list) -> dict:
        """
        Axis-aligned bounding box that encloses all people in a group.
        Uses each person's center ± half their individual size.
        """
        pad = self.bbox_padding
        mins, maxs = [], []
        for p in members:
            c, s = p['center'], p['size']
            mins.append(c - s / 2 - pad)
            maxs.append(c + s / 2 + pad)

        mn  = np.min(mins, axis=0)
        mx  = np.max(maxs, axis=0)
        ctr = (mn + mx) / 2
        sz  = mx - mn
        return {
            'center': ctr.tolist(),
            'size':   sz.tolist(),
        }


    def _build_markers(self, header: Header, groups: dict,
                       noise: list, marker_array: MarkerArray):
        marker_id = 0

        for gid, members in groups.items():
            r, g, b = self._group_color(gid)

            for person in members:
                m = Marker()
                m.lifetime = Duration(sec=2, nanosec=0)
                m.header  = header
                m.ns      = 'person_boxes'
                m.id      = marker_id; marker_id += 1
                m.type    = Marker.CUBE
                m.action  = Marker.ADD
                c, s = person['center'], person['size']
                m.pose.position.x = float(c[0])
                m.pose.position.y = float(c[1])
                m.pose.position.z = float(c[2])
                m.pose.orientation.w = 1.0
                m.scale.x = float(s[0]); m.scale.y = float(s[1]); m.scale.z = float(s[2])
                m.color.r = r; m.color.g = g; m.color.b = b; m.color.a = 0.5
                m.lifetime = Duration(sec=1)
                marker_array.markers.append(m)

                # Person label
                lbl = Marker()
                lbl.header = header; lbl.ns = 'person_labels'; lbl.id = marker_id; marker_id += 1
                lbl.type   = Marker.TEXT_VIEW_FACING; lbl.action = Marker.ADD
                lbl.pose.position.x = float(c[0])
                lbl.pose.position.y = float(c[1])
                lbl.pose.position.z = float(c[2]) + float(s[1]) / 2 + 0.15
                lbl.scale.z = 0.18
                lbl.color.r = r; lbl.color.g = g; lbl.color.b = b; lbl.color.a = 1.0
                lbl.text    = f'G{gid} | ID:{person["track_id"]}'
                lbl.lifetime = Duration(sec=1)
                marker_array.markers.append(lbl)

            gb   = self._compute_group_bbox(members)
            ctr  = np.array(gb['center'])
            sz   = np.array(gb['size']) / 2  # half-extents

            corners = np.array([
                ctr + sz * np.array([sx, sy, sz_])
                for sx in [-1, 1]
                for sy in [-1, 1]
                for sz_ in [-1, 1]
            ])

            edges = [
                (0,1),(2,3),(4,5),(6,7),  # along z
                (0,2),(1,3),(4,6),(5,7),  # along y
                (0,4),(1,5),(2,6),(3,7),  # along x
            ]
            wire = Marker()
            wire.header   = header; wire.ns = 'group_bbox_wire'; wire.id = marker_id; marker_id += 1
            wire.type     = Marker.LINE_LIST; wire.action = Marker.ADD
            wire.scale.x  = 0.04   # line width
            wire.color.r  = r; wire.color.g = g; wire.color.b = b; wire.color.a = 1.0
            wire.lifetime = Duration(sec=1)
            for (i, j) in edges:
                p1 = Point(); p1.x, p1.y, p1.z = [float(v) for v in corners[i]]
                p2 = Point(); p2.x, p2.y, p2.z = [float(v) for v in corners[j]]
                wire.points.extend([p1, p2])
            marker_array.markers.append(wire)

            # 3. Group label at top-center of group bbox
            glbl = Marker()
            glbl.header = header; glbl.ns = 'group_labels'; glbl.id = marker_id; marker_id += 1
            glbl.type   = Marker.TEXT_VIEW_FACING; glbl.action = Marker.ADD
            glbl.pose.position.x = float(ctr[0])
            glbl.pose.position.y = float(ctr[1])
            glbl.pose.position.z = float(ctr[2]) + float(sz[1]) + 0.25
            glbl.scale.z  = 0.22
            glbl.color.r  = r; glbl.color.g = g; glbl.color.b = b; glbl.color.a = 1.0
            glbl.text     = f'Group {gid}  ({len(members)} persons)'
            glbl.lifetime = Duration(sec=1)
            marker_array.markers.append(glbl)

            # Connection lines between group members
            # for i in range(len(members)):
            #     for j in range(i + 1, len(members)):
            #         line = Marker()
            #         line.header  = header; line.ns = 'group_lines'; line.id = marker_id; marker_id += 1
            #         line.type    = Marker.LINE_LIST; line.action = Marker.ADD
            #         line.scale.x = 0.025
            #         line.color.r = r; line.color.g = g; line.color.b = b; line.color.a = 0.6
            #         line.lifetime = Duration(sec=1)
            #         ci = members[i]['center']; cj = members[j]['center']
            #         p1 = Point(); p1.x, p1.y, p1.z = float(ci[0]), float(ci[1]), float(ci[2])
            #         p2 = Point(); p2.x, p2.y, p2.z = float(cj[0]), float(cj[1]), float(cj[2])
            #         line.points.extend([p1, p2])
            #         marker_array.markers.append(line)

        # Noise / lone individuals (grey)
        # for person in noise:
        #     m = Marker()
        #     m.header  = header; m.ns = 'noise_boxes'; m.id = marker_id; marker_id += 1
        #     m.type    = Marker.CUBE; m.action = Marker.ADD
        #     c, s = person['center'], person['size']
        #     m.pose.position.x = float(c[0])
        #     m.pose.position.y = float(c[1])
        #     m.pose.position.z = float(c[2])
        #     m.pose.orientation.w = 1.0
        #     m.scale.x = float(s[0]); m.scale.y = float(s[1]); m.scale.z = float(s[2])
        #     m.color.r = 0.6; m.color.g = 0.6; m.color.b = 0.6; m.color.a = 0.4
        #     m.lifetime = Duration(sec=1)
        #     marker_array.markers.append(m)


def main(args=None):
    rclpy.init(args=args)
    node = GroupDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()