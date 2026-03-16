import os
import cv2
import json
import numpy as np
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points

class NuScenesTrackerVisualizer:
    
    def __init__(self, nuscenes_dir, version='v1.0-mini', camera_name='CAM_FRONT'):
        self.nusc = NuScenes(version=version, dataroot=nuscenes_dir, verbose=True)
        self.camera_name = camera_name
        self.camera_sensor_token = self._get_camera_sensor_token()
        print(f"初始化 - 相机: {self.camera_name}, 传感器Token: {self.camera_sensor_token}")
        
        self.fixed_ego_pose = None
        self.fixed_calib = None
        self.fixed_intrinsic = None
        self.history_bboxes = []  
        self.canvas_size = (1600, 900) 

    def _get_camera_sensor_token(self):
        for sensor in self.nusc.sensor:
            if sensor['channel'] == self.camera_name and sensor['modality'] == 'camera':
                return sensor['token']
        raise ValueError(f"未找到相机: {self.camera_name}")
        
    def get_instance_frames_over_time(self, instance_token):
        annotation_tokens = self.nusc.field2token('sample_annotation', 'instance_token', instance_token)
        frames = []
        
        for ann_token in annotation_tokens:
            annotation = self.nusc.get('sample_annotation', ann_token)
            sample = self.nusc.get('sample', annotation['sample_token'])
            
            camera_data_token = None
            for sd_key, sd_token in sample['data'].items():
                sd = self.nusc.get('sample_data', sd_token)
                calib = self.nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
                sensor = self.nusc.get('sensor', calib['sensor_token'])
                if sensor['token'] == self.camera_sensor_token:
                    camera_data_token = sd_token
                    break
                    
            if camera_data_token:
                frames.append({
                    'sample_token': sample['token'],
                    'annotation_token': ann_token,
                    'camera_data_token': camera_data_token,
                    'timestamp': sample['timestamp'],
                    'translation': annotation['translation'],
                    'rotation': annotation['rotation'],
                    'size': annotation['size']  
                })
                
        frames.sort(key=lambda x: x['timestamp'])
        return frames

    def is_moving_vehicle(self, instance_token, min_distance=2.0):
        frames = self.get_instance_frames_over_time(instance_token)
        if len(frames) < 2:
            return False, []
        
        first_pos = np.array(frames[0]['translation'])
        last_pos = np.array(frames[-1]['translation'])
        distance = np.linalg.norm(last_pos - first_pos)
        return distance > min_distance, frames

    def _is_box_in_canvas(self, box):
        if box.center[2] <= 0:  # 过滤相机后方物体
            return False
            
        corners = box.corners()
        points = view_points(corners, self.fixed_intrinsic, normalize=True)[:2, :]
        box_points = points.T.astype(np.int32)
        
        w, h = self.canvas_size
        for (x, y) in box_points:
            if x < 0 or x >= w or y < 0 or y >= h:
                return False
                
        return True

    def render_moving_car(self, instance_token, output_dir=None):
        """以3D边界框序列展示轨迹"""
        frames = self.get_instance_frames_over_time(instance_token)
        if not frames:
            print(f"实例 {instance_token} 无有效帧数据")
            return False
        
        first_frame = frames[0]
        first_sd = self.nusc.get('sample_data', first_frame['camera_data_token'])
        self.fixed_ego_pose = self.nusc.get('ego_pose', first_sd['ego_pose_token'])
        self.fixed_calib = self.nusc.get('calibrated_sensor', first_sd['calibrated_sensor_token'])
        self.fixed_intrinsic = np.array(self.fixed_calib['camera_intrinsic'])
        self.history_bboxes = []

        bg_image_path = os.path.join(self.nusc.dataroot, first_sd['filename'])
        bg_image = cv2.imread(bg_image_path)
        if bg_image is None:
            print(f"无法加载背景图像: {bg_image_path}")
            return False

        cv2.namedWindow('3D Bounding Box Trajectory', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('3D Bounding Box Trajectory', self.canvas_size[0], self.canvas_size[1])

        total_frames = len(frames)
        
        for i, frame in enumerate(frames):
            print(f"处理第 {i+1}/{total_frames} 帧")
            
            sd = self.nusc.get('sample_data', frame['camera_data_token'])
            current_ego = self.nusc.get('ego_pose', sd['ego_pose_token'])
            current_calib = self.nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])

            box = Box(
                frame['translation'],
                frame['size'],
                Quaternion(frame['rotation'])
            )
            box_in_fixed_cam = self._transform_to_fixed_camera(box, current_ego, current_calib)

            if not self._is_box_in_canvas(box_in_fixed_cam):
                print(f"第 {i+1} 帧的3D框超出画布范围，跳过")
                continue

            alpha = 1.0 - (i / total_frames) 
            color = (
                int(255 * (i / total_frames)),  
                255,                            
                0                               
            )

            self.history_bboxes.append({
                'box': box_in_fixed_cam,
                'color': color,
                'alpha': alpha,
                'category': self.nusc.get('sample_annotation', frame['annotation_token'])['category_name']
            })

            current_img = bg_image.copy()
            
            for j, hist_bbox in enumerate(self.history_bboxes[:-1]):
                self._render_3d_box(
                    current_img, 
                    hist_bbox['box'], 
                    hist_bbox['category'], 
                    hist_bbox['color'],
                    alpha=hist_bbox['alpha'] * 0.6  
                )
            
            if self.history_bboxes: 
                current_bbox = self.history_bboxes[-1]
                self._render_3d_box(
                    current_img, 
                    current_bbox['box'], 
                    current_bbox['category'], 
                    (0, 0, 255), 
                    alpha=1.0
                )

            cv2.imshow('3D Bounding Box Trajectory', current_img)
            if cv2.waitKey(200) & 0xFF == ord('q'): 
                break

            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                cv2.imwrite(os.path.join(output_dir, f"frame_{i:04d}.jpg"), current_img)

        cv2.destroyAllWindows()
        return True

    def _transform_to_fixed_camera(self, box, current_ego, current_calib):
        
        # 世界坐标 -> 当前自车坐标
        box.translate(-np.array(current_ego['translation']))
        box.rotate(Quaternion(current_ego['rotation']).inverse)
        
        # 当前自车坐标 -> 世界坐标
        box.rotate(Quaternion(current_ego['rotation']))
        box.translate(np.array(current_ego['translation']))
        
        # 世界坐标 -> 固定自车坐标
        box.translate(-np.array(self.fixed_ego_pose['translation']))
        box.rotate(Quaternion(self.fixed_ego_pose['rotation']).inverse)
        
        # 固定自车坐标 -> 固定相机坐标
        box.translate(-np.array(self.fixed_calib['translation']))
        box.rotate(Quaternion(self.fixed_calib['rotation']).inverse)
        
        return box

    def _render_3d_box(self, image, box, category, color=(0,255,0), alpha=1.0):
        if box.center[2] < 0:  
            return
        
        corners = box.corners()
        points = view_points(corners, self.fixed_intrinsic, normalize=True)[:2, :]
        box_points = points.T.astype(np.int32)
        
        if alpha < 1.0:
            overlay = image.copy()
        else:
            overlay = image

        cv2.polylines(overlay, [box_points[:4]], True, color, 2)  
        cv2.polylines(overlay, [box_points[4:]], True, color, 2)  
        for i in range(4):
            cv2.line(overlay, tuple(box_points[i]), tuple(box_points[i+4]), color, 2)
        
        if alpha == 1.0:
            text_x = int(np.mean(box_points[:, 0]))
            text_y = int(np.min(box_points[:, 1]) - 10)
            cv2.putText(
                overlay, category, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )

        if alpha < 1.0:
            cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    def find_moving_cars(self, min_distance=2.0):
        moving_cars = []
        for instance in self.nusc.instance:
            category = self.nusc.get('category', instance['category_token'])
            if 'vehicle' in category['name']:
                is_moving, frames = self.is_moving_vehicle(instance['token'], min_distance)
                if is_moving and len(frames) > 5:
                    dist = np.linalg.norm(
                        np.array(frames[-1]['translation']) - np.array(frames[0]['translation'])
                    )
                    moving_cars.append((instance, frames, dist))
        
        return sorted(moving_cars, key=lambda x: x[2], reverse=True)

    def generate_json_for_instance(self, instance_token, output_file="3d_bounding_box_data_1.json"):

        frames = self.get_instance_frames_over_time(instance_token)
        if not frames:
            print(f"实例 {instance_token} 无有效帧数据")
            return

        first_frame = frames[0]
        first_sd = self.nusc.get('sample_data', first_frame['camera_data_token'])
        self.fixed_ego_pose = self.nusc.get('ego_pose', first_sd['ego_pose_token'])
        self.fixed_calib = self.nusc.get('calibrated_sensor', first_sd['calibrated_sensor_token'])
        self.fixed_intrinsic = np.array(self.fixed_calib['camera_intrinsic'])

        json_data = {"boxes_3d": []}

        for frame in frames:
            box = Box(
                frame['translation'],
                frame['size'],
                Quaternion(frame['rotation'])
            )
            
            sd = self.nusc.get('sample_data', frame['camera_data_token'])
            current_ego = self.nusc.get('ego_pose', sd['ego_pose_token'])
            current_calib = self.nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
            box_in_fixed_cam = self._transform_to_fixed_camera(box, current_ego, current_calib)
            
            if not self._is_box_in_canvas(box_in_fixed_cam):
                continue
                
            annotation = self.nusc.get('sample_annotation', frame['annotation_token'])
            yaw = Quaternion(annotation['rotation']).yaw_pitch_roll[0]

            box_data = {
                "center": annotation['translation'],
                "wlh": annotation['size'],
                "yaw": yaw,
                "name": annotation['category_name']
            }
            json_data["boxes_3d"].append(box_data)

        with open(output_file, 'w') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        print(f"JSON文件已生成到: {output_file}")


if __name__ == "__main__":
    nuscenes_dir = r'E:\v1.0-mini'  # 数据集路径
    visualizer = NuScenesTrackerVisualizer(nuscenes_dir, camera_name='CAM_FRONT')
    
    moving_cars = visualizer.find_moving_cars(min_distance=1.0)
    
    if moving_cars:
        # 手动修改此处的索引即可切换显示的车辆
        # 有效范围：0 到 len(moving_cars)-1
        target_index = 9
        # 测试中发现当本参数=2或者5等时会有z轴差异，需要考虑问题所在
        
        if 0 <= target_index < len(moving_cars):
            selected_car, _, car_dist = moving_cars[target_index]
            car_category = visualizer.nusc.get('category', selected_car['category_token'])['name']
            print(f"当前显示第 {target_index+1}/{len(moving_cars)} 辆")
            print(f"车辆Token: {selected_car['token']}, 类别: {car_category}, 移动距离: {car_dist:.2f}米")
            
            visualizer.generate_json_for_instance(selected_car['token'])
            
            visualizer.render_moving_car(selected_car['token'], output_dir=f"3d_bbox_trajectory_2_{target_index}")
        else:
            print(f"索引无效！有效范围：0 到 {len(moving_cars)-1}")
    else:
        print("未找到移动车辆")