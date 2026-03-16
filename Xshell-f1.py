import numpy as np
import json
import cv2
import torch
from PIL import Image
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
from pytorch3d.ops import box3d_overlap
import os

    
def box_to_vertices(box):

    center = np.array(box['center'])
    wlh = np.array(box['wlh'])
    yaw = box['yaw']

    rot_matrix = Quaternion(axis=[0, 0, 1], angle=yaw).rotation_matrix
    
    vertices_rel = np.array([
        [-wlh[0]/2, -wlh[1]/2, -wlh[2]/2],
        [ wlh[0]/2, -wlh[1]/2, -wlh[2]/2],
        [ wlh[0]/2,  wlh[1]/2, -wlh[2]/2],
        [-wlh[0]/2,  wlh[1]/2, -wlh[2]/2],
        [-wlh[0]/2, -wlh[1]/2,  wlh[2]/2],
        [ wlh[0]/2, -wlh[1]/2,  wlh[2]/2],
        [ wlh[0]/2,  wlh[1]/2,  wlh[2]/2],
        [-wlh[0]/2,  wlh[1]/2,  wlh[2]/2]
    ])
    
    vertices_rot = (rot_matrix @ vertices_rel.T).T
    
    vertices = vertices_rot + center
    
    return vertices


def quaternion_to_yaw(rotation):

    q = Quaternion(rotation)
    return q.yaw_pitch_roll[0]


def get_matrix(data, inverse=False):

    transform_matrix = np.eye(4)
    
    transform_matrix[:3, :3] = Quaternion(data['rotation']).rotation_matrix
    
    transform_matrix[:3, 3] = data['translation']
    
    if inverse:
        transform_matrix = np.linalg.inv(transform_matrix)
    
    return transform_matrix


def transform_box_to_camera(box, nuscenes, sample_token, camera_name='CAM_FRONT'):

    camera_data = nuscenes.get('sample_data', nuscenes.get('sample', sample_token)['data'][camera_name])
    
    camera_ego_pose = nuscenes.get("ego_pose", camera_data["ego_pose_token"])
    
    camera_calibrated_data = nuscenes.get("calibrated_sensor", camera_data["calibrated_sensor_token"])
    
    camera_params = {
        'ego_pose': camera_ego_pose,
        'calibrated_data': camera_calibrated_data,
        'intrinsic': camera_calibrated_data["camera_intrinsic"]
    }
    
    box_transform = np.eye(4)
    box_transform[:3, :3] = Quaternion(axis=[0, 0, 1], angle=box['yaw']).rotation_matrix
    box_transform[:3, 3] = box['center']
    
    global_to_ego = get_matrix(camera_ego_pose, True)
    
    ego_to_camera = get_matrix(camera_calibrated_data, True)
    
    global_to_camera = ego_to_camera @ global_to_ego
    
    camera_box_transform = global_to_camera @ box_transform
    
    camera_center = camera_box_transform[:3, 3]
    camera_yaw = Quaternion(matrix=camera_box_transform[:3, :3]).yaw_pitch_roll[0]
    
    camera_transformed_box = {
        'center': camera_center.tolist(),
        'wlh': box['wlh'],
        'yaw': camera_yaw
    }
    
    return camera_transformed_box, camera_params


def calculate_angle_diff(target_box, nuscenes_box):

    angle_diff = abs(target_box['yaw'] - nuscenes_box['yaw'])
    
    angle_similarity = 1- angle_diff / np.pi 
    
    return angle_similarity




def extract_3d_boxes_from_nuscenes(nusc, processed_samples=None):

    boxes = []
    sample_tokens = []
    box_tokens = []
    
    if processed_samples is None:
        processed_samples = []
    
    for sample in nusc.sample:
        sample_token = sample['token']
        
        if sample_token in processed_samples:
            continue

        scene = nusc.get('scene', sample['scene_token'])
        scene_desc = scene.get('description', '').lower()
        

        exclude_keywords = ['night'] 
        if any(keyword in scene_desc for keyword in exclude_keywords):
            continue  # 跳过该场景下的所有样本


        for ann_token in sample['anns']:
            ann = nusc.get('sample_annotation', ann_token)
            
            # 只处理car类别
            if ann['category_name'] == 'vehicle.car':
                visibility_token = ann['visibility_token']
                visibility = nusc.get('visibility', visibility_token)
                
                cam_front_visibility = visibility['level']
                
                if cam_front_visibility in ['v80-100', 'v40-60']:  

                    box = {
                        'center': ann['translation'],
                        'wlh': ann['size'],
                        'yaw': Quaternion(ann['rotation']).yaw_pitch_roll[0],
                        'name': ann['category_name'],
                        'visibility': cam_front_visibility
                    }
                    
                    boxes.append(box)
                    sample_tokens.append(sample_token)
                    box_tokens.append(ann_token)
    
    return boxes, sample_tokens, box_tokens


def project_3d_box_to_2d_mask(box, nuscenes, sample_token, camera_name='CAM_FRONT'):

    sample_data = nuscenes.get('sample', sample_token)
    camera_data = nuscenes.get('sample_data', sample_data['data'][camera_name])
    
    camera_ego_pose = nuscenes.get("ego_pose", camera_data["ego_pose_token"])
    
    global_to_ego = get_matrix(camera_ego_pose, True)
    
    camera_calibrated_data = nuscenes.get("calibrated_sensor", camera_data["calibrated_sensor_token"])
    
    ego_to_camera = get_matrix(camera_calibrated_data, True)
    
    camera_intrinsic = np.eye(4)
    camera_intrinsic[:3, :3] = camera_calibrated_data["camera_intrinsic"]
    
    # global->camera_ego_pose->camera->image
    global_to_image = camera_intrinsic @ ego_to_camera @ global_to_ego
    

    global_points = box.corners().T 
    
    global_points_hom = np.concatenate([global_points, np.ones((global_points.shape[0], 1))], axis=1)
    
    image_points = global_points_hom @ global_to_image.T
    
    image_points[:, :2] /= image_points[:, [2]]
    
    img_path = nuscenes.get_sample_data_path(camera_data['token'])
    image = cv2.imread(img_path)
    if image is None:
        print(f"无法加载图像: {img_path}")
        return None, None
    image_shape = image.shape[:2]
    
    # z > 0
    valid_points = image_points[image_points[:, 2] > 0]
    valid_points_2d = valid_points[:, :2].astype(int)
    
    out_of_bounds_count = 0
    total_points = len(valid_points_2d)
    
    for point in valid_points_2d:
        x, y = point[0], point[1]
        if x < 0 or x >= image_shape[1] or y < 0 or y >= image_shape[0]:
            out_of_bounds_count += 1

    out_of_bounds_ratio = out_of_bounds_count / total_points if total_points > 0 else 0
    print(f"3D框在{camera_name}视角的出画比例为{out_of_bounds_ratio:.2%}")
    
    if out_of_bounds_ratio > 0.5:
        print(f"3D框在{camera_name}视角的出画比例为{out_of_bounds_ratio:.2%}，超过50%，将跳过渲染")
        return None, None
    
    mask = np.zeros(image_shape, dtype=np.uint8)
    
    for point in valid_points_2d:
        x, y = point[0], point[1]
        if 0 <= x < image_shape[1] and 0 <= y < image_shape[0]:
            cv2.circle(mask, (x, y), 3, 255, -1)
    
    if len(valid_points_2d) > 0:
        x_coords = valid_points_2d[:, 0]
        y_coords = valid_points_2d[:, 1]
        x1 = int(np.clip(np.min(x_coords), 0, image_shape[1]-1))
        y1 = int(np.clip(np.min(y_coords), 0, image_shape[0]-1))
        x2 = int(np.clip(np.max(x_coords), 0, image_shape[1]-1))
        y2 = int(np.clip(np.max(y_coords), 0, image_shape[0]-1))
 
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    
    return mask, image_shape



def main():

    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    traj_dir = os.path.join(script_dir, 'traj')
    
    with open(os.path.join(traj_dir, 'box.json'), 'r') as f:
        all_boxes = json.load(f)
    
    nusc = NuScenes(
        version='v1.0-trainval',
        dataroot=os.path.join(script_dir, '..', 'v1.0-trainval'),  
        verbose=True
    )
    
    # 只处理第8帧（根据box.json中的数据）
    frame_index = 8
    print(f"处理第 {frame_index} 帧")
    
    if frame_index >= len(all_boxes['boxes_3d']):
        print(f"帧索引 {frame_index} 超出范围")
        return
    
    target_box = all_boxes['boxes_3d'][frame_index]
    print(f"目标边界框: {target_box}")
    
    # 使用指定的样本token和边界框token获取ego_pose参数
    sample_token = 'db9921688a634e7e9fc963b9363aa3bb'
    #box_token = '64d522f129634049b4bba059ffd2ff5c'
    
    sample_data = nusc.get('sample', sample_token)
    
    camera_data = nusc.get('sample_data', sample_data['data']['CAM_FRONT'])
    
    target_ego_pose = nusc.get("ego_pose", camera_data["ego_pose_token"])
    
    target_box_camera, target_camera_params = transform_box_to_camera(target_box, nusc, sample_token)
    
    print("\ntarget车:")
    print(f"相机坐标系下：坐标 {target_box_camera['center']}, 角度 {target_box_camera['yaw']:.4f}")
    
    nuscenes_boxes, sample_tokens, box_tokens = extract_3d_boxes_from_nuscenes(nusc)
    print(f"从nuScenes数据集中提取了 {len(nuscenes_boxes)} 个边界框")
    
    if len(nuscenes_boxes) == 0:
        print("没有可用的边界框")
        return
    
    similarities = []
    ious = [] 
    angle_diffs = []  
    size_diffs=[]
    
    for i, (box, sample_tok) in enumerate(zip(nuscenes_boxes, sample_tokens)):
        box_camera, _ = transform_box_to_camera(box, nusc, sample_tok)
        
        target_vertices = box_to_vertices(target_box_camera)
        box_vertices = box_to_vertices(box_camera)
    
        target_vertices = target_vertices[np.newaxis, ...]
        box_vertices = box_vertices[np.newaxis, ...]
    
        target_vertices = torch.from_numpy(target_vertices).float()
        box_vertices = torch.from_numpy(box_vertices).float()
        _, iou = box3d_overlap(target_vertices, box_vertices)

        iou = iou.item() if hasattr(iou, 'item') else float(iou)
        ious.append(iou)
        
        angle_similarity = calculate_angle_diff(target_box_camera, box_camera)
        angle_diffs.append(angle_similarity)

        target_wlh = np.array(target_box['wlh'])
        box_wlh = np.array(box['wlh'])
        size_similarity =  (np.abs(target_wlh - box_wlh) / np.maximum(target_wlh, box_wlh)).mean()
        size_diffs.append(size_similarity)

        # 计算相似性分数
        similarity = iou *0.3  + angle_similarity*0.7 - size_similarity *0.03
        similarities.append(similarity)
    
    similarities = np.array(similarities)
    ious = np.array(ious)
    angle_diffs = np.array(angle_diffs)
    size_diffs=np.array(size_diffs)
    
    top_indices = np.argsort(similarities)[::-1][:10]
    
    print("\n前10名:")

    results = []
    for i, idx in enumerate(top_indices):
        idx = int(idx)
        
        box_camera, camera_params = transform_box_to_camera(nuscenes_boxes[idx], nusc, sample_tokens[idx])
        
        most_similar_sample_token = sample_tokens[idx]
        most_similar_box_token = box_tokens[idx]
        
        similarity_val = float(similarities[idx])
        iou_val = float(ious[idx])
        angle_diff_val = float(angle_diffs[idx])
        size_diff_val=float(size_diffs[idx])
        
        print(f"\n第{i+1}个最相似边界框 (综合相似性: {similarity_val:.4f}):")
        print(f"  3D IoU = {iou_val:.4f}, 角度 为 {angle_diff_val:.4f},尺寸 为{size_diff_val:.4f}")
        print(f"  相机坐标系下：坐标 {box_camera['center']}, 角度 {box_camera['yaw']:.4f}")
        
        print(f"  样本token: {most_similar_sample_token}")
        print(f"  box token: {most_similar_box_token}")
        
        most_similar_sample_data = nusc.get('sample', most_similar_sample_token)
        most_similar_camera_data = nusc.get('sample_data', most_similar_sample_data['data']['CAM_FRONT'])
        front_camera_image_path = nusc.get_sample_data_path(most_similar_camera_data['token'])
        front_camera_image_name = os.path.basename(front_camera_image_path)  
        print(f"  前视相机图片: {front_camera_image_name}")
        
        results.append({
            'rank': i+1,
            'similarity': similarity_val,
            'iou': iou_val,
            'angle_diff': angle_diff_val,
            'size_diff' :size_diff_val,
            'camera_coords': box_camera['center'],
            'camera_yaw': float(box_camera['yaw']),
            'sample_token': most_similar_sample_token,
            'box_token': most_similar_box_token,
            'front_camera_image': front_camera_image_name
        })
        
        
        try:
            nusc_box = nusc.get_box(most_similar_box_token)
            
            bbox_mask, image_shape = project_3d_box_to_2d_mask(nusc_box, nusc, most_similar_sample_token)
            
            if bbox_mask is not None:
                cv2.imwrite(f'm_{frame_index}_{i+1}.png', bbox_mask)
                
                img = cv2.imread(front_camera_image_path)
                
                if img is not None:

                    contours, _ = cv2.findContours(bbox_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
                        
                        cv2.imwrite(f'r_{frame_index}_{i+1}.png', img)
                        
                else:
                    print("无法加载原始图像，跳过渲染")
            else:
                print("出画过多")
        except Exception as e:
            print(f"渲染过程中出现错误: {e}")
    
    result_data = {
        'frame_index': frame_index,
        'target_box': {
            'original': target_box,
            'camera_coords': target_box_camera['center'],
            'camera_yaw': float(target_box_camera['yaw'])
        },
        'top_10_similar_boxes': results
    }
    
    with open(f'{frame_index}_top10.json', 'w') as f:
        json.dump(result_data, f, indent=2)
    print(f"\n结果已保存为 '{frame_index}_top10.json'")


if __name__ == "__main__":
    main()
