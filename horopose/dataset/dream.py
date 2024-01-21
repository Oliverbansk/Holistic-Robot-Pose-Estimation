import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import random
from collections import OrderedDict, defaultdict
from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from horopose.utils.geometry import quat_to_rotmat_np
from horopose.utils.transforms import invert_T
from dataset.augmentations import (CropResizeToAspectAugmentation, FlipAugmentation,
                            PillowBlur, PillowBrightness, PillowColor,
                            PillowContrast, PillowSharpness, occlusion_aug,
                            to_torch_uint8)
from dataset.const import KEYPOINT_NAMES, flip_pairs, rgb_augmentations
from dataset.roboutils import (bbox_transform, get_bbox, get_bbox_raw,
                        get_extended_bbox, make_masks_from_det,
                        process_padding, process_truncation, resize_image,
                        tensor_to_image)


KUKA_SYNT_TRAIN_DR_INCORRECT_IDS = {83114, 28630, }

def build_frame_index(base_dir):
    im_paths = base_dir.glob('*.jpg')
    infos = defaultdict(list)
    for n, im_path in tqdm(enumerate(sorted(im_paths))):
        view_id = int(im_path.with_suffix('').with_suffix('').name)
        if view_id == 0 and "panda_synth_test_photo" in str(base_dir):
            continue
        if 'kuka_synth_train_dr' in str(base_dir) and int(view_id) in KUKA_SYNT_TRAIN_DR_INCORRECT_IDS:
            pass
        else:
            scene_id = view_id
            infos['rgb_path'].append(im_path.as_posix())
            infos['scene_id'].append(scene_id)
            infos['view_id'].append(view_id)
    infos = pd.DataFrame(infos)
    return infos



class DreamDataset(torch.utils.data.Dataset):
    def __init__(self,
                 base_dir,
                 rootnet_resize_hw=(256, 256),
                 other_resize_hw=(256, 256),
                 visibility_check=True,
                 strict_crop=True,
                 color_jitter=True,
                 rgb_augmentation=True,
                 occlusion_augmentation=True,
                 flip=False,
                 rotate=False,
                 padding=False,
                 occlu_p=0.5,
                 process_truncation=False,
                 extend_ratio=[0.2,0.13]
                 ):
        self.base_dir = Path(base_dir)
        self.ds_name = os.path.basename(base_dir)
        self.rootnet_resize_hw=rootnet_resize_hw
        self.other_resize_hw=other_resize_hw
        self.color_jitter=color_jitter
        self.rgb_augmentation=rgb_augmentation
        self.rgb_augmentations=rgb_augmentations
        self.occlusion_augmentation=occlusion_augmentation
        self.total_occlusions = 1
        self.rootnet_flip=flip
        self.rootnet_rotate=rotate
        self.visibility_check = visibility_check
        self.process_truncation = process_truncation
        self.padding = padding
        self.occlu_p = occlu_p
        self.strict_crop = strict_crop
        self.extend_ratio = extend_ratio
    
        self.frame_index = build_frame_index(self.base_dir)
        self.synthetic = True
        if 'panda' in str(base_dir):
            self.keypoint_names = KEYPOINT_NAMES['panda']
            self.label = 'panda'
            if "panda-3cam" in self.ds_name or "panda-orb" in self.ds_name :
                self.synthetic = False
        elif 'baxter' in str(base_dir):
            self.keypoint_names = KEYPOINT_NAMES['baxter']
            self.label = 'baxter'
        elif 'kuka' in str(base_dir):
            self.keypoint_names = KEYPOINT_NAMES['kuka']
            self.label = 'kuka'
        else:
            raise NotImplementedError
        
        self.scale = 0.01 if 'synthetic' in str(self.base_dir) else 1.0
        self.all_labels = [self.label]
        self.flip_pairs=None
        if self.label == 'baxter':
            self.flip_pairs = flip_pairs

    def __len__(self):
        return len(self.frame_index)
    
    def _get_original_and_shared_data(self, idx):
        
        row = self.frame_index.iloc[idx]
        scene_id = row.scene_id
        rgb_path = Path(row.rgb_path)
        assert rgb_path
        rgb = np.asarray(Image.open(rgb_path))
        images_original = torch.FloatTensor(rgb.copy()).permute(2,0,1)
        mask = None
        annotations = json.loads(rgb_path.with_suffix('').with_suffix('.json').read_text())

        # Camera
        TWC = np.eye(4)
        camera_infos_path = self.base_dir / '_camera_settings.json'
        h, w = rgb.shape[0], rgb.shape[1]
        if camera_infos_path.exists():
            cam_infos = json.loads(camera_infos_path.read_text())
            assert len(cam_infos['camera_settings']) == 1
            cam_infos = cam_infos['camera_settings'][0]['intrinsic_settings']
            fx, fy, cx, cy = [cam_infos[k] for k in ('fx', 'fy', 'cx', 'cy')]
        else:
            fx, fy = 320, 320
            cx, cy = w/2, h/2

        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])

        camera = dict(
            TWC=TWC,
            resolution=(w, h),
            K=K,
        )
        label = self.label

        # Joints
        obj_data = annotations['objects'][0]
        if 'quaternion_xyzw' in obj_data:
            rotMat = quat_to_rotmat_np(np.array(obj_data['quaternion_xyzw']))
            translation = np.array(obj_data['location']) * self.scale
            TWO = np.zeros((4,4), dtype=float)
            TWO[:3, :3] = rotMat
            TWO[:3, 3] = translation
            TWO[3,3] = 1.0
            R_NORMAL_UE = np.array([
                [0, -1, 0],
                [0, 0, -1],
                [1, 0, 0],
            ])
            TWO[:3, :3] = TWO[:3, :3] @ R_NORMAL_UE  
            
        else:
            rotMat = quat_to_rotmat_np(np.array([1.0,0.0,0.0,0.0]))
            translation = np.array(obj_data['location']) * self.scale
            TWO = np.zeros((4,4), dtype=float)
            TWO[:3, :3] = rotMat
            TWO[:3, 3] = translation
            TWO[3,3] = 1.0
        TWC = torch.as_tensor(TWC)
        TWO = torch.as_tensor(TWO)
        TCO = invert_T(TWC) @ TWO
        TCO_r = torch.FloatTensor(np.asarray(TCO))

        joints = annotations['sim_state']['joints']
        joints = OrderedDict({d['name'].split('/')[-1]: float(d['position']) for d in joints})
        if self.label == 'kuka':
            joints = {k.replace('iiwa7_', 'iiwa_'): v for k,v in joints.items()}

        # keypoints 
        keypoints_data = obj_data['keypoints']
        keypoints_2d = np.concatenate([np.array(kp['projected_location'])[None] for kp in keypoints_data], axis=0)
        keypoints_2d = np.unique(keypoints_2d, axis=0)
        
        # bboxes
        bbox_gt2d = np.concatenate([np.min(keypoints_2d, axis=0), np.max(keypoints_2d, axis=0)])
        bbox = get_bbox(bbox_gt2d,w,h,strict=self.strict_crop)
        bboxes_r = bbox.copy()
        bboxes_raw = get_bbox_raw(bbox_gt2d)
        bbox_gt2d_extended_original = get_extended_bbox(bbox_gt2d, 20, 20, 20, 20, bounded=True, image_size=(w, h))
        
        if "bounding_box" in obj_data:
            bbox_strict_info = obj_data["bounding_box"]
            bbox_strict = np.array([bbox_strict_info["min"][0], bbox_strict_info["min"][1], bbox_strict_info["max"][0], bbox_strict_info["max"][1]])
            bbox_strict_bounded = np.array([max(0,bbox_strict[0]),max(0,bbox_strict[1]),min(w,bbox_strict[2]),min(h,bbox_strict[3])])
        else:
            bbox_strict_bounded = bbox_gt2d_extended_original
        bbox_strict_bounded_original = bbox_strict_bounded.copy()
        bbox_gt2d_extended_original = torch.FloatTensor(bbox_gt2d_extended_original)
        bbox_strict_bounded_original = torch.FloatTensor(bbox_strict_bounded_original)
        
        TCO_keypoints_3d = {kp['name']: np.array(kp['location']) * self.scale for kp in keypoints_data}
        TCO_keypoints_3d = np.array([TCO_keypoints_3d.get(k, np.nan) for k in self.keypoint_names])
        assert((np.isnan(TCO_keypoints_3d) == False).all())

        keypoints_2d = {kp['name']: kp['projected_location'] for kp in keypoints_data}
        keypoints_2d = np.array([np.append(keypoints_2d.get(k, np.nan) ,0)for k in self.keypoint_names])
        mask = make_masks_from_det(bbox[None], h, w).numpy().astype(np.uint8)[0] * 1     
        
        robot = dict(label=label, name=label, joints=joints,
                        TWO=TWO, 
                        bbox=bbox,
                        id_in_segm=1,
                        keypoints_2d=keypoints_2d,
                        TCO_keypoints_3d=TCO_keypoints_3d)

        state = dict(
            objects=[robot],
            camera=camera,
            frame_info=row.to_dict()
        )   
        
        K_original = state['camera']['K'].copy()   
        keypoints_2d_original = state['objects'][0]["keypoints_2d"].copy() 
        valid_mask = (keypoints_2d_original[:, 0] < 640.0) & (keypoints_2d_original[:, 0] >= 0) & \
                     (keypoints_2d_original[:, 1] < 480.0) & (keypoints_2d_original[:, 1] >= 0)
        
        if self.process_truncation:
            rgb, bbox, mask, state = process_truncation(rgb, bboxes_raw, mask, state)
        
        if self.color_jitter and random.random()<0.4:#color jitter #0.4
            self.color_factor=2*random.random()
            c_high = 1 + self.color_factor
            c_low = 1 - self.color_factor
            rgb=rgb.copy()
            rgb[:, :, 0] = np.clip(rgb[:, :, 0] * random.uniform(c_low, c_high), 0, 255)
            rgb[:, :, 1] = np.clip(rgb[:, :, 1] * random.uniform(c_low, c_high), 0, 255)
            rgb[:, :, 2] = np.clip(rgb[:, :, 2] * random.uniform(c_low, c_high), 0, 255)
            rgb=Image.fromarray(rgb)
            
        for _ in range(self.total_occlusions):
            if self.occlusion_augmentation and random.random() < self.occlu_p: #0.5
                rgb=np.array(rgb)
                synth_ymin, synth_h, synth_xmin, synth_w = occlusion_aug(bbox,np.array([h,w]), min_area=0.0, max_area=0.3, max_try_times=5)
                rgb=rgb.copy()
                rgb[synth_ymin:synth_ymin + synth_h, synth_xmin:synth_xmin + synth_w, :] = np.random.rand(synth_h, synth_w, 3) * 255
                rgb=Image.fromarray(rgb)
                
        if self.rgb_augmentation :
            augSharpness = PillowSharpness(p=0.3, factor_interval=(0., 50.)) #0.3
            augContrast = PillowContrast(p=0.3, factor_interval=(0.7, 1.8)) #0.3
            augBrightness = PillowBrightness(p=0.3, factor_interval=(0.7, 1.8)) #0.3
            augColor = PillowColor(p=0.3, factor_interval=(0., 4.)) #0.3
            rgb, mask, state = augSharpness(rgb, mask, state)
            rgb, mask, state = augContrast(rgb, mask, state)
            rgb, mask, state = augBrightness(rgb, mask, state)
            rgb, mask, state = augColor(rgb, mask, state)  
            rgb = np.array(rgb)
            
        keypoints_3d_original = state['objects'][0]["TCO_keypoints_3d"].copy()
        jointpose_r = state["objects"][0]["joints"]
        valid_mask_r = torch.FloatTensor(valid_mask)
        
        return {
                "meta": {"rgb":rgb, "bbox":bbox, "mask":mask, "state":state, "bboxes_raw":bboxes_raw},
                "image_id": idx,
                "scene_id": scene_id,
                "images_original": images_original,
                "bbox_strict_bounded_original": bbox_strict_bounded_original,
                "bbox_gt2d_extended_original": bbox_gt2d_extended_original,
                "TCO": TCO_r,
                "K_original":K_original,
                "jointpose": jointpose_r,
                "keypoints_2d_original":keypoints_2d_original[:,0:2],
                "valid_mask": valid_mask_r,
                "keypoints_3d_original": keypoints_3d_original,
            }
        

        
        
        
    def _get_rootnet_data(self, shared):
        
        rgb = deepcopy(shared["meta"]["rgb"])
        bbox = deepcopy(shared["meta"]["bbox"])
        mask = deepcopy(shared["meta"]["mask"])
        state = deepcopy(shared["meta"]["state"])
        bboxes_raw = deepcopy(shared["meta"]["bboxes_raw"])
        K_original = deepcopy(shared["K_original"])
        bbox_strict_bounded_original = deepcopy(shared["bbox_strict_bounded_original"])
        
        if self.rootnet_rotate:
            pass
            # RotationAugmentation
        
        resize_hw = self.rootnet_resize_hw
        rgb = np.asarray(rgb)
        rgb, mask, state = resize_image(rgb, bbox, mask, state)

        crop=CropResizeToAspectAugmentation(resize=resize_hw)
        rgb, mask, state = crop(rgb, mask, state)
        if self.rootnet_flip:
            rgb, mask, state = FlipAugmentation(p=0.5,flip_pairs=self.flip_pairs)(rgb, mask, state)
        if self.padding:
            rgb, bbox, mask, state = process_padding(rgb, bboxes_raw, mask, state, padding_pixel=30)
            rgb, mask, state = crop(rgb, mask, state)
        rgb, mask = to_torch_uint8(rgb), to_torch_uint8(mask)
        rgb = rgb.permute(2,0,1)

        images_r = torch.FloatTensor(np.asarray(rgb))
        K_r = torch.FloatTensor(np.asarray(state['camera']['K']))
        K_original_inv = np.linalg.inv(K_original)
        bbox_strict_bounded_transformed = bbox_transform(bbox_strict_bounded_original, K_original_inv, np.asarray(state['camera']['K']), resize_hw=resize_hw)
        bbox_strict_bounded_transformed = np.array([max(0,bbox_strict_bounded_transformed[0]),max(0, bbox_strict_bounded_transformed[1]),
                                                    min(resize_hw[0],bbox_strict_bounded_transformed[2]),min(resize_hw[1],bbox_strict_bounded_transformed[3])])
        bbox_strict_bounded_transformed = torch.FloatTensor(bbox_strict_bounded_transformed)

        bbox_from_transformed_gt2d = np.concatenate([np.min(state["objects"][0]["keypoints_2d"], axis=0)[0:2], np.max(state["objects"][0]["keypoints_2d"], axis=0)[0:2]])
        w_, h_ = (bbox_from_transformed_gt2d[2] - bbox_from_transformed_gt2d[0]), (bbox_from_transformed_gt2d[3] - bbox_from_transformed_gt2d[1])
        bbox_gt2d_extended = get_extended_bbox(bbox_from_transformed_gt2d, w_*self.extend_ratio[0], h_*self.extend_ratio[1], 
                                               w_*self.extend_ratio[0], h_*self.extend_ratio[1], bounded=True, image_size=resize_hw)
        bbox_gt2d_extended = torch.FloatTensor(bbox_gt2d_extended)
        
        keypoints_3d_r = torch.FloatTensor(state["objects"][0]["TCO_keypoints_3d"])
        keypoints_2d_r = torch.FloatTensor(state["objects"][0]["keypoints_2d"])[:,0:2]
        keypoints_2d = keypoints_2d_r.numpy()
        valid_mask_crop = torch.FloatTensor( ((keypoints_2d[:, 0] < resize_hw[0]) & (keypoints_2d[:, 0] >= 0) & \
                                            (keypoints_2d[:, 1] < resize_hw[1]) & (keypoints_2d[:, 1] >= 0)))
        
        return {
            "images":images_r, 
            "bbox_strict_bounded": bbox_strict_bounded_transformed, 
            "bbox_gt2d_extended" : bbox_gt2d_extended,
            "K":K_r,
            "keypoints_3d":keypoints_3d_r,
            "keypoints_2d":keypoints_2d_r,
            "valid_mask_crop": valid_mask_crop,
        }
        
        
    def _get_other_data(self, shared):
        rgb = deepcopy(shared["meta"]["rgb"])
        bbox = deepcopy(shared["meta"]["bbox"])
        mask = deepcopy(shared["meta"]["mask"])
        state = deepcopy(shared["meta"]["state"])
        bboxes_raw = deepcopy(shared["meta"]["bboxes_raw"])
        K_original = deepcopy(shared["K_original"])
        bbox_strict_bounded_original = deepcopy(shared["bbox_strict_bounded_original"])
        
        resize_hw = self.other_resize_hw
        rgb = np.asarray(rgb)
        rgb, mask, state = resize_image(rgb, bbox, mask, state)

        crop=CropResizeToAspectAugmentation(resize=resize_hw)
        rgb, mask, state = crop(rgb, mask, state)
        if self.padding:
            rgb, bbox, mask, state = process_padding(rgb, bboxes_raw, mask, state, padding_pixel=30)
            rgb, mask, state = crop(rgb, mask, state)
        rgb, mask = to_torch_uint8(rgb), to_torch_uint8(mask)
        rgb = rgb.permute(2,0,1)

        images_r = torch.FloatTensor(np.asarray(rgb))
        K_r = torch.FloatTensor(np.asarray(state['camera']['K']))
        K_original_inv = np.linalg.inv(K_original)
        bbox_strict_bounded_transformed = bbox_transform(bbox_strict_bounded_original, K_original_inv, np.asarray(state['camera']['K']), resize_hw=resize_hw)
        bbox_strict_bounded_transformed = np.array([max(0,bbox_strict_bounded_transformed[0]),max(0, bbox_strict_bounded_transformed[1]),
                                                    min(resize_hw[0],bbox_strict_bounded_transformed[2]),min(resize_hw[1],bbox_strict_bounded_transformed[3])])
        bbox_strict_bounded_transformed = torch.FloatTensor(bbox_strict_bounded_transformed)

        bbox_from_transformed_gt2d = np.concatenate([np.min(state["objects"][0]["keypoints_2d"], axis=0)[0:2], np.max(state["objects"][0]["keypoints_2d"], axis=0)[0:2]])
        w_, h_ = (bbox_from_transformed_gt2d[2] - bbox_from_transformed_gt2d[0]), (bbox_from_transformed_gt2d[3] - bbox_from_transformed_gt2d[1])
        bbox_gt2d_extended = get_extended_bbox(bbox_from_transformed_gt2d, w_*self.extend_ratio[0], h_*self.extend_ratio[1], w_*self.extend_ratio[0], h_*self.extend_ratio[1], bounded=True, image_size=resize_hw)
        bbox_gt2d_extended = torch.FloatTensor(bbox_gt2d_extended)
        
        keypoints_3d_r = torch.FloatTensor(state["objects"][0]["TCO_keypoints_3d"])
        keypoints_2d_r = torch.FloatTensor(state["objects"][0]["keypoints_2d"])[:,0:2]
        keypoints_2d = keypoints_2d_r.numpy()
        valid_mask_crop = torch.FloatTensor( ((keypoints_2d[:, 0] < resize_hw[0]) & (keypoints_2d[:, 0] >= 0) & \
                                            (keypoints_2d[:, 1] < resize_hw[1]) & (keypoints_2d[:, 1] >= 0)))
        
        return {
            "images":images_r, 
            "bbox_strict_bounded": bbox_strict_bounded_transformed, 
            "bbox_gt2d_extended" : bbox_gt2d_extended,
            "K":K_r,
            "keypoints_3d":keypoints_3d_r,
            "keypoints_2d":keypoints_2d_r,
            "valid_mask_crop": valid_mask_crop,
        }
        
        
        
    
    def __getitem__(self, idx):
        
        shared_data = self._get_original_and_shared_data(idx)
        rootnet_data = self._get_rootnet_data(shared_data)
        other_data = self._get_other_data(shared_data)
        
        return {
                "image_id": shared_data["image_id"],
                "scene_id": shared_data["scene_id"],
                "images_original": shared_data["images_original"],
                "bbox_strict_bounded_original": shared_data["bbox_strict_bounded_original"],
                "bbox_gt2d_extended_original": shared_data["bbox_gt2d_extended_original"],
                "TCO":shared_data["TCO"],
                "K_original":shared_data["K_original"],
                "jointpose":shared_data["jointpose"],
                "keypoints_2d_original":shared_data["keypoints_2d_original"],
                "valid_mask": shared_data["valid_mask"],
                "keypoints_3d_original": shared_data["keypoints_3d_original"],
                "root": rootnet_data,
                "other": other_data
            }