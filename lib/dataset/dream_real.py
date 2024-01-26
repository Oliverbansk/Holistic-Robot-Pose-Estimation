import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import random
import time
from collections import OrderedDict, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from lib.config import MEMORY
from lib.dataset.roboutils import (bbox_transform, get_bbox, get_bbox_raw,
                                   get_extended_bbox, make_masks_from_det,
                                   process_padding, process_truncation,
                                   resize_image, tensor_to_image)
from lib.utils.geometries import quat_to_rotmat_np
from lib.utils.transforms import (hnormalized, invert_T,
                                  point_projection_from_3d)
from PIL import Image
from tqdm import tqdm
from .augmentations import (CropResizeToAspectAugmentation, FlipAugmentation,
                            PillowBlur, PillowBrightness, PillowColor,
                            PillowContrast, PillowSharpness, occlusion_aug,
                            to_torch_uint8)
from .const import KEYPOINT_NAMES, flip_pairs, rgb_augmentations


def build_frame_index(base_dir):
    im_paths = base_dir.glob('*.jpg')
    infos = defaultdict(list)
    im_paths = sorted( im_paths, key=lambda x:int(str(x).split("/")[-1].split(".")[0]) )
    for n, im_path in tqdm(enumerate(im_paths)):
        if "move_camera2" in os.path.abspath(base_dir):
            if not n >= 2000: 
                continue
        if "move_both" in os.path.abspath(base_dir):
            if not (n >= 900 and n <= 1000): 
                continue
        if "move_arm" in os.path.abspath(base_dir):
            if n % 5 != 0: 
                continue
        view_id = int(str(im_path).split("/")[-1].split(".")[0])
        scene_id = view_id
        infos['rgb_path'].append(im_path.as_posix())
        infos['scene_id'].append(scene_id)
        infos['view_id'].append(view_id)
    infos = pd.DataFrame(infos)
    return infos

class DreamRealDataset(torch.utils.data.Dataset):
    
    def __init__(self,
                 base_dir,
                 rootnet_resize_hw=(256, 256),
                 other_resize_hw=(256, 256),
                 extend_ratio=[0.2,0.13]):
        self.base_dir = Path(base_dir)
        self.ds_name = os.path.basename(base_dir)
        self.rootnet_resize_hw=rootnet_resize_hw
        self.other_resize_hw=other_resize_hw
        self.extend_ratio = extend_ratio
        self.strict_crop = True
        self.frame_index = build_frame_index(self.base_dir)
        self.json_path = os.path.join(base_dir, "kp.json")
        self.keypoints_2d = json.loads(Path(self.json_path).read_text())
        
    def __len__(self):
        return len(self.frame_index)
    
    def _get_original_and_shared_data(self, idx):
        
        label = "real"
        joints = np.ones(8)
        TCO = TWO = TWC = np.eye(4)
        TCO_r = TCO.copy()
        keypoints_3d = np.ones((7,3))
        
        
        row = self.frame_index.iloc[idx]
        scene_id = row.scene_id
        rgb_path = Path(row.rgb_path)
        assert rgb_path
        rgb = np.asarray(Image.open(rgb_path))
        h, w = rgb.shape[0], rgb.shape[1]
        images_original = torch.FloatTensor(rgb.copy()).permute(2,0,1)
        keypoints_2d = np.array(self.keypoints_2d[str(scene_id)])
        K = np.array([[399.6578776041667,0,319.8955891927083],
                    [0,399.4959309895833,244.0602823893229],
                    [0,0,1]])
        # depths = 1.0
        # keypoints_3d = (keypoints_2d[:,0] + K[0,2]) 
        bbox_gt2d = np.concatenate([np.min(keypoints_2d, axis=0), np.max(keypoints_2d, axis=0)])
        bbox = get_bbox(bbox_gt2d,w,h,strict=self.strict_crop)
        bboxes_raw = get_bbox_raw(bbox_gt2d)
        bbox_gt2d_extended_original = get_extended_bbox(bbox_gt2d, 20, 20, 20, 20, bounded=True, image_size=(w, h))
        bbox_strict_bounded = bbox_gt2d_extended_original
        bbox_strict_bounded_original = bbox_strict_bounded.copy()
        bbox_gt2d_extended_original = torch.FloatTensor(bbox_gt2d_extended_original)
        bbox_strict_bounded_original = torch.FloatTensor(bbox_strict_bounded_original)
        mask = make_masks_from_det(bbox[None], h, w).numpy().astype(np.uint8)[0] * 1
        
        camera = dict(
            TWC=TWC,
            resolution=(w, h),
            K=K,
        )
        
        robot = dict(label=label, name=label, joints=joints,
                        TWO=TWO, 
                        bbox=bbox,
                        id_in_segm=1,
                        keypoints_2d=keypoints_2d,
                        TCO_keypoints_3d=keypoints_3d)
        
        state = dict(
            objects=[robot],
            camera=camera,
            frame_info=row.to_dict()
        )
        
        K_original = state['camera']['K'].copy()   
        keypoints_2d_original = state['objects'][0]["keypoints_2d"].copy() 
        valid_mask = (keypoints_2d_original[:, 0] < 640.0) & (keypoints_2d_original[:, 0] >= 0) & \
                     (keypoints_2d_original[:, 1] < 480.0) & (keypoints_2d_original[:, 1] >= 0)
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
        
        resize_hw = self.rootnet_resize_hw
        rgb = np.asarray(rgb)
        rgb, mask, state = resize_image(rgb, bbox, mask, state)

        crop=CropResizeToAspectAugmentation(resize=resize_hw)
        rgb, mask, state = crop(rgb, mask, state, use_3d=False)
        
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
        rgb, mask, state = crop(rgb, mask, state, use_3d=False)
        
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