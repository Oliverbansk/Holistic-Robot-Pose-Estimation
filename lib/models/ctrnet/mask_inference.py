import os
import sys
base_dir = os.path.abspath(".")
sys.path.append(base_dir)
import argparse
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image as PILImage
from .CtRNet import CtRNet


class seg_mask_inference(torch.nn.Module):
    def __init__(self, intrinsics, dataset, image_hw=(480, 640), scale=0.5):
        super(seg_mask_inference, self).__init__()
        self.args = self.set_args(intrinsics, dataset, image_hw, scale)
        self.net = CtRNet(self.args)
        self.trans_to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
    def set_args(self, intrinsics, dataset, image_hw=(480, 640), scale=0.5):
        parser = argparse.ArgumentParser()
        args = parser.parse_args("")
        args.use_gpu = True
        args.robot_name = 'Panda' 
        args.n_kp = 7
        args.scale = scale
        args.height, args.width = image_hw
        args.fx, args.fy, args.px, args.py = intrinsics
        args.width, args.height = int(args.width * args.scale), int(args.height * args.scale)
        args.fx, args.fy, args.px, args.py = args.fx * args.scale, args.fy * args.scale, args.px * args.scale, args.py * args.scale
        
        if "realsense" in dataset:
            args.keypoint_seg_model_path = "models/panda_segmentation/realsense.pth"
        elif "azure" in dataset:
            args.keypoint_seg_model_path = "models/panda_segmentation/azure.pth"
        elif "kinect" in dataset:
            args.keypoint_seg_model_path = "models/panda_segmentation/kinect.pth"
        elif "orb" in dataset:
            args.keypoint_seg_model_path = "models/panda_segmentation/orb.pth"
        else:
            args.keypoint_seg_model_path = "models/panda_segmentation/azure.pth"
        
        return args
         
    def preprocess_img_tensor(self, img_tensor):
        width, height = img_tensor.shape[3], img_tensor.shape[2]
        img_array = np.uint8(img_tensor.detach().cpu().numpy()).transpose(0, 2, 3, 1)
        new_size = (int(width*self.args.scale),int(height*self.args.scale))
        pil_image = [self.trans_to_tensor(PILImage.fromarray(img).resize(new_size)) for img in img_array]
        return torch.stack(pil_image)
    
    def forward(self, img_tensor):
         
        image = self.preprocess_img_tensor(img_tensor).cuda()
        segmentation = self.net.inference_batch_images_onlyseg(image)
        
        return segmentation
    
class seg_keypoint_inference(torch.nn.Module):
    def __init__(self, image_hw=(480, 640), scale=0.5):
        super(seg_keypoint_inference, self).__init__()
        self.args = self.set_args(image_hw, scale)
        self.net = CtRNet(self.args)
        self.trans_to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
    def set_args(self, image_hw=(480, 640), scale=0.5):
        parser = argparse.ArgumentParser()
        args = parser.parse_args("")
        args.use_gpu = True
        args.robot_name = 'Panda' 
        args.n_kp = 7
        args.scale = scale
        args.height, args.width = image_hw
        args.fx, args.fy, args.px, args.py = 320,320,320,240
        args.width, args.height = int(args.width * args.scale), int(args.height * args.scale)
        args.fx, args.fy, args.px, args.py = args.fx * args.scale, args.fy * args.scale, args.px * args.scale, args.py * args.scale
        args.keypoint_seg_model_path = "models/panda_segmentation/azure.pth"
        
        return args
         
    def preprocess_img_tensor(self, img_tensor):
        width, height = img_tensor.shape[3], img_tensor.shape[2]
        img_array = np.uint8(img_tensor.detach().cpu().numpy()).transpose(0, 2, 3, 1)
        new_size = (int(width*self.args.scale),int(height*self.args.scale))
        pil_image = [self.trans_to_tensor(PILImage.fromarray(img).resize(new_size)) for img in img_array]
        return torch.stack(pil_image)
    
    def forward(self, img_tensor):
         
        image = self.preprocess_img_tensor(img_tensor).cuda()
        keypoints, segmentation = self.net.inference_batch_images_seg_kp(image)
        
        return keypoints, segmentation