import torch
import torch.nn.functional as F
import kornia
import numpy as np

from .keypoint_seg_resnet import KeyPointSegNet
from utils.BPnP import BPnP, BPnP_m3d, batch_project


class CtRNet(torch.nn.Module):
    def __init__(self, args):
        super(CtRNet, self).__init__()

        self.args = args

        if args.use_gpu:
            self.device = "cuda"
        else:
            self.device = "cpu"

        # load keypoint segmentation model
        self.keypoint_seg_predictor = KeyPointSegNet(args, use_gpu=args.use_gpu)

        if args.use_gpu:
            self.keypoint_seg_predictor = self.keypoint_seg_predictor.cuda()
        
        self.keypoint_seg_predictor = torch.nn.DataParallel(self.keypoint_seg_predictor, device_ids=[0])
        
        if args.keypoint_seg_model_path is not None:
            print("Loading keypoint segmentation model from {}".format(args.keypoint_seg_model_path))
            self.keypoint_seg_predictor.load_state_dict(torch.load(args.keypoint_seg_model_path))

        self.keypoint_seg_predictor.eval()

        # load BPnP
        self.bpnp = BPnP.apply
        self.bpnp_m3d = BPnP_m3d.apply

        # set up camera intrinsics

        self.intrinsics = np.array([[   args.fx,    0.     ,    args.px   ],
                                    [   0.     ,    args.fy,    args.py   ],
                                    [   0.     ,    0.     ,    1.        ]])
        print("Camera intrinsics: {}".format(self.intrinsics))
        
        self.K = torch.tensor(self.intrinsics, device=self.device, dtype=torch.float)


    def inference_single_image(self, img, joint_angles):
        # img: (3, H, W)
        # joint_angles: (7)
        # robot: robot model

        # detect 2d keypoints and segmentation masks
        points_2d, segmentation = self.keypoint_seg_predictor(img[None])
        foreground_mask = torch.sigmoid(segmentation)
        _,t_list = self.robot.get_joint_RT(joint_angles)
        points_3d = torch.from_numpy(np.array(t_list)).float().to(self.device)
        if self.args.robot_name == "Panda":
            points_3d = points_3d[[0,2,3,4,6,7,8]] # remove 1 and 5 links as they are overlapping with 2 and 6

        #init_pose = torch.tensor([[  1.5497,  0.5420, -0.3909, -0.4698, -0.0211,  1.3243]])
        #cTr = bpnp(points_2d_pred, points_3d, K, init_pose)
        cTr = self.bpnp(points_2d, points_3d, self.K)

        return cTr, points_2d, foreground_mask
    
    def inference_batch_images(self, img, joint_angles):
        # img: (B, 3, H, W)
        # joint_angles: (B, 7)
        # robot: robot model

        # detect 2d keypoints and segmentation masks
        points_2d, segmentation = self.keypoint_seg_predictor(img)
        foreground_mask = torch.sigmoid(segmentation)

        points_3d_batch = []
        for b in range(joint_angles.shape[0]):
            _,t_list = self.robot.get_joint_RT(joint_angles[b])
            points_3d = torch.from_numpy(np.array(t_list)).float().to(self.device)
            if self.args.robot_name == "Panda":
                points_3d = points_3d[:,[0,2,3,4,6,7,8]]
            points_3d_batch.append(points_3d[None])

        points_3d_batch = torch.cat(points_3d_batch, dim=0)

        cTr = self.bpnp_m3d(points_2d, points_3d_batch, self.K)

        return cTr, points_2d, foreground_mask
    
    def inference_batch_images_seg_kp(self, img):
        # img: (B, 3, H, W)
        # joint_angles: (B, 7)
        # robot: robot model

        # detect 2d keypoints and segmentation masks
        points_2d, segmentation = self.keypoint_seg_predictor(img)
        foreground_mask = torch.sigmoid(segmentation)
        
        return points_2d, foreground_mask
    
    def inference_batch_images_onlyseg(self, img):
        # img: (B, 3, H, W)
        # joint_angles: (B, 7)
        # robot: robot model

        # detect 2d keypoints and segmentation masks
        points_2d, segmentation = self.keypoint_seg_predictor(img)
        foreground_mask = torch.sigmoid(segmentation)

        return foreground_mask

    
    def cTr_to_pose_matrix(self, cTr):
        """
        cTr: (batch_size, 6)
        pose_matrix: (batch_size, 4, 4)
        """
        batch_size = cTr.shape[0]
        pose_matrix = torch.zeros((batch_size, 4, 4), device=self.device)
        pose_matrix[:, :3, :3] = kornia.geometry.conversions.axis_angle_to_rotation_matrix(cTr[:, :3])
        pose_matrix[:, :3, 3] = cTr[:, 3:]
        pose_matrix[:, 3, 3] = 1
        return pose_matrix
    
    def to_valid_R_batch(self, R):
        # R is a batch of 3x3 rotation matrices
        U, S, V = torch.svd(R)
        return torch.bmm(U, V.transpose(1,2))
    
    def render_single_robot_mask(self, cTr, robot_mesh, robot_renderer):
        # cTr: (6)
        # img: (1, H, W)

        R = kornia.geometry.conversions.angle_axis_to_rotation_matrix(cTr[:3][None])  # (1, 3, 3)
        R = torch.transpose(R,1,2)
        #R = to_valid_R_batch(R)
        T = cTr[3:][None]   # (1, 3)

        if T[0,-1] < 0:
            rendered_image = robot_renderer.silhouette_renderer(meshes_world=robot_mesh, R = -R, T = -T)
        else:
            rendered_image = robot_renderer.silhouette_renderer(meshes_world=robot_mesh, R = R, T = T)

        if torch.isnan(rendered_image).any():
            rendered_image = torch.nan_to_num(rendered_image)
        
        return rendered_image[..., 3]


    def train_on_batch(self, img, joint_angles, robot_renderer, criterions, phase='train'):
        # img: (B, 3, H, W)
        # joint_angles: (B, 7)
        with torch.set_grad_enabled(phase == 'train'):
            # detect 2d keypoints
            points_2d, segmentation = self.keypoint_seg_predictor(img)

            mask_list = list()
            seg_weight_list = list()

            for b in range(img.shape[0]):
                # get 3d points
                _,t_list = self.robot.get_joint_RT(joint_angles[b])
                points_3d = torch.from_numpy(np.array(t_list)).float().to(self.device)
                if self.args.robot_name == "Panda":
                    points_3d = points_3d[:,[0,2,3,4,6,7,8]]

                # get camera pose
                cTr = self.bpnp(points_2d[b][None], points_3d, self.K)

                # config robot mesh
                robot_mesh = robot_renderer.get_robot_mesh(joint_angles[b])

                # render robot mask
                rendered_image = self.render_single_robot_mask(cTr.squeeze(), robot_mesh, robot_renderer)

                mask_list.append(rendered_image)
                points_2d_proj = batch_project(cTr, points_3d, self.K)
                reproject_error = criterions["mse_mean"](points_2d[b], points_2d_proj.squeeze())
                seg_weight = torch.exp(-reproject_error * self.args.reproj_err_scale)
                seg_weight_list.append(seg_weight)

            mask_batch = torch.cat(mask_list,0)

            loss_bce = 0
            for b in range(segmentation.shape[0]):
                loss_bce = loss_bce + seg_weight_list[b] * criterions["bce"](segmentation[b].squeeze(), mask_batch[b].detach())

            img_ref = torch.sigmoid(segmentation).detach()
            #loss_reproj = 0.0005 * criterionMSE_mean(points_2d, points_2d_proj_batch)
            loss_mse = 0.001 * criterions["mse_sum"](mask_batch, img_ref.squeeze())
            loss = loss_mse + loss_bce 
            
        return loss





