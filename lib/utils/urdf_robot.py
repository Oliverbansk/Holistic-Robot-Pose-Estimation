import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import platform
import numpy as np
import pandas as pd
import pyrender
import torch
from config import (BAXTER_DESCRIPTION_PATH, KUKA_DESCRIPTION_PATH,
                    OWI_DESCRIPTION, OWI_KEYPOINTS_PATH,
                    PANDA_DESCRIPTION_PATH, PANDA_DESCRIPTION_PATH_VISUAL)
from dataset.const import JOINT_NAMES, LINK_NAMES
from PIL import Image
from utils.geometries import (quat_to_rotmat, rot6d_to_rotmat, rot9d_to_rotmat,
                            rotmat_to_quat, rotmat_to_rot6d)
from utils.mesh_renderer import RobotMeshRenderer, PandaArm
from utils.urdfpytorch import URDF

if platform.system() == "Linux":
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

class URDFRobot:
    def __init__(self,robot_type):
        self.robot_type = robot_type
        if self.robot_type == "panda":
            self.urdf_path = PANDA_DESCRIPTION_PATH
            self.urdf_path_visual = PANDA_DESCRIPTION_PATH_VISUAL
            self.dof = 8
            self.robot_for_render = PandaArm(self.urdf_path)
        elif self.robot_type == "kuka":
            self.urdf_path = KUKA_DESCRIPTION_PATH
            self.urdf_path_visual = KUKA_DESCRIPTION_PATH
            self.dof = 7
            self.robot_for_render = PandaArm(self.urdf_path)
        elif self.robot_type == "baxter":
            self.urdf_path = BAXTER_DESCRIPTION_PATH
            self.urdf_path_visual = BAXTER_DESCRIPTION_PATH
            self.dof = 15
            self.robot_for_render = PandaArm(self.urdf_path)
        elif self.robot_type == "owi":
            self.urdf_path = OWI_DESCRIPTION
            self.urdf_path_visual = OWI_DESCRIPTION
            self.dof = 4
            self.robot_for_render = None
        self.robot = URDF.load(self.urdf_path)
        self.robot_visual = URDF.load(self.urdf_path_visual)
        self.actuated_joint_names = JOINT_NAMES[self.robot_type]
        self.global_scale = 1.0
        self.device = None
        self.link_names, self.offsets = self.get_link_names_and_offsets()
        
    def get_link_names_and_offsets(self):
        if self.robot_type == "panda" or self.robot_type == "kuka":
            kp_offsets = torch.zeros((len(LINK_NAMES[self.robot_type]),3),dtype=torch.float).unsqueeze(0).unsqueeze(-1) * self.global_scale
            kp_offsets = kp_offsets.to(torch.float)
            return LINK_NAMES[self.robot_type], kp_offsets
        elif self.robot_type == "baxter":
            joint_name_to_joint = {joint.name: joint for joint in self.robot.joints}
            offsets = []
            link_names = []
            joint_names_for_links = [
                'torso_t0', 'right_s0','left_s0', 'right_s1', 'left_s1',
                'right_e0','left_e0', 'right_e1','left_e1','right_w0', 'left_w0',
                'right_w1','left_w1','right_w2', 'left_w2','right_hand','left_hand'
            ]
            for joint_name in joint_names_for_links:
                joint = joint_name_to_joint[joint_name]
                offset = joint.origin[:3, -1]
                link_name = joint.parent
                link_names.append(link_name)
                offsets.append(offset)
            kp_offsets = torch.as_tensor(np.stack(offsets)).unsqueeze(0).unsqueeze(-1) * self.global_scale
            kp_offsets = kp_offsets.to(torch.float)
            return link_names, kp_offsets
        elif self.robot_type == "owi":
            keypoint_infos = pd.read_json(OWI_KEYPOINTS_PATH)
            kp_offsets = torch.as_tensor(np.stack(keypoint_infos['offset'])).unsqueeze(0).unsqueeze(-1).to(torch.float)
            return LINK_NAMES[self.robot_type], kp_offsets
        else:
            raise(NotImplementedError)
    
    def get_keypoints(self, jointcfgs, b2c_rot, b2c_trans):
        
        # jointcfgs, b2c_rot, b2c_trans all comes in batch (as model outputs)
        # b2c means base to camera
    
        batch_size = b2c_rot.shape[0]
        if b2c_rot.shape[1] == 6:
            rotmat = rot6d_to_rotmat(b2c_rot)
        elif b2c_rot.shape[1] == 4:
            rotmat = quat_to_rotmat(b2c_rot)
        elif b2c_rot.shape[1] == 9:
            rotmat = rot9d_to_rotmat(b2c_rot)
        else:
            raise NotImplementedError
        trans = b2c_trans.unsqueeze(dim=2)
        pad = torch.zeros((batch_size,1,4),dtype=torch.float).cuda()
        base2cam = torch.cat([rotmat,trans],dim=2).cuda()
        base2cam = torch.cat([base2cam,pad],dim=1).cuda()
        base2cam[:,3,3] = 1.0
        base2cam = base2cam.unsqueeze(1)
        TWL_base = self.get_TWL(jointcfgs).cuda()
        TWL = base2cam @ TWL_base
        pts = TWL[:, :, :3, :3] @ self.offsets.cuda() + TWL[:, :, :3, [-1]]
        return pts.squeeze(-1)
    
    def get_TWL(self, cfgs):
        fk = self.robot.link_fk_batch(cfgs, use_names=True)
        TWL = torch.stack([fk[link] for link in self.link_names]).permute(1, 0, 2, 3)
        TWL[..., :3, -1] *= self.global_scale
        return TWL

    def get_rotation_at_specific_root(self, jointcfgs, b2c_rot, b2c_trans, root = 0):
        if root == 0:
            return b2c_rot
        batch_size = b2c_rot.shape[0]
        if b2c_rot.shape[1] == 6:
            rotmat = rot6d_to_rotmat(b2c_rot)
        elif b2c_rot.shape[1] == 4:
            rotmat = quat_to_rotmat(b2c_rot)
        elif b2c_rot.shape[1] == 9:
            rotmat = rot9d_to_rotmat(b2c_rot)
        else:
            raise NotImplementedError
        trans = b2c_trans.unsqueeze(dim=2)
        pad = torch.zeros((batch_size,1,4),dtype=torch.float).cuda()
        base2cam = torch.cat([rotmat,trans],dim=2).cuda()
        base2cam = torch.cat([base2cam,pad],dim=1).cuda()
        base2cam[:,3,3] = 1.0
        base2cam = base2cam.unsqueeze(1)
        TWL_base = self.get_TWL(jointcfgs).cuda()
        TWL = base2cam @ TWL_base
        assert root < TWL.shape[1], (root, TWL.shape[1])
        if b2c_rot.shape[1] == 6:
            rotation = rotmat_to_rot6d(TWL[:, root, :3, :3]).cuda()
        elif b2c_rot.shape[1] == 4:
            rotation = rotmat_to_quat(TWL[:, root, :3, :3]).cuda()
        return rotation
        

    def get_keypoints_only_fk(self, jointcfgs):

        # only using joint angles to perform forward kinematics
        # the fk process is used when assuming the world frame is at the robot base, so rotation is identity and translation is origin/zeros
        # the output from this fk function is used for pnp process
        
        TWL = self.get_TWL(jointcfgs).cuda()
        pts = TWL[:, :, :3, :3] @ self.offsets.cuda() + TWL[:, :, :3, [-1]]
        return pts.squeeze(-1)
    
    def get_keypoints_only_fk_at_specific_root(self, jointcfgs, root=0):

        # only using joint angles to perform forward kinematics
        # the fk process is used when assuming the world frame is at the robot base, so rotation is identity and translation is origin/zeros
        # the output from this fk function is used for pnp process
        
        if root == 0:
            return self.get_keypoints_only_fk(jointcfgs)
        else:
            assert root > 0 and root < len(self.link_names)
        
        TWL_base = self.get_TWL(jointcfgs).cuda()
        TWL_root_inv = torch.linalg.inv(TWL_base[:,root:root+1,:,:])
        TWL = TWL_root_inv @ TWL_base
        pts = TWL[:, :, :3, :3] @ self.offsets.cuda() + TWL[:, :, :3, [-1]]
        return pts.squeeze(-1)

    
    def get_keypoints_root(self, jointcfgs, b2c_rot, b2c_trans, root = 0):
        
        # jointcfgs, b2c_rot, b2c_trans all comes in batch (as model outputs)
        # b2c here means *** root *** to camera

        if root == 0:
            return self.get_keypoints(jointcfgs, b2c_rot, b2c_trans)
        else:
            assert root > 0 and root < len(self.link_names)
        
        batch_size = b2c_rot.shape[0]
        if b2c_rot.shape[1] == 6:
            rotmat = rot6d_to_rotmat(b2c_rot)
        elif b2c_rot.shape[1] == 4:
            rotmat = quat_to_rotmat(b2c_rot)
        elif b2c_rot.shape[1] == 9:
            rotmat = rot9d_to_rotmat(b2c_rot)
        else:
            raise NotImplementedError
        trans = b2c_trans.unsqueeze(dim=2)
        pad = torch.zeros((batch_size,1,4),dtype=torch.float).cuda()
        base2cam = torch.cat([rotmat,trans],dim=2).cuda()
        base2cam = torch.cat([base2cam,pad],dim=1).cuda()
        base2cam[:,3,3] = 1.0
        base2cam = base2cam.unsqueeze(1)
        TWL_base = self.get_TWL(jointcfgs).cuda()
        TWL_root_inv = torch.linalg.inv(TWL_base[:,root:root+1,:,:])
        TWL_base = TWL_root_inv @ TWL_base
        TWL = base2cam @ TWL_base
        pts = TWL[:, :, :3, :3] @ self.offsets.cuda() + TWL[:, :, :3, [-1]]
        return pts.squeeze(-1)
        
    def set_robot_renderer(self, K_original, original_image_size=(480, 640), scale=0.5, device="cpu"):
        
        fx, fy, cx, cy = K_original[0,0]*scale, K_original[1,1]*scale, K_original[0,2]*scale, K_original[1,2]*scale
        image_size = (int(original_image_size[0]*scale), int(original_image_size[1]*scale))
        
        base_dir = os.path.dirname(self.urdf_path)
        
        mesh_files = [
                    base_dir + "/meshes/visual/link0/link0.obj",
                    base_dir + "/meshes/visual/link1/link1.obj",
                    base_dir + "/meshes/visual/link2/link2.obj",
                    base_dir + "/meshes/visual/link3/link3.obj",
                    base_dir + "/meshes/visual/link4/link4.obj",
                    base_dir + "/meshes/visual/link5/link5.obj",
                    base_dir + "/meshes/visual/link6/link6.obj",
                    base_dir + "/meshes/visual/link7/link7.obj",
                    base_dir + "/meshes/visual/hand/hand.obj",
                    ]
        
        focal_length = [-fx,-fy]
        principal_point = [cx, cy]
        
        robot_renderer = RobotMeshRenderer(
            focal_length=focal_length, principal_point=principal_point, image_size=image_size, 
            robot=self.robot_for_render, mesh_files=mesh_files, device=device)
        
        return robot_renderer
    
    def get_robot_mesh_list(self, joint_angles, renderer):
        
        robot_meshes = []
        for joint_angle in joint_angles:
            if self.robot_type == "panda":
                joints = joint_angle[:-1].detach().cpu()
            else:
                joints = joint_angle.detach().cpu()
            robot_mesh = renderer.get_robot_mesh(joints).cuda()
            robot_meshes.append(robot_mesh)
        
        return robot_meshes
          
    def get_rendered_mask_single_image(self, rot, trans, robot_mesh, robot_renderer_gpu):
        
        R = rot6d_to_rotmat(rot).reshape(1,3,3)
        R = torch.transpose(R,1,2).cuda()
        
        T = trans.reshape(1,3).cuda()
        
        if T[0,-1] < 0:
            rendered_image = robot_renderer_gpu.silhouette_renderer(meshes_world=robot_mesh, R = -R, T = -T)
        else:
            rendered_image = robot_renderer_gpu.silhouette_renderer(meshes_world=robot_mesh, R = R, T = T)

        if torch.isnan(rendered_image).any():
            rendered_image = torch.nan_to_num(rendered_image)
        
        return rendered_image[..., 3]
    
    def get_rendered_mask_single_image_at_specific_root(self, joint_angles, rot, trans, robot_mesh, robot_renderer_gpu, root=0):
        
        if root == 0:
            return self.get_rendered_mask_single_image(rot, trans, robot_mesh, robot_renderer_gpu)
        else:
            rotmat = rot6d_to_rotmat(rot).cuda()
            trans = trans.unsqueeze(dim=1).cuda()
            pad = torch.zeros((1,4),dtype=torch.float).cuda()
            base2cam = torch.cat([rotmat,trans],dim=1).cuda()
            base2cam = torch.cat([base2cam,pad],dim=0).cuda()
            base2cam[3,3] = 1.0
            TWL_base = self.get_TWL(joint_angles.unsqueeze(0)).cuda().detach() # detach joint with rot/trans
            TWL_root_inv = torch.linalg.inv(TWL_base[:,root:root+1,:,:]).squeeze()
            new_base2cam = base2cam @ TWL_root_inv
            new_rot = rotmat_to_rot6d(new_base2cam[:3,:3])
            new_trans = new_base2cam[:3,3]
            return self.get_rendered_mask_single_image(new_rot, new_trans, robot_mesh, robot_renderer_gpu)
        
    def get_textured_rendering(self, joint, rot, trans, intrinsics=(320, 320, 320, 240), save_path=(None,None,None), original_image=None, root=0):
        
        if root != 0:
            rotmat = rot6d_to_rotmat(rot)
            trans = trans.unsqueeze(dim=1)
            pad = torch.zeros((1,4),dtype=torch.float)
            base2cam = torch.cat([rotmat,trans],dim=1)
            base2cam = torch.cat([base2cam,pad],dim=0)
            base2cam[3,3] = 1.0
            TWL_base = self.get_TWL(joint.unsqueeze(0))
            TWL_root_inv = torch.linalg.inv(TWL_base[:,root:root+1,:,:]).squeeze()
            new_base2cam = base2cam @ TWL_root_inv
            rot = rotmat_to_rot6d(new_base2cam[:3,:3])
            trans = new_base2cam[:3,3]
        
        save_path1, save_path2, save_path3 = save_path
        rotmat = rot6d_to_rotmat(rot)
        trans = trans.unsqueeze(dim=1)
        pad = torch.zeros((1,4),dtype=torch.float)
        camera_pose = torch.cat([rotmat,trans],dim=1)
        camera_pose = torch.cat([camera_pose,pad],dim=0)
        camera_pose[3,3] = 1.0
        joint = joint.numpy()
        camera_pose = camera_pose.numpy()
        rotation = np.array([[1,0,0,0],
                            [0,-1,0,0],
                            [0,0,-1,0],
                            [0,0,0,1]])
        camera_pose = np.matmul(rotation,camera_pose)
        camera_pose = np.linalg.inv(camera_pose)
        fk = self.robot_visual.visual_trimesh_fk(cfg=joint)
        scene = pyrender.Scene()
        camera = pyrender.IntrinsicsCamera(*intrinsics)
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3)
        light_pose = np.eye(4)
        light_pose[:3, 3] = np.array([0, -1, 1])
        scene.add(light, pose=light_pose)
        light_pose[:3, 3] = np.array([0, 1, 1])
        scene.add(light, pose=light_pose)
        light_pose[:3, 3] = np.array([1, 1, 2])
        scene.add(light, pose=light_pose)
        for tm in fk:
            pose = fk[tm]
            mesh = pyrender.Mesh.from_trimesh(tm, smooth=False)
            scene.add(mesh, pose=pose)
        scene.add(camera, pose=camera_pose)
        scene.add(light, pose=camera_pose)
        renderer = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480)
        color, depth = renderer.render(scene)
        rendered_img = Image.fromarray(np.uint8(color)).convert("RGBA")
        rendered_img.save(save_path1)
        original_img = Image.fromarray(np.transpose(np.uint8(original_image),(1,2,0))).convert("RGBA")
        original_img.save(save_path2)
        blend_ratio = 0.7
        blended_image = Image.blend(original_img, rendered_img, blend_ratio)
        blended_image.save(save_path3)
        
    def get_textured_rendering_individual(self, joint, rot, trans, intrinsics=(320, 320, 320, 240), root=0):
        
        if root != 0:
            rotmat = rot6d_to_rotmat(rot)
            trans = trans.unsqueeze(dim=1)
            pad = torch.zeros((1,4),dtype=torch.float)
            base2cam = torch.cat([rotmat,trans],dim=1)
            base2cam = torch.cat([base2cam,pad],dim=0)
            base2cam[3,3] = 1.0
            TWL_base = self.get_TWL(joint.unsqueeze(0))
            TWL_root_inv = torch.linalg.inv(TWL_base[:,root:root+1,:,:]).squeeze()
            new_base2cam = base2cam @ TWL_root_inv
            rot = rotmat_to_rot6d(new_base2cam[:3,:3])
            trans = new_base2cam[:3,3]
        
        rotmat = rot6d_to_rotmat(rot)
        trans = trans.unsqueeze(dim=1)
        pad = torch.zeros((1,4),dtype=torch.float)
        camera_pose = torch.cat([rotmat,trans],dim=1)
        camera_pose = torch.cat([camera_pose,pad],dim=0)
        camera_pose[3,3] = 1.0
        joint = joint.numpy()
        camera_pose = camera_pose.numpy()
        rotation = np.array([[1,0,0,0],
                            [0,-1,0,0],
                            [0,0,-1,0],
                            [0,0,0,1]])
        camera_pose = np.matmul(rotation,camera_pose)
        camera_pose = np.linalg.inv(camera_pose)
        fk = self.robot_visual.visual_trimesh_fk(cfg=joint)
        scene = pyrender.Scene()
        camera = pyrender.IntrinsicsCamera(*intrinsics)
        # # azure
        # light = pyrender.PointLight(color=[1.5, 1.5, 1.5], intensity=2.6)
        # realsense, kinect
        light = pyrender.PointLight(color=[1.4, 1.4, 1.4], intensity=2.4)
        light_pose = np.eye(4)
        light_pose[:3, 3] = np.array([0, 1, 0])
        scene.add(light, pose=light_pose)
        # orb
        # light = pyrender.PointLight(color=[1.4, 1.4, 1.4], intensity=2.4)
        # light_pose = np.eye(4)
        # light_pose[:3, 3] = np.array([0, -1, 0])
        # scene.add(light, pose=light_pose)
        
        for tm in fk:
            pose = fk[tm]
            mesh = pyrender.Mesh.from_trimesh(tm, smooth=False)
            scene.add(mesh, pose=pose)
        scene.add(camera, pose=camera_pose)
        scene.add(light, pose=camera_pose)
        renderer = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480)
        color, depth = renderer.render(scene)
        rendered_img = Image.fromarray(np.uint8(color))
        return rendered_img