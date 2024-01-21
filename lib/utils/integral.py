import torch
import torch.nn as nn
from torch.nn import functional as F
from utils.transforms import uvd_to_xyz


def flip(x):
    assert (x.dim() == 3 or x.dim() == 4)
    dim = x.dim() - 1

    return x.flip(dims=(dim,))

def norm_heatmap_hrnet(norm_type, heatmap, tau=5, sample_num=1):
    # Input tensor shape: [N,C,...]
    shape = heatmap.shape
    if norm_type == 'softmax':
        heatmap = heatmap.reshape(*shape[:2], -1)
        # global soft max
        heatmap = F.softmax(heatmap, 2)
        return heatmap.reshape(*shape)
    elif norm_type == 'sampling':
        heatmap = heatmap.reshape(*shape[:2], -1)

        eps = torch.rand_like(heatmap)
        log_eps = torch.log(-torch.log(eps))
        gumbel_heatmap = heatmap - log_eps / tau

        gumbel_heatmap = F.softmax(gumbel_heatmap, 2)
        return gumbel_heatmap.reshape(*shape)
    elif norm_type == 'multiple_sampling':

        heatmap = heatmap.reshape(*shape[:2], 1, -1)

        eps = torch.rand(*heatmap.shape[:2], sample_num, heatmap.shape[3], device=heatmap.device)
        log_eps = torch.log(-torch.log(eps))
        gumbel_heatmap = heatmap - log_eps / tau
        gumbel_heatmap = F.softmax(gumbel_heatmap, 3)
        gumbel_heatmap = gumbel_heatmap.reshape(shape[0], shape[1], sample_num, shape[2])

        # [B, S, K, -1]
        return gumbel_heatmap.transpose(1, 2)
    else:
        raise NotImplementedError
    
def norm_heatmap_resnet(norm_type, heatmap):
    # Input tensor shape: [N,C,...]
    shape = heatmap.shape
    if norm_type == 'softmax':
        heatmap = heatmap.reshape(*shape[:2], -1)
        # global soft max
        heatmap = F.softmax(heatmap, 2)
        return heatmap.reshape(*shape)
    else:
        raise NotImplementedError
    
def get_intrinsic_matrix_batch(f, c, bsz, inv=False):
        
        intrinsic_matrix = torch.zeros((bsz, 3, 3)).to(torch.float)

        if inv:
            intrinsic_matrix[:, 0, 0] = 1.0 / f[0].to(float)
            intrinsic_matrix[:, 0, 2] = - c[0].to(float) / f[0].to(float)
            intrinsic_matrix[:, 1, 1] = 1.0 / f[1].to(float)
            intrinsic_matrix[:, 1, 2] = - c[1].to(float) / f[1].to(float)
            intrinsic_matrix[:, 2, 2] = 1
        else:
            intrinsic_matrix[:, 0, 0] = f[0]
            intrinsic_matrix[:, 0, 2] = c[0]
            intrinsic_matrix[:, 1, 1] = f[1]
            intrinsic_matrix[:, 1, 2] = c[1]
            intrinsic_matrix[:, 2, 2] = 1

        return intrinsic_matrix.cuda(device=0)
    
class HeatmapIntegralPose(nn.Module):
    """
    This module takes in heatmap output and performs soft-argmax(integral operation).
    """
    def __init__(self, backbone, **kwargs):
        super(HeatmapIntegralPose, self).__init__()
        self.backbone_name = backbone
        self.norm_type = kwargs["norm_type"]
        self.num_joints = kwargs["num_joints"]
        self.depth_dim = kwargs["depth_dim"]
        self.height_dim = kwargs["height_dim"]
        self.width_dim = kwargs["width_dim"]
        self.rootid = kwargs["rootid"] if "rootid" in kwargs else 0
        self.fixroot = kwargs["fixroot"] if "fixroot" in kwargs else False
        
        # self.focal_length = kwargs['FOCAL_LENGTH'] if 'FOCAL_LENGTH' in kwargs else 320
        bbox_3d_shape = kwargs['bbox_3d_shape'] if 'bbox_3d_shape' in kwargs else (2300, 2300, 2300)
        self.bbox_3d_shape = torch.tensor(bbox_3d_shape).float()
        self.depth_factor = self.bbox_3d_shape[2] * 1e-3
        self.image_size = kwargs["image_size"]
    
    
    def forward(self, out, flip_test=False, **kwargs):
        """
        Adapted from https://github.com/Jeff-sjtu/HybrIK/tree/main/hybrik/models
        """
        
        K = kwargs["K"]
        root_trans = kwargs["root_trans"]
        batch_size = out.shape[0]
        inv_k = get_intrinsic_matrix_batch((K[:,0,0],K[:,1,1]), (K[:,0,2],K[:,1,2]), bsz=batch_size, inv=True)
        
        if self.backbone_name in ["resnet", "resnet34", "resnet50"]:
            # out = out.reshape(batch_size, self.num_joints, self.depth_dim, self.height_dim, self.width_dim)
            out = out.reshape((out.shape[0], self.num_joints, -1))
            out = norm_heatmap_resnet(self.norm_type, out)
            assert out.dim() == 3, out.shape
            heatmaps = out / out.sum(dim=2, keepdim=True)
            heatmaps = heatmaps.reshape((heatmaps.shape[0], self.num_joints, self.depth_dim, self.height_dim, self.width_dim))
            hm_x0 = heatmaps.sum((2, 3)) # (B, K, W)
            hm_y0 = heatmaps.sum((2, 4)) # (B, K, H)
            hm_z0 = heatmaps.sum((3, 4)) # (B, K, D)

            range_tensor = torch.arange(hm_x0.shape[-1], dtype=torch.float32, device=hm_x0.device)
            
            hm_x = hm_x0 * range_tensor
            hm_y = hm_y0 * range_tensor
            hm_z = hm_z0 * range_tensor

            coord_x = hm_x.sum(dim=2, keepdim=True)
            coord_y = hm_y.sum(dim=2, keepdim=True)
            coord_z = hm_z.sum(dim=2, keepdim=True)
            
            coord_x = coord_x / float(self.width_dim) - 0.5
            coord_y = coord_y / float(self.height_dim) - 0.5
            coord_z = coord_z / float(self.depth_dim) - 0.5

            #  -0.5 ~ 0.5
            pred_uvd_jts = torch.cat((coord_x, coord_y, coord_z), dim=2)  
            if self.fixroot: 
                pred_uvd_jts[:,self.rootid,2] = 0.0
            pred_uvd_jts_flat = pred_uvd_jts.reshape(batch_size, -1)
            
            pred_xyz_jts = uvd_to_xyz(uvd_jts=pred_uvd_jts, image_size=self.image_size, intrinsic_matrix_inverse=inv_k, 
                                           root_trans=root_trans, depth_factor=self.depth_factor, return_relative=False)
            
            # pred_uvd_jts_back = xyz_to_uvd(xyz_jts=pred_xyz_jts, image_size=self.image_size, intrinsic_matrix=K, 
            #                                root_trans=root_trans, depth_factor=self.depth_factor, return_relative=False)
            # print("(pred_uvd_jts-pred_uvd_jts_back).sum()",(pred_uvd_jts.cuda()-pred_uvd_jts_back.cuda()).sum())
            
            return pred_uvd_jts, pred_xyz_jts
        
        elif self.backbone_name == "hrnet" or self.backbone_name == "hrnet32" or self.backbone_name == "hrnet48":
            out = out.reshape((out.shape[0], self.num_joints, -1))
            heatmaps = norm_heatmap_hrnet(self.norm_type, out)
            assert heatmaps.dim() == 3, heatmaps.shape
            heatmaps = heatmaps.reshape((heatmaps.shape[0], self.num_joints, self.depth_dim, self.height_dim, self.width_dim))

            hm_x0 = heatmaps.sum((2, 3))  # (B, K, W)
            hm_y0 = heatmaps.sum((2, 4))  # (B, K, H)
            hm_z0 = heatmaps.sum((3, 4))  # (B, K, D)

            range_tensor = torch.arange(hm_x0.shape[-1], dtype=torch.float32, device=hm_x0.device).unsqueeze(-1)
            # hm_x = hm_x0 * range_tensor
            # hm_y = hm_y0 * range_tensor
            # hm_z = hm_z0 * range_tensor

            # coord_x = hm_x.sum(dim=2, keepdim=True)
            # coord_y = hm_y.sum(dim=2, keepdim=True)
            # coord_z = hm_z.sum(dim=2, keepdim=True)
            coord_x = hm_x0.matmul(range_tensor)
            coord_y = hm_y0.matmul(range_tensor)
            coord_z = hm_z0.matmul(range_tensor)

            coord_x = coord_x / float(self.width_dim) - 0.5
            coord_y = coord_y / float(self.height_dim) - 0.5
            coord_z = coord_z / float(self.depth_dim) - 0.5

            #  -0.5 ~ 0.5
            pred_uvd_jts = torch.cat((coord_x, coord_y, coord_z), dim=2)
            if self.fixroot: 
                pred_uvd_jts[:,self.rootid,2] = 0.0
            pred_uvd_jts_flat = pred_uvd_jts.reshape(batch_size, -1)
            
            pred_xyz_jts = uvd_to_xyz(uvd_jts=pred_uvd_jts, image_size=self.image_size, intrinsic_matrix_inverse=inv_k, 
                                           root_trans=root_trans, depth_factor=self.depth_factor, return_relative=False)
            
            # pred_uvd_jts_back = xyz_to_uvd(xyz_jts=pred_xyz_jts, image_size=self.image_size, intrinsic_matrix=K, 
            #                                root_trans=root_trans, depth_factor=self.depth_factor, return_relative=False)
            # print("(pred_uvd_jts-pred_uvd_jts_back).sum()",(pred_uvd_jts.cuda()-pred_uvd_jts_back.cuda()).sum())
            
            return pred_uvd_jts, pred_xyz_jts
        
        else:
            raise(NotImplementedError)
        
        
class HeatmapIntegralJoint(nn.Module):
    """
    This module takes in heatmap output and performs soft-argmax(integral operation).
    """
    def __init__(self, backbone, **kwargs):
        super(HeatmapIntegralJoint, self).__init__()
        self.backbone_name = backbone
        self.norm_type = kwargs["norm_type"]
        self.dof = kwargs["dof"]
        self.joint_bounds = kwargs["joint_bounds"]
        assert self.joint_bounds.shape == (self.dof, 2), self.joint_bounds.shape
    
    
    def forward(self, out, **kwargs):
        """
        Adapted from https://github.com/Jeff-sjtu/HybrIK/tree/main/hybrik/models
        """
        
        batch_size = out.shape[0]
        
        if self.backbone_name in ["resnet34", "resnet50"]:
            out = out.reshape(batch_size, self.dof, -1)
            out = norm_heatmap_resnet(self.norm_type, out)
            assert out.dim() == 3, out.shape
            heatmaps = out / out.sum(dim=2, keepdim=True)
            heatmaps = heatmaps.reshape((heatmaps.shape[0], self.dof, -1)) # no depth dimension
            
            resolution = heatmaps.shape[-1]
            range_tensor = torch.arange(resolution, dtype=torch.float32, device=heatmaps.device).reshape(1,1,resolution)
            hm_int = heatmaps * range_tensor
            coord_joint_raw = hm_int.sum(dim=2, keepdim=True)
            coord_joint = coord_joint_raw / float(resolution) # 0~1
            
            bounds = self.joint_bounds.reshape(1,self.dof,2).cuda()
            jointrange = bounds[:,:,[1]] - bounds[:,:,[0]]
            joints = coord_joint * jointrange + bounds[:,:,[0]]
            
            return joints.squeeze(-1)
        
        else:
            raise(NotImplementedError)
    