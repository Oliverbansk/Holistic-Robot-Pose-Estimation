import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch


def hnormalized(vector):
    hnormalized_vector = (vector / vector[-1])[:-1]
    return hnormalized_vector

def point_projection_from_3d(camera_K, points):
    corr = zip(camera_K, points)
    projections = [hnormalized(np.matmul(K, loc.T)).T for K,loc in corr]
    projections = np.array(projections)
    return projections

def point_projection_from_3d_tensor(camera_K, points):
    corr = zip(camera_K, points)
    projections = [hnormalized(torch.matmul(K, loc.T)).T for K,loc in corr]
    projections = torch.stack(projections)
    return projections

def invert_T(T):
    R = T[..., :3, :3]
    t = T[..., :3, [-1]]
    R_inv = R.transpose(-2, -1)
    t_inv = - R_inv @ t
    T_inv = T.clone()
    T_inv[..., :3, :3] = R_inv
    T_inv[..., :3, [-1]] = t_inv
    return T_inv

def uvd_to_xyz(uvd_jts, image_size, intrinsic_matrix_inverse, root_trans, depth_factor, return_relative=False):
    
    """
        Adapted from https://github.com/Jeff-sjtu/HybrIK/tree/main/hybrik/models
    """
    
    # intrinsic_param is of the inverse version (inv=True)
    assert uvd_jts.dim() == 3 and uvd_jts.shape[2] == 3, uvd_jts.shape
    uvd_jts_new = uvd_jts.clone()
    assert torch.sum(torch.isnan(uvd_jts)) == 0, ('uvd_jts', uvd_jts)

    # remap uv coordinate to input (256x256) space
    uvd_jts_new[:, :, 0] = (uvd_jts[:, :, 0] + 0.5) * image_size
    uvd_jts_new[:, :, 1] = (uvd_jts[:, :, 1] + 0.5) * image_size
    # remap d to m (depth_factor unit: m)
    uvd_jts_new[:, :, 2] = uvd_jts[:, :, 2] * depth_factor
    assert torch.sum(torch.isnan(uvd_jts_new)) == 0, ('uvd_jts_new', uvd_jts_new)

    dz = uvd_jts_new[:, :, 2].cuda()

    # transform uv coordinate to x/z y/z coordinate
    uv_homo_jts = torch.cat((uvd_jts_new[:, :, :2], torch.ones_like(uvd_jts_new)[:, :, 2:]), dim=2).cuda()
    device = intrinsic_matrix_inverse.device
    uv_homo_jts = uv_homo_jts.to(device)
    # batch-wise matrix multipy : (B,1,3,3) * (B,K,3,1) -> (B,K,3,1)
    xyz_jts = torch.matmul(intrinsic_matrix_inverse.unsqueeze(1), uv_homo_jts.unsqueeze(-1))
    xyz_jts = xyz_jts.squeeze(dim=3).cuda()
    
    # recover absolute z : (B,K) + (B,1)
    abs_z = dz + root_trans[:, 2].unsqueeze(-1).cuda()
    # multipy absolute z : (B,K,3) * (B,K,1)
    xyz_jts = xyz_jts * abs_z.unsqueeze(-1)

    if return_relative:
        # (B,K,3) - (B,1,3)
        xyz_jts = xyz_jts - root_trans.unsqueeze(1).cuda()

    # xyz_jts = xyz_jts / depth_factor.unsqueeze(-1)
    # output xyz unit: m

    return xyz_jts.cuda()
    
    
def xyz_to_uvd(xyz_jts, image_size, intrinsic_matrix, root_trans, depth_factor, return_relative=False):
        
    """
        Adapted from https://github.com/Jeff-sjtu/HybrIK/tree/main/hybrik/models
    """

    assert xyz_jts.dim() == 3 and xyz_jts.shape[2] == 3, xyz_jts.shape
    xyz_jts = xyz_jts.cuda()
    intrinsic_matrix = intrinsic_matrix.cuda()
    root_trans = root_trans.cuda()
    uvd_jts = torch.empty_like(xyz_jts).cuda()
    if return_relative:
        # (B,K,3) - (B,1,3)
        xyz_jts = xyz_jts + root_trans.unsqueeze(1)
    assert torch.sum(torch.isnan(xyz_jts)) == 0, ('xyz_jts', xyz_jts)
    
    # batch-wise matrix multipy : (B,1,3,3) * (B,K,3,1) -> (B,K,3,1)
    uvz_jts = torch.matmul(intrinsic_matrix.unsqueeze(1), xyz_jts.unsqueeze(-1))
    uvz_jts = uvz_jts.squeeze(dim=3)
    
    uv_homo = uvz_jts / uvz_jts[:, :, 2].unsqueeze(-1)
    
    abs_z = xyz_jts[:, :, 2]
    dz = abs_z - root_trans[:, 2].unsqueeze(-1)
    
    uvd_jts[:, :, 2] = dz / depth_factor
    uvd_jts[:, :, 0] = uv_homo[:, :, 0] / float(image_size) - 0.5
    uvd_jts[:, :, 1] = uv_homo[:, :, 1] / float(image_size) - 0.5
    
    assert torch.sum(torch.isnan(uvd_jts)) == 0, ('uvd_jts', uvd_jts)

    return uvd_jts
    
    
def xyz_to_uvd_from_gt2d(xyz_jts, gt_uv_2d, image_size, root_trans, depth_factor, return_relative=False):

    assert xyz_jts.dim() == 3 and xyz_jts.shape[2] == 3, xyz_jts.shape
    assert gt_uv_2d.dim() == 3 and gt_uv_2d.shape[2] == 2, gt_uv_2d.shape
    xyz_jts = xyz_jts.cuda()
    root_trans = root_trans.cuda()
    uvd_jts = torch.empty_like(xyz_jts).cuda()
    if return_relative:
        # (B,K,3) - (B,1,3)
        xyz_jts = xyz_jts + root_trans.unsqueeze(1)
    assert torch.sum(torch.isnan(xyz_jts)) == 0, ('xyz_jts', xyz_jts)
    
    abs_z = xyz_jts[:, :, 2]
    dz = abs_z - root_trans[:, 2].unsqueeze(-1)
    
    uvd_jts[:, :, 2] = dz / depth_factor
    uvd_jts[:, :, 0] = gt_uv_2d[:, :, 0] / float(image_size) - 0.5
    uvd_jts[:, :, 1] = gt_uv_2d[:, :, 1] / float(image_size) - 0.5
    
    assert torch.sum(torch.isnan(uvd_jts)) == 0, ('uvd_jts', uvd_jts)
    
    return uvd_jts

def uvz2xyz_singlepoint(uv, z, K):
    batch_size = uv.shape[0]
    assert uv.shape == (batch_size, 2) and z.shape == (batch_size,1) and K.shape == (batch_size,3,3), (uv.shape, z.shape, K.shape) 
    inv_k = get_intrinsic_matrix_batch((K[:,0,0],K[:,1,1]), (K[:,0,2],K[:,1,2]), bsz=batch_size, inv=True)
    device = inv_k.device
    xy_unnormalized = uv * z
    xyz_transformed = torch.cat([xy_unnormalized, z], dim=1)
    xyz_transformed = xyz_transformed.to(device)
    assert xyz_transformed.shape == (batch_size, 3) and inv_k.shape == (batch_size, 3, 3)
    xyz = torch.matmul(inv_k, xyz_transformed.unsqueeze(-1)).squeeze(-1).cuda()
    return xyz

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