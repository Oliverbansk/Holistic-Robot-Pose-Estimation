import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import defaultdict
import numpy as np
import torch
from lib.dataset.const import (INITIAL_JOINT_ANGLE, JOINT_NAMES,
                                    JOINT_TO_KP)
from lib.utils.BPnP import BPnP_m3d
from lib.utils.geometries import (
    angle_axis_to_rotation_matrix, compute_geodesic_distance_from_two_matrices,
    quat_to_rotmat, rot6d_to_rotmat, rotmat_to_quat, rotmat_to_rot6d)
from lib.utils.metrics import compute_metrics_batch, summary_add_pck
from lib.utils.transforms import point_projection_from_3d_tensor
from lib.utils.utils import cast
from torchnet.meter import AverageValueMeter
from tqdm import tqdm

def farward_loss(args, input_batch, model, robot, device, device_id, train=True):
    
    if train:
        model.train()
    else:
        model.eval() 

    dtype = torch.float
    root_images = cast(input_batch["root"]["images"],device).float() / 255.
    root_K = cast(input_batch["root"]["K"],device).float()

    reg_images = cast(input_batch["other"]["images"],device).float() / 255.
    other_K = cast(input_batch["other"]["K"],device).float()

    TCO = cast(input_batch["TCO"],device).float()
    K_original = cast(input_batch["K_original"],device).float()
    gt_jointpose = input_batch["jointpose"]
    gt_keypoints2d_original = cast(input_batch["keypoints_2d_original"],device).float()
    valid_mask = cast(input_batch["valid_mask"],device).float()
    gt_keypoints2d = cast(input_batch["other"]["keypoints_2d"],device).float()
    valid_mask_crop = cast(input_batch["other"]["valid_mask_crop"],device).float()
    gt_keypoints3d = cast(input_batch["other"]["keypoints_3d"],device).float()

    batch_size = root_images.shape[0]
    robot_type = args.urdf_robot_name
    if args.use_origin_bbox:
        bboxes = cast(input_batch["bbox_strict_bounded_original"], device).float()
    else:
        bboxes = cast(input_batch["root"]["bbox_strict_bounded"], device).float()
    if args.use_extended_bbox:
        bboxes = cast(input_batch["root"]["bbox_gt2d_extended"], device).float()
    gt_pose = []
    gt_rot = []
    gt_trans = []
    for n in range(batch_size):
        jointpose = torch.as_tensor([gt_jointpose[k][n] for k in JOINT_NAMES[robot_type]])
        jointpose = cast(jointpose, device,dtype=dtype)
        rot6d = rotmat_to_rot6d(TCO[n,:3,:3])
        trans = TCO[n,:3,3]
        gt_pose.append(jointpose)
        gt_rot.append(rot6d)
        gt_trans.append(trans)
    gt_pose = torch.stack(gt_pose, 0).to(torch.float32)
    gt_rot = torch.stack(gt_rot, 0).to(torch.float32)
    if args.rotation_dim == 4:
        gt_rot = rotmat_to_quat(TCO[:,:3,:3])
    gt_trans = torch.stack(gt_trans, 0).to(torch.float32)

    if "synth" not in args.train_ds_names:
        world_3d_pts = robot.get_keypoints_only_fk(gt_pose)
        out = BPnP_m3d.apply(gt_keypoints2d_original, world_3d_pts, K_original[0])
        out_rot = angle_axis_to_rotation_matrix(out[:,0:3])[:,:3,:3]
        if args.rotation_dim == 6:
            out_rot = rotmat_to_rot6d(out_rot)
        elif args.rotation_dim == 4:
            out_rot = rotmat_to_quat(out_rot)
        gt_rot = cast(out_rot, device).float()

    if args.reference_keypoint_id == 0: # use robot base as root for heatmap
        gt_root_trans = gt_trans
        gt_root_rot = gt_rot
    else:
        assert(args.reference_keypoint_id < len(robot.link_names)), args.reference_keypoint_id
        gt_root_trans = gt_keypoints3d[:,args.reference_keypoint_id,:]
        gt_root_rot = robot.get_rotation_at_specific_root(gt_pose, gt_rot, gt_trans, root = args.reference_keypoint_id)
    assert(gt_root_trans.shape == (batch_size, 3)), gt_root_trans
    gt_root_depth = gt_root_trans[:,2].unsqueeze(-1)
    gt_root_uv = gt_keypoints2d[:,args.reference_keypoint_id,0:2]

    real_bbox = torch.tensor([1000.0, 1000.0]).to(torch.float32)
    if args.use_origin_bbox:
        fx, fy = K_original[:,0,0], K_original[:,1,1]
    else:
        fx, fy = root_K[:,0,0], root_K[:,1,1]
    if args.use_extended_bbox:
        fx, fy = root_K[:,0,0], root_K[:,1,1]
    assert(fx.shape == fy.shape and fx.shape == (batch_size,)), (fx.shape,fy.shape)
    area = torch.max(torch.abs(bboxes[:,2]-bboxes[:,0]), torch.abs(bboxes[:,3]-bboxes[:,1])) ** 2
    assert(area.shape == (batch_size,)), area.shape
    k_values = torch.tensor([torch.sqrt(fx[n]*fy[n]*real_bbox[0]*real_bbox[1] / area[n]) for n in range(batch_size)]).to(torch.float32)

    model.to(device)
    model.float()
    model = torch.nn.DataParallel(model, device_ids=device_id, output_device=device_id[0])

    joint_to_kp_indice = JOINT_TO_KP[robot_type]
    assert len(joint_to_kp_indice) == robot.dof, (len(joint_to_kp_indice), robot.dof)
    joint_valid_mask = valid_mask[:, joint_to_kp_indice]
    gt_pose_before_mask = gt_pose.clone()

    if args.use_joint_valid_mask:
        assert joint_valid_mask.shape == gt_pose.shape, (joint_valid_mask.shape, gt_pose.shape)
        gt_pose = gt_pose * joint_valid_mask
        mean_joints_const = torch.tensor([INITIAL_JOINT_ANGLE['mean'][robot_type][k] for k in JOINT_NAMES[robot_type]]).unsqueeze(0).float()
        mean_joints = mean_joints_const.expand(batch_size, -1).cuda() * (1 - joint_valid_mask)
        gt_pose = gt_pose + mean_joints
    if args.multi_kp:
        pred_pose, pred_rot, pred_trans, pred_root_uv, pred_root_depth, pred_depths, \
            pred_uvd, pred_keypoints3d_int, pred_keypoints3d_fk = model(reg_images, root_images, k_values, K=other_K)
    else:
        pred_pose, pred_rot, pred_trans, pred_root_uv, pred_root_depth, \
            pred_uvd, pred_keypoints3d_int, pred_keypoints3d_fk = model(reg_images, root_images, k_values, K=other_K)
    pred_keypoints2d_reproj_int = point_projection_from_3d_tensor(other_K, pred_keypoints3d_int)
    pred_keypoints2d_reproj_fk = point_projection_from_3d_tensor(other_K, pred_keypoints3d_fk)

    if args.known_joint:
        pred_pose = gt_pose.clone()

    assert(pred_pose.shape == gt_pose.shape),f"{pred_pose.shape},{gt_pose.shape}"
    assert(pred_rot.shape == gt_rot.shape),f"{pred_rot.shape},{gt_rot.shape}"
    assert(pred_trans.shape == gt_root_trans.shape),f"{pred_trans.shape},{gt_root_trans.shape}"
    assert(pred_root_uv.shape == gt_root_uv.shape),f"{pred_root_uv.shape},{gt_root_uv.shape}"
    assert(pred_root_depth.shape == gt_root_depth.shape),f"{pred_root_depth.shape},{gt_root_depth.shape}"
    assert(pred_keypoints3d_int.shape == gt_keypoints3d.shape),f"{pred_keypoints3d_int.shape},{gt_keypoints3d.shape}"
    assert(pred_keypoints3d_fk.shape == gt_keypoints3d.shape),f"{pred_keypoints3d_fk.shape},{gt_keypoints3d.shape}"
    assert(pred_keypoints2d_reproj_int.shape == gt_keypoints2d.shape),f"{pred_keypoints2d_reproj_int.shape},{gt_keypoints2d.shape}"
    assert(pred_keypoints2d_reproj_fk.shape == gt_keypoints2d.shape),f"{pred_keypoints2d_reproj_fk.shape},{gt_keypoints2d.shape}"

    # metrics during validation
    if not train:
        image_dis3d_avg, image_dis2d_avg, batch_dis3d_avg, batch_dis2d_avg, \
        batch_l1jointerror_avg, image_l1jointerror_avg, root_depth_error, batch_error_relative, _ = compute_metrics_batch(
                                                                        robot=robot,
                                                                        gt_keypoints3d=gt_keypoints3d,
                                                                        gt_keypoints2d=gt_keypoints2d_original,
                                                                        K_original=K_original,
                                                                        gt_joint=gt_pose_before_mask,
                                                                        pred_joint=pred_pose,
                                                                        pred_rot=pred_rot,
                                                                        pred_trans=pred_trans,
                                                                        pred_depth=None,
                                                                        pred_xy=None,
                                                                        pred_xyz_integral=None,
                                                                        reference_keypoint_id=args.reference_keypoint_id
                                                                        )
        image_dis3d_avg_int, image_dis2d_avg_int, batch_dis3d_avg_int, batch_dis2d_avg_int, \
        batch_l1jointerror_avg_int, image_l1jointerror_avg_int, root_depth_error_int, batch_error_relative_int, _ = compute_metrics_batch(
                                                                        robot=robot,
                                                                        gt_keypoints3d=gt_keypoints3d,
                                                                        gt_keypoints2d=gt_keypoints2d_original,
                                                                        K_original=K_original,
                                                                        gt_joint=gt_pose_before_mask,
                                                                        pred_joint=None,
                                                                        pred_rot=None,
                                                                        pred_trans=None,
                                                                        pred_depth=None,
                                                                        pred_xy=None,
                                                                        pred_xyz_integral=pred_keypoints3d_int,
                                                                        reference_keypoint_id=args.reference_keypoint_id
                                                                        )
        if args.rotation_dim == 6:
            rotation_diff = torch.mean(compute_geodesic_distance_from_two_matrices(rot6d_to_rotmat(pred_rot), rot6d_to_rotmat(gt_rot)))
        elif args.rotation_dim == 4:
            rotation_diff = torch.mean(compute_geodesic_distance_from_two_matrices(quat_to_rotmat(pred_rot), quat_to_rotmat(gt_rot)))
        metric_dict = {
            "image_dis3d_avg": image_dis3d_avg, "image_dis2d_avg": image_dis2d_avg, "batch_dis3d_avg": batch_dis3d_avg,
            "batch_dis2d_avg": batch_dis2d_avg, "batch_l1jointerror_avg": batch_l1jointerror_avg, 
            "image_l1jointerror_avg": image_l1jointerror_avg, "root_depth_error": root_depth_error,
            "image_dis3d_avg_int" : image_dis3d_avg_int, "image_dis2d_avg_int": image_dis2d_avg_int, "batch_dis3d_avg_int": batch_dis3d_avg_int,
            "batch_dis2d_avg_int": batch_dis2d_avg_int, "root_depth_error_int": root_depth_error_int, "rotation_diff": rotation_diff
        }
        
    # Losses
    if args.joint_individual_weights is not None:
        assert(len(args.joint_individual_weights) == robot.dof)
        joint_individual_weights = cast(torch.FloatTensor(args.joint_individual_weights).reshape(1,-1),device)
        pred_pose = pred_pose * joint_individual_weights
        gt_pose = gt_pose * joint_individual_weights

    if args.known_joint:
        pred_pose = gt_pose.clone()
        
    MSELoss = torch.nn.MSELoss()
    SmoothL1Loss = torch.nn.SmoothL1Loss()
    L1Loss = torch.nn.L1Loss()

    if args.pose_loss_func == "smoothl1":
        loss_pose = SmoothL1Loss(pred_pose , gt_pose)
    elif args.pose_loss_func == "l1":
        loss_pose = L1Loss(pred_pose , gt_pose)
    elif args.pose_loss_func == "mse":
        loss_pose = MSELoss(pred_pose , gt_pose)
    else:
        raise(NotImplementedError)
        
    if args.rot_loss_func == "smoothl1":
        loss_rot = SmoothL1Loss(pred_rot , gt_root_rot)
    elif args.rot_loss_func == "l1":
        loss_rot = L1Loss(pred_rot , gt_root_rot)
    elif args.rot_loss_func == "mse":
        loss_rot = MSELoss(pred_rot , gt_root_rot)
    elif args.rot_loss_func == "mat_mse":
        loss_rot = MSELoss(rot6d_to_rotmat(pred_rot) , rot6d_to_rotmat(gt_root_rot))
    else:
        raise(NotImplementedError)

    if args.depth_loss_func == "smoothl1":
        loss_depth = SmoothL1Loss(pred_root_depth , gt_root_depth)
    elif args.depth_loss_func == "l1":
        loss_depth = L1Loss(pred_root_depth , gt_root_depth)
    elif args.depth_loss_func == "mse":
        loss_depth = MSELoss(pred_root_depth , gt_root_depth)
    else:
        raise(NotImplementedError)

    if args.uv_loss_func == "smoothl1":
        loss_uv = SmoothL1Loss(pred_root_uv/args.image_size , gt_root_uv/args.image_size)
    elif args.uv_loss_func == "l1":
        loss_uv = L1Loss(pred_root_uv/args.image_size , gt_root_uv/args.image_size)
    elif args.uv_loss_func == "mse":
        loss_uv = MSELoss(pred_root_uv/args.image_size , gt_root_uv/args.image_size)
    elif args.uv_loss_func == "l2norm":
        error_root_uv = torch.norm((pred_root_uv - gt_root_uv)/args.image_size, dim = 1)
        # error_root_uv = cast(error_root_uv,device)
        # loss_uv = torch.mean(error_root_uv)
        error_root_uv = cast(error_root_uv,device) * valid_mask_crop[:,args.reference_keypoint_id]
        loss_uv = torch.sum(error_root_uv) / torch.sum(valid_mask_crop[:,args.reference_keypoint_id] != 0)
    else:
        raise(NotImplementedError)

    if args.trans_loss_func == "smoothl1":
        loss_trans = SmoothL1Loss(pred_trans, gt_root_trans)
    elif args.trans_loss_func == "l1":
        loss_trans = L1Loss(pred_trans, gt_root_trans)
    elif args.trans_loss_func == "mse":
        loss_trans = MSELoss(pred_trans, gt_root_trans)
    elif args.trans_loss_func == "l2norm":
        error_root_trans = torch.norm(pred_trans - gt_root_trans, dim = 1)
        error_root_trans = cast(error_root_trans,device)
        loss_trans = torch.mean(error_root_trans)
        if loss_trans > 5e-1:
            coeff = torch.exp(- 20.0 * error_root_trans).detach()
            error_root_trans = error_root_trans * coeff
            loss_trans = torch.mean(error_root_trans)
    else:
        raise(NotImplementedError)

    if args.kp3d_loss_func == "l2norm":
        error3d = torch.norm(pred_keypoints3d_fk - gt_keypoints3d, dim = 2)
        error3d = cast(error3d,device)
        loss_error3d = torch.mean(error3d)
    else:
        raise(NotImplementedError)

    pred_keypoints2d_reproj_fk = pred_keypoints2d_reproj_fk / torch.tensor([args.image_size, args.image_size], dtype=torch.float).reshape(1,1,2).cuda()
    gt_keypoints2d_normalized = gt_keypoints2d.clone() / torch.tensor([args.image_size, args.image_size], dtype=torch.float).reshape(1,1,2).cuda()
    if args.kp2d_loss_func == "l2norm":
        error2d_reproj = torch.norm(pred_keypoints2d_reproj_fk - gt_keypoints2d_normalized, dim = 2)
        error2d_reproj = cast(error2d_reproj * valid_mask_crop, device)
        loss_error2d = torch.sum(error2d_reproj) / torch.sum(valid_mask_crop != 0)
    else:
        raise(NotImplementedError)

    if args.kp3d_int_loss_func == "l2norm":
        error3d_int = torch.norm(pred_keypoints3d_int - gt_keypoints3d, dim = 2)
        error3d_int = cast(error3d_int,device)
        loss_error3d_int = torch.mean(error3d_int)
        if args.fix_mask:
            error3d_int = cast(error3d_int * valid_mask_crop, device)
            loss_error3d_int = torch.sum(error3d_int) / torch.sum(valid_mask_crop != 0)
    else:
        raise(NotImplementedError)

    pred_keypoints2d_reproj_int = pred_keypoints2d_reproj_int / torch.tensor([args.image_size, args.image_size], dtype=torch.float).reshape(1,1,2).cuda()
    if args.kp2d_int_loss_func == "l2norm":
        error2d_reproj_int = torch.norm(pred_keypoints2d_reproj_int - gt_keypoints2d_normalized, dim = 2)
        error2d_reproj_int = cast(error2d_reproj_int * valid_mask_crop, device)
        loss_error2d_int = torch.sum(error2d_reproj_int) / torch.sum(valid_mask_crop != 0)
    else:
        raise(NotImplementedError)

    if args.align_3d_loss_func == "l2norm":
        align3d_error = torch.norm(pred_keypoints3d_fk - pred_keypoints3d_int, dim = 2)
        align3d_error = cast(align3d_error,device)
        loss_error3d_align = torch.mean(align3d_error)
        if args.fix_mask:
            align3d_error = cast(align3d_error * valid_mask_crop, device)
            loss_error3d_align = torch.sum(align3d_error) / torch.sum(valid_mask_crop != 0)
    else:
        raise(NotImplementedError)

    if args.multi_kp:
        gt_kp_depths = gt_keypoints3d[:,args.kps_need_depth,2]
        assert gt_kp_depths.shape == pred_depths.shape, (gt_kp_depths.shape, pred_depths.shape)
        loss_depth_multi = L1Loss(pred_depths , gt_kp_depths)
    else:
        loss_depth_multi = torch.tensor([0])

    loss = args.pose_loss_weight * loss_pose + args.rot_loss_weight * loss_rot + \
            args.uv_loss_weight * loss_uv + args.depth_loss_weight * loss_depth + args.trans_loss_weight * loss_trans + \
            args.kp2d_loss_weight * loss_error2d + args.kp3d_loss_weight * loss_error3d + \
            args.kp2d_int_loss_weight * loss_error2d_int + args.kp3d_int_loss_weight * loss_error3d_int + \
            args.align_3d_loss_weight * loss_error3d_align
            
    if args.multi_kp:
        loss += loss_depth_multi

    loss_dict = {   
        "loss_joint": loss_pose, "loss_rot": loss_rot, 
        "loss_uv": loss_uv, "loss_depth": loss_depth, "loss_trans": loss_trans,
        "loss_error2d": loss_error2d, "loss_error3d": loss_error3d,
        "loss_error2d_int": loss_error2d_int, "loss_error3d_int": loss_error3d_int,
        "loss_error3d_align": loss_error3d_align
    }

    if train:
        return loss, loss_dict
    else:
        return loss, loss_dict, metric_dict
    
    
def validate(args, epoch, dsname, loader, model, robot, writer, device, device_id):
    assert (ds != "photo" or args.urdf_robot_name != "baxter") # ds == photo -> not baxter
    assert (ds in ["dr", "photo"] or args.urdf_robot_name == "panda") # ds != dr/photo -> panda
    
    # loader = test_loader_dict[ds]
    ds = "_"+dsname
    model.eval()
    loss_val = AverageValueMeter()
    losses_pose, losses_rot, losses_trans, losses_uv, losses_depth, losses_error2d, losses_error3d, losses_error2d_int, losses_error3d_int, losses_error3d_align, rotation_diff = \
        AverageValueMeter(), AverageValueMeter(), AverageValueMeter(), AverageValueMeter(), AverageValueMeter(), AverageValueMeter(), AverageValueMeter(), AverageValueMeter(), AverageValueMeter(), AverageValueMeter(), AverageValueMeter()
    alldis = defaultdict(list)
    alldis_int = defaultdict(list)
    add_thresholds = [1,5,10,20,40,60,80,100]
    pck_thresholds = [2.5,5.0,7.5,10.0,12.5,15.0,17.5,20.0]
    metric_dis3d = [AverageValueMeter() for i in range(len(robot.link_names))]
    metric_dis2d = [AverageValueMeter() for i in range(len(robot.link_names))]
    metric_dis3d_int = [AverageValueMeter() for i in range(len(robot.link_names))]
    metric_dis2d_int = [AverageValueMeter() for i in range(len(robot.link_names))]
    metric_l1joint = [AverageValueMeter() for i in range(robot.dof)]
    with torch.no_grad():
        for idx, sample in enumerate(tqdm(loader, dynamic_ncols=True)):
            vloss, loss_dict, metric_dict = farward_loss(args=args, input_batch=sample, model=model, robot=robot, device=device, device_id=device_id, train=False)
            loss_val.add(vloss.detach().cpu().numpy())
            losses_pose.add(loss_dict["loss_joint"].detach().cpu().numpy())
            losses_rot.add(loss_dict["loss_rot"].detach().cpu().numpy())
            losses_trans.add(loss_dict["loss_trans"].detach().cpu().numpy())
            losses_depth.add(loss_dict["loss_depth"].detach().cpu().numpy())
            losses_uv.add(loss_dict["loss_uv"].detach().cpu().numpy())
            losses_error2d.add(loss_dict["loss_error2d"].detach().cpu().numpy())
            losses_error3d.add(loss_dict["loss_error3d"].detach().cpu().numpy())
            losses_error2d_int.add(loss_dict["loss_error2d_int"].detach().cpu().numpy())
            losses_error3d_int.add(loss_dict["loss_error3d_int"].detach().cpu().numpy())
            losses_error3d_align.add(loss_dict["loss_error3d_align"].detach().cpu().numpy())
            alldis["dis3d"].extend(metric_dict["image_dis3d_avg"])
            alldis["dis2d"].extend(metric_dict["image_dis2d_avg"])
            alldis["jointerror"].extend(metric_dict["image_l1jointerror_avg"])
            alldis_int["dis3d"].extend(metric_dict["image_dis3d_avg_int"])
            alldis_int["dis2d"].extend(metric_dict["image_dis2d_avg_int"])
            rotation_diff.add(metric_dict["rotation_diff"].detach().cpu().numpy())
            for id in range(len(robot.link_names)):
                metric_dis3d[id].add(metric_dict["batch_dis3d_avg"][id])
                metric_dis2d[id].add(metric_dict["batch_dis2d_avg"][id])
            for id in range(len(robot.link_names)):
                metric_dis3d_int[id].add(metric_dict["batch_dis3d_avg_int"][id])
                metric_dis2d_int[id].add(metric_dict["batch_dis2d_avg_int"][id])
            for id in range(robot.dof):
                metric_l1joint[id].add(metric_dict["batch_l1jointerror_avg"][id])
                
    summary = summary_add_pck(alldis)
    summary_int = summary_add_pck(alldis_int)
    mean_joint_error = np.mean(alldis["jointerror"]) / np.pi * 180.0  # degree
    writer.add_scalar('Val/loss'+ds, loss_val.mean , epoch)
    writer.add_scalar('Val/pose_loss'+ds, losses_pose.mean , epoch)
    writer.add_scalar('Val/rot_loss'+ds, losses_rot.mean , epoch)
    writer.add_scalar('Val/rot_diff'+ds, rotation_diff.mean , epoch)
    writer.add_scalar('Val/trans_loss'+ds, losses_trans.mean , epoch)
    writer.add_scalar('Val/uv_loss'+ds, losses_uv.mean , epoch)
    writer.add_scalar('Val/depth_loss'+ds, losses_depth.mean , epoch)
    writer.add_scalar('Val/error2d_loss'+ds, losses_error2d.mean, epoch)
    writer.add_scalar('Val/error3d_loss'+ds, losses_error3d.mean, epoch)
    writer.add_scalar('Val/error2d_int_loss'+ds, losses_error2d.mean, epoch)
    writer.add_scalar('Val/error3d_int_loss'+ds, losses_error3d.mean, epoch)
    writer.add_scalar('Val/error3d_align_loss'+ds, losses_error3d_align.mean, epoch)
    writer.add_scalar('Val/mean_joint_error'+ds, mean_joint_error, epoch)
    writer.add_scalar('Val/AUC_ADD'+ds, summary['ADD/AUC'], epoch)
    writer.add_scalar('Val/AUC_PCK'+ds, summary['PCK/AUC'], epoch)
    writer.add_scalar('Val/AUC_ADD_integral_xyz_metrics'+ds, summary_int['ADD/AUC'], epoch)
    writer.add_scalar('Val/AUC_PCK_integral_xyz_metrics'+ds, summary_int['PCK/AUC'], epoch)
    for k in range(len(add_thresholds)):
        writer.add_scalar('Val/ADD_'+str(add_thresholds[k])+'_mm'+ds, summary[f'ADD_{add_thresholds[k]}_mm'] , epoch)
    for k in range(len(pck_thresholds)):
        writer.add_scalar('Val/PCK_'+str(pck_thresholds[k])+'_pixel'+ds, summary[f'PCK_{pck_thresholds[k]}_pixel'] , epoch)
    for k in range(len(add_thresholds)):
        writer.add_scalar('Val/ADD_'+str(add_thresholds[k])+'_mm'+"_integral_xyz_metrics"+ds, summary_int[f'ADD_{add_thresholds[k]}_mm'] , epoch)
    for k in range(len(pck_thresholds)):
        writer.add_scalar('Val/PCK_'+str(pck_thresholds[k])+'_pixel'+"_integral_xyz_metrics"+ds, summary_int[f'PCK_{pck_thresholds[k]}_pixel'] , epoch)
    for k in range(len(robot.link_names)):
        writer.add_scalar('Val/distance3D_keypoint_'+str(k+1)+ds, metric_dis3d[k].mean , epoch)
    for k in range(len(robot.link_names)):
        writer.add_scalar('Val/distance2D_keypoint_'+str(k+1)+ds, metric_dis2d[k].mean , epoch)
    for k in range(len(robot.link_names)):
        writer.add_scalar('Val/distance3D_keypoint_'+str(k+1)+"_integral_xyz_metrics"+ds, metric_dis3d_int[k].mean , epoch)
    for k in range(len(robot.link_names)):
        writer.add_scalar('Val/distance2D_keypoint_'+str(k+1)+"_integral_xyz_metrics"+ds, metric_dis2d_int[k].mean , epoch)
    for k in range(robot.dof):
        writer.add_scalar('Val/l1error_joint_'+str(k+1)+ds, metric_l1joint[k].mean, epoch)
    model.train()
    return summary['ADD/AUC']