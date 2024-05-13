import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import defaultdict
import cv2
import numpy as np
import torch
from config import LOCAL_DATA_DIR
from lib.dataset.const import (INITIAL_JOINT_ANGLE, INTRINSICS_DICT,
                                    JOINT_NAMES, JOINT_TO_KP)
from lib.dataset.dream import DreamDataset
from lib.dataset.multiepoch_dataloader import MultiEpochDataLoader
from lib.dataset.samplers import PartialSampler, PercentSampler
from lib.models.ctrnet.mask_inference import seg_mask_inference
from lib.models.full_net import get_rootNetwithRegInt_model
from lib.utils.BPnP import BPnP_m3d
from lib.utils.transforms import point_projection_from_3d_tensor
from lib.utils.urdf_robot import URDFRobot
from lib.utils.vis import vis_3dkp_single_view
from lib.utils.geometries import (angle_axis_to_rotation_matrix,
                            compute_geodesic_distance_from_two_matrices,
                            rot6d_to_rotmat, rotmat_to_rot6d)
from lib.utils.metrics import compute_metrics_batch, summary_add_pck
from lib.utils.utils import cast, set_random_seed, create_logger, get_scheduler
from torch.utils.data import DataLoader
from torchnet.meter import AverageValueMeter
from tqdm import tqdm


def train_sim2real(args):
    
    torch.autograd.set_detect_anomaly(True)
    set_random_seed(808)
    
    save_folder, ckpt_folder, log_folder, writer = create_logger(args)
    
    urdf_robot_name = args.urdf_robot_name
    robot = URDFRobot(urdf_robot_name)

    # GPU info 
    device_id = args.device_id
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    
    
    train_ds_names = args.train_ds_names
    test_ds_name_real = [os.path.abspath(LOCAL_DATA_DIR / "dream/real/panda-3cam_azure"),
                         os.path.abspath(LOCAL_DATA_DIR / "dream/real/panda-3cam_kinect360"),
                         os.path.abspath(LOCAL_DATA_DIR / "dream/real/panda-3cam_realsense"),
                         os.path.abspath(LOCAL_DATA_DIR / "dream/real/panda-orb")]
    rootnet_hw = (int(args.rootnet_image_size),int(args.rootnet_image_size))
    other_hw = (int(args.other_image_size),int(args.other_image_size))
    ds_train = DreamDataset(train_ds_names, rootnet_resize_hw=rootnet_hw, other_resize_hw=other_hw, color_jitter=False, rgb_augmentation=False, occlusion_augmentation=False)
    train_sampler = PartialSampler(ds_train, epoch_size=args.epoch_size)
    if args.resample_train:
        train_sampler = PercentSampler(ds_train, epoch_size=args.epoch_size, perc=args.resample_perc)
    ds_iter_train = DataLoader(
        ds_train, sampler=train_sampler, batch_size=args.batch_size, num_workers=args.n_dataloader_workers, pin_memory=True
    )
    ds_iter_train = MultiEpochDataLoader(ds_iter_train)
    
    test_loader_dict = {}
    ds_shorts = ["azure", "kinect", "realsense", "orb"]
    for shorts in ds_shorts:
        if shorts in train_ds_names:
            code_name = shorts
            break
            
    for ds_name, ds_short in zip(test_ds_name_real, ds_shorts):
        ds_test_real = DreamDataset(ds_name, rootnet_resize_hw=rootnet_hw, other_resize_hw=other_hw, 
                                    color_jitter=False, rgb_augmentation=False, occlusion_augmentation=False, 
                                    process_truncation=args.fix_truncation) 
        ds_iter_test_real = DataLoader(
            ds_test_real, batch_size=args.batch_size, num_workers=args.n_dataloader_workers,
        )
        test_loader_dict[ds_short] = ds_iter_test_real
    
    print("len(ds_iter_train): ",len(ds_iter_train))
    for ds_short in ds_shorts:
        print(f"len(ds_iter_test_{ds_short}): ", len(test_loader_dict[ds_short]))

    init_param_dict = {
        "robot_type" : urdf_robot_name,
        "pose_params": INITIAL_JOINT_ANGLE,
        "cam_params": np.eye(4,dtype=float),
        "init_pose_from_mean": True
    }
    if args.use_rootnet_with_reg_int_shared_backbone:
        print("regression and integral shared backbone, with rootnet 2 backbones in total")
        model = get_rootNetwithRegInt_model(init_param_dict, args)
                                     
    seg_net = seg_mask_inference(INTRINSICS_DICT[code_name], code_name)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    if args.pretrained_weight_on_synth is not None:
        print(f"using {args.pretrained_weight_on_synth} as starting point of self-training")
        pretrained_path = os.path.join("experiments" , args.pretrained_weight_on_synth)
        pretrained_checkpoint = torch.load(pretrained_path)
        pretrained_state_dict = pretrained_checkpoint['model_state_dict']
        model.load_state_dict(pretrained_state_dict)
        model.to(device)
        print(f"The pretrained model is from epoch {pretrained_checkpoint['epoch']} and has auc {pretrained_checkpoint['auc_add']}")
    else:
        if not args.resume_run:
            assert 0, "no pretrained_weight_on_synth loaded"
            
    
    curr_max_auc = 0.0
    
    if args.resume_run:
        print(f"resuming experiments of {args.resume_experiment_name}")
        resume_dir =  os.path.join("experiments" , args.resume_experiment_name)
        path = os.path.join(resume_dir, 'ckpt/curr_best_auc(add)_model.pk')
        checkpoint = torch.load(path)
        state_dict = checkpoint['model_state_dict']
        model.load_state_dict(state_dict)
        model.to(device)
        optimizer_dict = checkpoint['optimizer_state_dict']
        optimizer.load_state_dict(optimizer_dict)
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        start_epoch = checkpoint['epoch']+1
        last_epoch = checkpoint['lr_scheduler_last_epoch']
        curr_max_auc = checkpoint["auc_add"] if "auc_add" in checkpoint else checkpoint["auc_add_onreal"] 
    else:
        start_epoch = 0
        last_epoch = -1
    end_epoch = args.n_epochs
    
    lr_scheduler = get_scheduler(args, optimizer, last_epoch)
    
    
    # for loop
    for epoch in range(start_epoch, end_epoch + 1):
        print('In epoch {} ----------------- (script: self-supervised training on real )'.format(epoch + 1))
        
        def farward_loss(args,input_batch, device, model, train=True, save=False, batchid=None, epoch_log=None, vis_step=None, view_ids=None, errors=None):
            
            if train:
                model.train()
            else:
                model.eval() 
                
            for module in model.modules():
                if isinstance(module, torch.nn.BatchNorm2d) or isinstance(module, torch.nn.BatchNorm1d):
                    module.eval()
                    # for param in module.parameters():
                    #     param.requires_grad = False
                        
            dtype = torch.float
            root_images = cast(input_batch["root"]["images"],device).float() / 255.
            root_K = cast(input_batch["root"]["K"],device).float()
            
            reg_images = cast(input_batch["other"]["images"],device).float() / 255.
            other_K = cast(input_batch["other"]["K"],device).float()
            
            images_id = input_batch["image_id"]
            images_original = cast(input_batch["images_original"],device).float() / 255.
            images_original_255 = images_original * 255.
            TCO = cast(input_batch["TCO"],device).float()
            K_original = cast(input_batch["K_original"],device).float()
            gt_jointpose = input_batch["jointpose"]
            gt_keypoints2d_original = cast(input_batch["keypoints_2d_original"],device).float()
            valid_mask = cast(input_batch["valid_mask"],device).float()
            gt_keypoints2d = cast(input_batch["other"]["keypoints_2d"],device).float()
            valid_mask_crop = cast(input_batch["other"]["valid_mask_crop"],device).float()
            gt_keypoints3d = cast(input_batch["other"]["keypoints_3d"],device).float()
            
            batch_size = root_images.shape[0]
            robot_type = urdf_robot_name
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
                rot6D = rotmat_to_rot6d(TCO[n,:3,:3])
                trans = TCO[n,:3,3]
                gt_pose.append(jointpose)
                gt_rot.append(rot6D)
                gt_trans.append(trans)
            gt_pose = torch.stack(gt_pose, 0).to(torch.float32)
            gt_rot = torch.stack(gt_rot, 0).to(torch.float32)
            gt_trans = torch.stack(gt_trans, 0).to(torch.float32)
            
            if "synth" not in train_ds_names:
                world_3d_pts = robot.get_keypoints_only_fk(gt_pose)
                out = BPnP_m3d.apply(gt_keypoints2d_original, world_3d_pts, K_original[0])
                out_rot = angle_axis_to_rotation_matrix(out[:,0:3])[:,:3,:3]
                out_rot = rotmat_to_rot6d(out_rot)
                # print("out_rot - gt_rot:", torch.sum(torch.abs(out_rot - gt_rot)))
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
           
            if args.use_rootnet_with_reg_int_shared_backbone or args.use_rootnet_with_reg_with_int_separate_backbone:
                pred_pose, pred_rot, pred_trans, pred_root_uv, pred_root_depth, \
                    pred_uvd, pred_keypoints3d_int, pred_keypoints3d = model(reg_images, root_images, k_values, K=other_K)
                pred_keypoints2d_reproj_int = point_projection_from_3d_tensor(other_K, pred_keypoints3d_int)
                pred_keypoints2d_reproj = point_projection_from_3d_tensor(other_K, pred_keypoints3d)
            else:
                pred_pose, pred_rot, pred_trans, pred_root_uv, pred_root_depth = model(reg_images, root_images, k_values, K=other_K)
                if args.reference_keypoint_id == 0:
                    pred_keypoints3d = robot.get_keypoints(pred_pose, pred_rot, pred_trans)
                else:
                    pred_keypoints3d = robot.get_keypoints_root(pred_pose,pred_rot,pred_trans,root=args.reference_keypoint_id)
                pred_keypoints2d_reproj = point_projection_from_3d_tensor(other_K, pred_keypoints3d)
                pred_keypoints3d_int, pred_keypoints2d_reproj_int = None, None
                
            if args.known_joint:
                pred_pose = gt_pose.clone()
                
            assert(pred_pose.shape == gt_pose.shape),f"{pred_pose.shape},{gt_pose.shape}"
            assert(pred_rot.shape == gt_root_rot.shape),f"{pred_rot.shape},{gt_root_rot.shape}"
            assert(pred_trans.shape == gt_root_trans.shape),f"{pred_trans.shape},{gt_root_trans.shape}"
            assert(pred_root_uv.shape == gt_root_uv.shape),f"{pred_root_uv.shape},{gt_root_uv.shape}"
            assert(pred_root_depth.shape == gt_root_depth.shape),f"{pred_root_depth.shape},{gt_root_depth.shape}"
            assert(pred_keypoints3d.shape == gt_keypoints3d.shape),f"{pred_keypoints3d.shape},{gt_keypoints3d.shape}"
            assert(pred_keypoints2d_reproj.shape == gt_keypoints2d.shape),f"{pred_keypoints2d_reproj.shape},{gt_keypoints2d.shape}"
            
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
                image_dis3d_avg_int, image_dis2d_avg_int = np.array([0],dtype=np.float), np.array([0],dtype=np.float)
                if pred_keypoints3d_int is not None:
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
                rotation_diff = torch.mean(compute_geodesic_distance_from_two_matrices(rot6d_to_rotmat(pred_rot), rot6d_to_rotmat(gt_rot)))
                metric_dict = {
                    "image_dis3d_avg": image_dis3d_avg, "image_dis2d_avg": image_dis2d_avg, "batch_dis3d_avg": batch_dis3d_avg,
                    "batch_dis2d_avg": batch_dis2d_avg, "batch_l1jointerror_avg": batch_l1jointerror_avg, 
                    "image_l1jointerror_avg": image_l1jointerror_avg, "root_depth_error": root_depth_error, "idx": images_id,
                    "image_dis3d_avg_int" : image_dis3d_avg_int, "image_dis2d_avg_int": image_dis2d_avg_int, "rotation_diff": rotation_diff
                }
                
            # Losses
            if args.joint_individual_weights is not None:
                assert(len(args.joint_individual_weights) == robot.dof)
                joint_individual_weights = cast(torch.FloatTensor(args.joint_individual_weights).reshape(1,-1),device)
                pred_pose = pred_pose * joint_individual_weights
                gt_pose = gt_pose * joint_individual_weights
                
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
                error_root_uv = cast(error_root_uv,device)
                loss_uv = torch.mean(error_root_uv)
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
                error3d = torch.norm(pred_keypoints3d - gt_keypoints3d, dim = 2)
                error3d = cast(error3d,device)
                loss_error3d = torch.mean(error3d)
            else:
                raise(NotImplementedError)

            pred_keypoints2d_reproj = pred_keypoints2d_reproj / torch.tensor([args.image_size, args.image_size], dtype=torch.float).reshape(1,1,2).cuda()
            gt_keypoints2d = gt_keypoints2d / torch.tensor([args.image_size, args.image_size], dtype=torch.float).reshape(1,1,2).cuda()
            if args.kp2d_loss_func == "l2norm":
                error2d_reproj = torch.norm(pred_keypoints2d_reproj - gt_keypoints2d, dim = 2)
                error2d_reproj = cast(error2d_reproj * valid_mask_crop, device)
                loss_error2d = torch.sum(error2d_reproj) / torch.sum(valid_mask_crop != 0)
            else:
                raise(NotImplementedError)
            
            # loss = args.pose_loss_weight * loss_pose + args.rot_loss_weight * loss_rot + \
            #        args.uv_loss_weight * loss_uv + args.depth_loss_weight * loss_depth + args.trans_loss_weight * loss_trans + \
            #        args.kp2d_loss_weight * loss_error2d + args.kp3d_loss_weight * loss_error3d
            # loss_pose, loss_rot, loss_uv, loss_depth, loss_trans, loss_error2d, loss_error3d = \
            #     torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(0)
            loss_dict = {   
                "loss_joint": loss_pose.detach(), "loss_rot": loss_rot.detach(), 
                "loss_uv": loss_uv.detach(), "loss_depth": loss_depth.detach(), "loss_trans": loss_trans.detach(),
                "loss_error2d": loss_error2d.detach(), "loss_error3d": loss_error3d.detach(),
                "loss_mask": torch.tensor([0]), "loss_scale": torch.tensor([0]), "loss_iou": torch.tensor([0]), "loss_error3d_align" : torch.tensor([0])
            }
            
            loss = torch.tensor(0.0, dtype=torch.float, requires_grad=True)
            if train:
                if args.known_joint:
                    pred_pose = gt_pose.clone()
                cpu_renderer = robot.set_robot_renderer(K_original[0], device="cpu")
                gpu_renderer = robot.set_robot_renderer(K_original[0], device="cuda")
                robot_mesh_batch = robot.get_robot_mesh_list(joint_angles=pred_pose, renderer=cpu_renderer)
                criterionBCE = torch.nn.BCELoss()
                mse_sum = torch.nn.MSELoss(reduction='sum')
                mse_mean = torch.nn.MSELoss(reduction='mean')
                seg_masks = seg_net(images_original_255).detach()
                rendered_masks = []
                for i in range(batch_size):
                    rendered_mask = robot.get_rendered_mask_single_image_at_specific_root(pred_pose[i], pred_rot[i], pred_trans[i], robot_mesh_batch[i], gpu_renderer, root=args.reference_keypoint_id)
                    rendered_masks.append(rendered_mask)
                if args.use_view:
                    if batchid < 2:
                        vis_folder = os.path.join(save_folder,  'vis')
                        train_vispath = os.path.join(vis_folder, f"train")
                        os.makedirs(train_vispath, exist_ok=True)
                        ri = rendered_masks[0].reshape(240,320).detach().cpu().numpy() * 255
                        si = seg_masks[0].reshape(240,320).detach().cpu().numpy() * 255
                        # cv2.imwrite(train_vispath+f"/render_epoch{epoch_log}_batch{batchid}.jpg", np.uint8(ri))
                        # cv2.imwrite(train_vispath+f"/segmentation_epoch{epoch_log}_batch{batchid}.jpg", np.uint8(si))
                        stacks = np.zeros((240,320,3),dtype=np.uint8)
                        stacks[:,:,0] = ri
                        stacks[:,:,2] = si
                        cv2.imwrite(train_vispath+f"/stack_epoch{epoch_log}_batch{batchid}.jpg", stacks)
                        image_o = np.transpose(images_original_255[i].detach().cpu().numpy().copy(), (1,2,0))
                        image_o = cv2.cvtColor(image_o, cv2.COLOR_RGB2BGR)
                        image_o = cv2.resize(image_o, (320, 240))
                        cv2.imwrite(train_vispath+f'/origin_epoch{epoch_log}_batch{batchid}.jpg',image_o)
                rendered_masks = torch.cat(rendered_masks, 0)
                if args.mask_loss_func == "mse_mean":
                    loss_mask = mse_mean(rendered_masks, seg_masks.squeeze())
                elif args.mask_loss_func == "bce":
                    loss_mask = criterionBCE(rendered_masks, seg_masks.squeeze())
                elif args.mask_loss_func == "mse_sum":
                    loss_mask = 0.001 * mse_sum(rendered_masks, seg_masks.squeeze()) # cvpr 2023
                else:
                    raise NotImplementedError
                
                seg_masks = seg_masks.squeeze()
        
                intersection = torch.sum(seg_masks * rendered_masks, dim=(1,2))
                seg_area = torch.sum(seg_masks, dim=(1,2))
                render_area = torch.sum(rendered_masks, dim=(1,2))
                union = seg_area + render_area - intersection
                iou = intersection / union
                loss_iou = 1 - torch.mean(iou)
                
                seg_only_area = seg_area - intersection
                render_only_area = render_area - intersection
                scale_ratio = seg_only_area / render_only_area
                ratio_filter = (scale_ratio.detach() > 5.0) | (scale_ratio.detach() < 0.2)
                loss_scale = torch.sum(torch.abs(torch.log(scale_ratio)) * ratio_filter) / (torch.sum(ratio_filter)+1e-9)
                
                align3d_error = torch.norm(pred_keypoints3d - pred_keypoints3d_int, dim = 2)
                align3d_error = cast(align3d_error,device)
                loss_error3d_align = torch.mean(align3d_error)
                
                loss_dict["loss_mask"] = loss_mask
                loss_dict["loss_iou"] = loss_iou
                loss_dict["loss_scale"] = loss_scale
                loss_dict["loss_error3d_align"] = loss_error3d_align

                loss = args.mask_loss_weight * loss_mask + args.iou_loss_weight * loss_iou + args.scale_loss_weight * loss_scale + args.align_3d_loss_weight * loss_error3d_align
                # print(loss, loss_mask, loss_iou, loss_scale, loss_error3d_align)
            
            if args.use_view and save:
                # if batchid is not None and batchid % vis_step == 0:
                #     img_errors = torch.mean(error3d, dim=1)
                #     assert len(img_errors.shape) == 1 and img_errors.shape[0] == batch_size
                #     max_index = torch.argmax(img_errors)
                #     vispath = os.path.join(vis_folder, f"{int(batchid//40)}")
                #     os.makedirs(vispath, exist_ok=True)
                #     cpu_renderer = robot.set_robot_renderer(K_original[0], device="cpu")
                #     gpu_renderer = robot.set_robot_renderer(K_original[0], device="cuda")
                #     robot_mesh_batch = robot.get_robot_mesh_list(joint_angles=pred_pose, renderer=cpu_renderer)
                #     seg_masks = seg_net(images_original_255).detach()
                #     rendered_mask = robot.get_rendered_mask_single_image_at_specific_root(pred_pose[max_index], pred_rot[max_index], pred_trans[max_index], robot_mesh_batch[max_index], gpu_renderer, root=args.reference_keypoint_id)
                #     if epoch_log == 0:
                #         image_o = np.transpose(images_original_255[max_index].detach().cpu().numpy().copy(), (1,2,0))
                #         image_o = cv2.cvtColor(image_o, cv2.COLOR_RGB2BGR)
                #         image_o = cv2.resize(image_o, (320, 240))
                #         cv2.imwrite(vispath + f'/origin.jpg',image_o)
                #         si = seg_masks[max_index].reshape(240,320).detach().cpu().numpy() * 255
                #         cv2.imwrite(vispath + f'/segmentation.jpg', np.uint8(si))
                #     ri = rendered_mask.reshape(240,320).detach().cpu().numpy() * 255
                #     si = seg_masks[max_index].reshape(240,320).detach().cpu().numpy() * 255
                #     cv2.imwrite(vispath + f'/render{epoch_log}.jpg', np.uint8(ri))
                #     stacks = np.zeros((240,320,3),dtype=np.uint8)
                #     stacks[:,:,0] = ri
                #     stacks[:,:,2] = si
                #     cv2.imwrite(vispath + f'/stack{epoch_log}.jpg', stacks)
                    
                if view_ids is not None:
                    view_batch_ids = list(np.array(view_ids, dtype=int)//args.batch_size)
                    if batchid in view_batch_ids:
                        for pt in range(len(images_id)):
                            pid = images_id[pt].item()
                            if pid in view_ids:
                                index = view_ids.index(pid)
                                vispath = os.path.join(vis_folder, f"{int(index+1)}")
                                os.makedirs(vispath, exist_ok=True)
                                cpu_renderer = robot.set_robot_renderer(K_original[0], device="cpu")
                                gpu_renderer = robot.set_robot_renderer(K_original[0], device="cuda")
                                robot_mesh_batch = robot.get_robot_mesh_list(joint_angles=pred_pose, renderer=cpu_renderer)
                                seg_masks = seg_net(images_original_255).detach()
                                rendered_mask = robot.get_rendered_mask_single_image_at_specific_root(pred_pose[pt], pred_rot[pt], pred_trans[pt], robot_mesh_batch[pt], gpu_renderer, root=args.reference_keypoint_id)
                                if epoch_log == 0:
                                    image_o = np.transpose(images_original_255[pt].detach().cpu().numpy().copy(), (1,2,0))
                                    image_o = cv2.cvtColor(image_o, cv2.COLOR_RGB2BGR)
                                    image_o = cv2.resize(image_o, (320, 240))
                                    cv2.imwrite(vispath + f'/origin.jpg',image_o)
                                    si = seg_masks[pt].reshape(240,320).detach().cpu().numpy() * 255
                                    cv2.imwrite(vispath + f'/segmentation.jpg', np.uint8(si))
                                ri = rendered_mask.reshape(240,320).detach().cpu().numpy() * 255
                                si = seg_masks[pt].reshape(240,320).detach().cpu().numpy() * 255
                                cv2.imwrite(vispath + f'/render{epoch_log}.jpg', np.uint8(ri))
                                stacks = np.zeros((240,320,3),dtype=np.uint8)
                                stacks[:,:,0] = ri
                                stacks[:,:,2] = si
                                cv2.imwrite(vispath + f'/stack{epoch_log}.jpg', stacks)
                                error_val = errors[index]
                                vis_3dkp_single_view(pred_keypoints3d[pt], gt_keypoints3d[pt], (vispath + f'/vis3da{epoch_log}.jpg'), 12, -20, error_val=error_val)
                                vis_3dkp_single_view(pred_keypoints3d[pt], gt_keypoints3d[pt], (vispath + f'/vis3db{epoch_log}.jpg'), 12, 0)
                                vis_3dkp_single_view(pred_keypoints3d[pt], gt_keypoints3d[pt], (vispath + f'/vis3dc{epoch_log}.jpg'), 12, 20)
                        
            if train:
                return loss, loss_dict
            else:
                return loss, loss_dict, metric_dict

        def validate(ds, epoch_log, get_lowest=False, view_ids=None, errors=None):
            to_save = True if ds in train_ds_names else False 
            if get_lowest:
                assert to_save, ds
            step = 40 if ds == "orb" else 10
            loader = test_loader_dict[ds]
            ds = "_"+ds
            model.eval()
            loss_val = AverageValueMeter()
            losses_pose, losses_rot, losses_trans, losses_uv, losses_depth, losses_error2d, losses_error3d, losses_mask, losses_scale, losses_iou, losses_align = \
                AverageValueMeter(), AverageValueMeter(), AverageValueMeter(), AverageValueMeter(), AverageValueMeter(), AverageValueMeter(), AverageValueMeter(), AverageValueMeter(), AverageValueMeter(), AverageValueMeter(), AverageValueMeter()
            alldis = defaultdict(list)
            alldis_int = defaultdict(list)
            add_thresholds = [1,5,10,20,40,60,80,100]
            pck_thresholds = [2.5,5.0,7.5,10.0,12.5,15.0,17.5,20.0]
            metric_dis3d = [AverageValueMeter() for i in range(len(robot.link_names))]
            metric_dis2d = [AverageValueMeter() for i in range(len(robot.link_names))]
            metric_l1joint = [AverageValueMeter() for i in range(robot.dof)]
            with torch.no_grad():
                for idx, sample in enumerate(tqdm(loader, dynamic_ncols=True)):
                    loss, loss_dict, metric_dict = farward_loss(args=args,input_batch=sample, device=device, model=model, train=False, \
                                                            save=to_save, batchid=idx, epoch_log=epoch_log, vis_step=step, view_ids=view_ids, errors=errors)
                    loss_val.add(loss.detach().cpu().numpy())
                    losses_mask.add(loss_dict["loss_mask"].detach().cpu().numpy())
                    losses_scale.add(loss_dict["loss_scale"].detach().cpu().numpy())
                    losses_iou.add(loss_dict["loss_iou"].detach().cpu().numpy())
                    losses_align.add(loss_dict["loss_error3d_align"].detach().cpu().numpy())
                    losses_pose.add(loss_dict["loss_joint"].detach().cpu().numpy())
                    losses_rot.add(loss_dict["loss_rot"].detach().cpu().numpy())
                    losses_trans.add(loss_dict["loss_trans"].detach().cpu().numpy())
                    losses_depth.add(loss_dict["loss_depth"].detach().cpu().numpy())
                    losses_uv.add(loss_dict["loss_uv"].detach().cpu().numpy())
                    losses_error2d.add(loss_dict["loss_error2d"].detach().cpu().numpy())
                    losses_error3d.add(loss_dict["loss_error3d"].detach().cpu().numpy())
                    alldis["dis3d"].extend(metric_dict["image_dis3d_avg"])
                    alldis["dis2d"].extend(metric_dict["image_dis2d_avg"])
                    alldis_int["dis3d"].extend(metric_dict["image_dis3d_avg_int"])
                    alldis_int["dis2d"].extend(metric_dict["image_dis2d_avg_int"])
                    alldis["jointerror"].extend(metric_dict["image_l1jointerror_avg"])
                    alldis["idx"].extend(list(metric_dict["idx"].numpy()))
                    for id in range(len(robot.link_names)):
                        metric_dis3d[id].add(metric_dict["batch_dis3d_avg"][id])
                        metric_dis2d[id].add(metric_dict["batch_dis2d_avg"][id])
                    for id in range(robot.dof):
                        metric_l1joint[id].add(metric_dict["batch_l1jointerror_avg"][id])
            
            assert len(alldis["idx"]) == len(alldis["dis3d"]), (len(alldis["idx"]),len(alldis["dis3d"]))
            result = [(alldis["dis3d"][i], alldis["idx"][i]) for i in range(len(alldis["idx"]))]
            result_ordered = sorted(result)
            errors_down = [item[0] for item in result_ordered][::-1]
            view_ids_down = [item[1] for item in result_ordered][::-1]
            errors_list = [errors_down[i] for i in np.arange(0,100,5)]
            view_ids_list = [view_ids_down[i] for i in np.arange(0,100,5)]
            print("view_ids_list:", view_ids_list)  
                    
            if get_lowest:
                return view_ids_list, errors_list
            
            summary = summary_add_pck(alldis)
            summary_int = summary_add_pck(alldis_int)
            mean_joint_error = np.mean(alldis["jointerror"]) / np.pi * 180.0  # degree
            writer.add_scalar('Val/loss'+ds, loss_val.mean , epoch_log)
            writer.add_scalar('Val/mask_loss'+ds, losses_mask.mean , epoch_log)
            writer.add_scalar('Val/scale_loss'+ds, losses_scale.mean , epoch_log)
            writer.add_scalar('Val/align_loss'+ds, losses_align.mean , epoch_log)
            writer.add_scalar('Val/iou_loss'+ds, losses_iou.mean , epoch_log)
            writer.add_scalar('Val/pose_loss'+ds, losses_pose.mean , epoch_log)
            writer.add_scalar('Val/rot_loss'+ds, losses_rot.mean , epoch_log)
            writer.add_scalar('Val/trans_loss'+ds, losses_trans.mean , epoch_log)
            writer.add_scalar('Val/uv_loss'+ds, losses_uv.mean , epoch_log)
            writer.add_scalar('Val/depth_loss'+ds, losses_depth.mean , epoch_log)
            writer.add_scalar('Val/error2d_loss'+ds, losses_error2d.mean, epoch_log)
            writer.add_scalar('Val/error3d_loss'+ds, losses_error3d.mean, epoch_log)
            writer.add_scalar('Val/mean_joint_error'+ds, mean_joint_error, epoch_log)
            writer.add_scalar('Val/AUC_ADD'+ds, summary['ADD/AUC'], epoch_log)
            writer.add_scalar('Val/AUC_PCK'+ds, summary['PCK/AUC'], epoch_log)
            writer.add_scalar('Val/AUC_ADD_integral_xyz_metrics'+ds, summary_int['ADD/AUC'], epoch)
            writer.add_scalar('Val/AUC_PCK_integral_xyz_metrics'+ds, summary_int['PCK/AUC'], epoch)
            for k in range(len(add_thresholds)):
                writer.add_scalar('Val/ADD_'+str(add_thresholds[k])+'_mm'+ds, summary[f'ADD_{add_thresholds[k]}_mm'] , epoch_log)
            for k in range(len(pck_thresholds)):
                writer.add_scalar('Val/PCK_'+str(pck_thresholds[k])+'_pixel'+ds, summary[f'PCK_{pck_thresholds[k]}_pixel'] , epoch_log)
            for k in range(len(robot.link_names)):
                writer.add_scalar('Val/distance3D_keypoint_'+str(k+1)+ds, metric_dis3d[k].mean , epoch_log)
            for k in range(len(robot.link_names)):
                writer.add_scalar('Val/distance2D_keypoint_'+str(k+1)+ds, metric_dis2d[k].mean , epoch_log)
            for k in range(robot.dof):
                writer.add_scalar('Val/l1error_joint_'+str(k+1)+ds, metric_l1joint[k].mean, epoch_log)
            model.train()
            return summary['ADD/AUC'], loss_val.mean
        
        if epoch == 0 or args.resume_run:
            for ds_short in ds_shorts:
                if ds_short in train_ds_names:
                    print(f"Getting the worst cases of the pretrained model on {ds_short} dataset")
                    view_ids, errors = validate(ds_short, 0, get_lowest=True)
            for ds_short in ds_shorts:
                if ds_short == code_name:
                    auc_add_real, loss_val_real = validate(ds_short, 0, view_ids=view_ids, errors=errors)
                
        model.train()
        iterator = tqdm(ds_iter_train, dynamic_ncols=True)
        losses = AverageValueMeter()
        losses_pose, losses_rot, losses_trans, losses_uv, losses_depth, losses_error2d, losses_error3d, losses_mask, losses_scale, losses_iou, losses_align = \
            AverageValueMeter(),AverageValueMeter(),AverageValueMeter(),AverageValueMeter(),AverageValueMeter(),AverageValueMeter(),AverageValueMeter(),AverageValueMeter(),AverageValueMeter(),AverageValueMeter(),AverageValueMeter()
        for batchid, sample in enumerate(iterator):
            optimizer.zero_grad()
            loss, loss_dict = farward_loss(args=args,input_batch=sample, device=device, model=model, train=True, batchid=batchid, epoch_log=epoch)
            loss.backward()
            if args.clip_gradient is not None:
                clipping_value = args.clip_gradient
                torch.nn.utils.clip_grad_norm_(model.parameters(), clipping_value)
            optimizer.step() 
            losses.add(loss.detach().cpu().numpy())
            losses_mask.add(loss_dict["loss_mask"].detach().cpu().numpy())
            losses_scale.add(loss_dict["loss_scale"].detach().cpu().numpy())
            losses_iou.add(loss_dict["loss_iou"].detach().cpu().numpy())
            losses_align.add(loss_dict["loss_error3d_align"].detach().cpu().numpy())
            losses_pose.add(loss_dict["loss_joint"].detach().cpu().numpy())
            losses_rot.add(loss_dict["loss_rot"].detach().cpu().numpy())
            losses_trans.add(loss_dict["loss_trans"].detach().cpu().numpy())
            losses_uv.add(loss_dict["loss_uv"].detach().cpu().numpy())
            losses_depth.add(loss_dict["loss_depth"].detach().cpu().numpy())
            losses_error2d.add(loss_dict["loss_error2d"].detach().cpu().numpy())
            losses_error3d.add(loss_dict["loss_error3d"].detach().cpu().numpy())

            if (batchid+1) % 10 == 0:    # Every 100 mini-batches/iterations
                writer.add_scalar('Train/loss', losses.mean , epoch * len(ds_iter_train) + batchid + 1)
                writer.add_scalar('Train/mask_loss', losses_mask.mean , epoch * len(ds_iter_train) + batchid + 1)
                writer.add_scalar('Train/scale_loss', losses_scale.mean , epoch * len(ds_iter_train) + batchid + 1)
                writer.add_scalar('Train/iou_loss', losses_iou.mean , epoch * len(ds_iter_train) + batchid + 1)
                writer.add_scalar('Train/align_loss', losses_align.mean , epoch * len(ds_iter_train) + batchid + 1)
                writer.add_scalar('Train/pose_loss', losses_pose.mean , epoch * len(ds_iter_train) + batchid + 1)
                writer.add_scalar('Train/rot_loss', losses_rot.mean , epoch * len(ds_iter_train) + batchid + 1)
                writer.add_scalar('Train/trans_loss', losses_trans.mean , epoch * len(ds_iter_train) + batchid + 1)
                writer.add_scalar('Train/uv_loss', losses_uv.mean , epoch * len(ds_iter_train) + batchid + 1)
                writer.add_scalar('Train/depth_loss', losses_depth.mean , epoch * len(ds_iter_train) + batchid + 1)
                writer.add_scalar('Train/error2d_loss', losses_error2d.mean, epoch * len(ds_iter_train) + batchid + 1)
                writer.add_scalar('Train/error3d_loss', losses_error3d.mean, epoch * len(ds_iter_train) + batchid + 1)
                losses.reset()
                losses_pose.reset()
                losses_rot.reset()
                losses_trans.reset()
                losses_uv.reset()
                losses_depth.reset()
                losses_error2d.reset()
                losses_error3d.reset()
                losses_mask.reset()
                losses_scale.reset()
                losses_iou.reset()
                losses_align.reset()
            writer.add_scalar('LR/learning_rate_opti', optimizer.param_groups[0]['lr'], epoch * len(ds_iter_train) + batchid + 1)
            if len(optimizer.param_groups) > 1:
                for pgid in range(1,len(optimizer.param_groups)):
                    writer.add_scalar(f'LR/learning_rate_opti_{pgid}', optimizer.param_groups[pgid]['lr'], epoch * len(ds_iter_train) + batchid + 1)
        if args.use_schedule:
            lr_scheduler.step()
            
   
        
        auc_add_onreal = 0.0
        for ds_short in ds_shorts:
            if ds_short == code_name:
                auc_add_real, loss_val_real = validate(ds_short, epoch+1, view_ids=view_ids, errors=errors)
                auc_add_onreal = auc_add_real
    
        save_path = os.path.join(ckpt_folder, 'curr_best_auc(add)_model.pk')

        save1 = True
        if os.path.exists(save_path): 
            ckpt = torch.load(save_path)
            if epoch <= ckpt["epoch"]: # prevent better model got covered during cluster rebooting 
                save1 = False
        
        if save1:        
            if auc_add_onreal > curr_max_auc:
                curr_max_auc = auc_add_onreal
                if args.use_schedule:
                    last_epoch = lr_scheduler.last_epoch
                else:
                    last_epoch = -1
                torch.save({
                            'epoch': epoch,
                            'auc_add': auc_add_onreal,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'lr_scheduler_last_epoch':last_epoch,
                            }, save_path)
                  
    print("Training Finished !")
    writer.flush()
