import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import yaml
from lib.config import LOCAL_DATA_DIR
from lib.core.config import make_cfg
from scripts.train_depthnet import train_depthnet
from scripts.train_sim2real import train_sim2real
from scripts.train_sim2real_real import train_sim2real_real
from scripts.train_full import train_full

# off no use
def make_config(args):
    cfg = argparse.ArgumentParser('').parse_args([])
    
    with open(args.config, encoding="utf-8") as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
        
        N_CPUS = int(os.environ.get('N_CPUS', 10))
        N_WORKERS = min(N_CPUS - 2, config["n_dataloader_workers"])
        cfg.no_cuda = config["no_cuda"]
        cfg.device_id = config["device_id"]
        cfg.exp_name = config["exp_name"]
        cfg.config_path = args.config
        
        # Data
        cfg.urdf_robot_name = config["urdf_robot_name"]
        cfg.train_ds_names =  os.path.abspath(LOCAL_DATA_DIR / config["train_ds_names"])
        if "move" in config["train_ds_names"]:
            cfg.train_ds_names = config["train_ds_names"]
        # cfg.val_ds_names = cfg.train_ds_names
        cfg.image_size = config["image_size"]
        cfg.padding = config["padding"] if "padding" in config else False
        cfg.occlu_p = config["occlu_p"] if "occlu_p" in config else 0.5
        cfg.jitter = config["jitter"] if "jitter" in config else True
        cfg.occlusion = config["occlusion"] if "occlusion" in config else True
        cfg.other_aug = config["other_aug"] if "other_aug" in config else True
        
        # Model
        cfg.backbone_name = config["backbone_name"]
        # cfg.integral_backbone_name = config["integral_backbone_name"] if "integral_backbone_name" in config else "resnet34"
        cfg.rootnet_backbone_name = config["rootnet_backbone_name"] if "rootnet_backbone_name" in config else "resnet50"
        cfg.rootnet_image_size = config["rootnet_image_size"] if "rootnet_image_size" in config else (256.0,256.0)
        cfg.other_image_size = config["other_image_size"] if "other_image_size" in config else (320.0,320.0)
        # cfg.split_reg_head = config["split_reg_head"] if "split_reg_head" in config else False
        # cfg.split_type = config["split_type"] if "split_type" in config else "3-full"
        cfg.use_rpmg = config["use_rpmg"] if "use_rpmg" in config else False
        
         # Optimizer
        cfg.lr = float(config["lr"])
        cfg.weight_decay = config["weight_decay"]
        cfg.use_schedule = config["use_schedule"]
        cfg.schedule_type = config["schedule_type"]
        cfg.n_epochs_warmup = config["n_epochs_warmup"]
        cfg.start_decay = config["start_decay"]
        cfg.end_decay = config["end_decay"]
        cfg.final_decay = config["final_decay"]
        cfg.exponent = float(config["exponent"]) if "exponent" in config else 1.0
        cfg.clip_gradient = config["clip_gradient"] if "clip_gradient" in config else None
        cfg.step_decay = config["step_decay"] if "step_decay" in config else 0.1
        cfg.step = config["step"] if "step" in config else 5
        
        # Training
        cfg.batch_size = config["batch_size"]
        cfg.epoch_size = config["epoch_size"]
        cfg.n_epochs = config["n_epochs"]
        cfg.n_dataloader_workers = N_WORKERS
        # cfg.save_epoch_interval = None

        # Method
        # cfg.use_direct_reg_branch = config["use_direct_reg_branch"]
        cfg.n_iter = config["n_iter"]
        cfg.p_dropout = config["p_dropout"] if "p_dropout" in config else 0.5
        cfg.pose_loss_func = config["pose_loss_func"]
        cfg.rot_loss_func = config["rot_loss_func"]
        cfg.trans_loss_func = config["trans_loss_func"]
        cfg.uv_loss_func = config["uv_loss_func"] if "uv_loss_func" in config else "l2norm"
        # cfg.xy_loss_func = config["xy_loss_func"] if "xy_loss_func" in config else "l2norm"
        cfg.depth_loss_func = config["depth_loss_func"] if "depth_loss_func" in config else "l1"
        cfg.kp3d_loss_func = config["kp3d_loss_func"] if "kp3d_loss_func" in config else "l2norm"
        cfg.kp2d_loss_func = config["kp2d_loss_func"] if "kp2d_loss_func" in config else "l2norm"
        cfg.kp3d_int_loss_func = config["kp3d_int_loss_func"] if "kp3d_int_loss_func" in config else "l2norm"
        cfg.kp2d_int_loss_func = config["kp2d_int_loss_func"] if "kp2d_int_loss_func" in config else "l2norm"
        cfg.align_3d_loss_func = config["align_3d_loss_func"] if "align_3d_loss_func" in config else "l2norm"
        cfg.pose_loss_weight = float(config["pose_loss_weight"]) if "pose_loss_weight" in config else 0.0
        cfg.rot_loss_weight = float(config["rot_loss_weight"]) if "rot_loss_weight" in config else 0.0
        cfg.trans_loss_weight = float(config["trans_loss_weight"]) if "trans_loss_weight" in config else 0.0
        cfg.uv_loss_weight = float(config["uv_loss_weight"]) if "uv_loss_weight" in config else 0.0
        # cfg.xy_loss_weight = float(config["xy_loss_weight"]) if "xy_loss_weight" in config else 0.0
        cfg.depth_loss_weight = float(config["depth_loss_weight"]) if "depth_loss_weight" in config else 0.0
        cfg.kp2d_loss_weight = float(config["kp2d_loss_weight"]) if "kp2d_loss_weight" in config else 0.0
        cfg.kp3d_loss_weight = float(config["kp3d_loss_weight"]) if "kp2d_loss_weight" in config else 0.0
        cfg.kp2d_int_loss_weight = float(config["kp2d_int_loss_weight"]) if "kp2d_int_loss_weight" in config else 0.0
        cfg.kp3d_int_loss_weight = float(config["kp3d_int_loss_weight"]) if "kp2d_int_loss_weight" in config else 0.0
        cfg.align_3d_loss_weight = float(config["align_3d_loss_weight"]) if "align_3d_loss_weight" in config else 0.0
        # cfg.error2d_loss_weight = float(config["error2d_loss_weight"]) if "error2d_loss_weight" in config else 0.0
        # cfg.error3d_loss_weight = float(config["error3d_loss_weight"]) if "error3d_loss_weight" in config else 0.0
        # cfg.use_2d_reprojection_loss = config["use_2d_reprojection_loss"] if "use_2d_reprojection_loss" in config else False
        # cfg.use_3d_loss = config["use_3d_loss"] if "use_3d_loss" in config else False
        cfg.joint_individual_weights = config["joint_individual_weights"]
        if cfg.joint_individual_weights == "None":
            cfg.joint_individual_weights = None
        cfg.use_joint_valid_mask = config["use_joint_valid_mask"] if "use_joint_valid_mask" in config else False
        cfg.reg_joint_map = config["reg_joint_map"] if "reg_joint_map" in config else False
        cfg.joint_conv_dim = config["joint_conv_dim"] if "joint_conv_dim" in config else [128,128,128]
        cfg.direct_reg_rot = config["direct_reg_rot"] if "direct_reg_rot" in config else False
        cfg.rot_iterative_matmul = config["rot_iterative_matmul"] if "rot_iterative_matmul" in config else False
        cfg.fix_root = config["fix_root"] if "fix_root" in config else False
        cfg.reg_from_bb_out = config["reg_from_bb_out"] if "reg_from_bb_out" in config else False
        cfg.depth_from_bb_out = config["depth_from_bb_out"] if "depth_from_bb_out" in config else False
        
        # cfg.use_integral_3d_loss = config["use_integral_3d_branch"] if "use_integral_3d_branch" in config else False
        # cfg.use_uvd_3d_loss = config["use_uvd_3d_loss"] if "use_uvd_3d_loss" in config else False
        # cfg.use_xyz_3d_loss = config["use_xyz_3d_loss"] if "use_xyz_3d_loss" in config else False
        # cfg.use_limb_loss = config["use_limb_loss"] if "use_limb_loss" in config else False
        # cfg.use_2d_reproj_loss = config["use_2d_reproj_loss"] if "use_2d_reproj_loss" in config else False
        # cfg.integral_3d_loss_func = config["integral_3d_loss_func"] if "integral_3d_loss_func" in config else "l2norm"
        # cfg.integral_xyz_3d_loss_func = config["integral_xyz_3d_loss_func"] if "integral_xyz_3d_loss_func" in config else "l2norm"
        # cfg.limb_loss_func = config["limb_loss_func"] if "limb_loss_func" in config else "l1"
        # cfg.integral_2d_reproj_loss_func = config["integral_2d_reproj_loss_func"] if "integral_2d_reproj_loss_func" in config else "l2norm"
        # cfg.integral_3d_loss_weight = config["integral_3d_loss_weight"] if "integral_3d_loss_weight" in config else 0.0
        # cfg.integral_xyz_3d_loss_weight = config["integral_xyz_3d_loss_weight"] if "integral_xyz_3d_loss_weight" in config else 0.0
        # cfg.limb_loss_weight = config["limb_loss_weight"] if "limb_loss_weight" in config else 0.0
        # cfg.integral_2d_reproj_loss_weight = config["integral_2d_reproj_loss_weight"] if "integral_2d_reproj_loss_weight" in config else 0.0
        cfg.bbox_3d_shape = config["bbox_3d_shape"] if "bbox_3d_shape" in config else [1300, 1300, 1300]
        cfg.reference_keypoint_id = config["reference_keypoint_id"]
        # cfg.use_pretrained_direct_reg_weights = config["use_pretrained_direct_reg_weights"] if "use_pretrained_direct_reg_weights" in config else False
        # cfg.pretrained_direct_reg_weights_path = config["pretrained_direct_reg_weights_path"] if "pretrained_direct_reg_weights_path" in config else None
        # cfg.freeze_integral = config["freeze_integral"] if "freeze_integral" in config else False
        cfg.rotation_dim = config["rotation_dim"] if "rotation_dim" in config else 6

        cfg.fix_truncation = config["fix_truncation"] if "fix_truncation" in config else False
        cfg.truncation_padding = config["truncation_padding"] if "truncation_padding" in config else [120,120,120,120]
        
        # if "use_gtz" in config.keys():
        #     cfg.use_gtz = config["use_gtz"]
        # else:
        #     cfg.use_gtz = False

        # if "use_angle_root_net_with_integral" in config.keys():
        #     cfg.use_angle_root_net_with_integral = config["use_angle_root_net_with_integral"]
        # else:
        #     cfg.use_angle_root_net_with_integral = False

        cfg.use_rootnet = config["use_rootnet"] if "use_rootnet" in config else False
        cfg.resample = config["resample"] if "resample" in config else False
        cfg.use_origin_bbox = config["use_origin_bbox"] if "use_origin_bbox" in config else False
        cfg.use_extended_bbox = config["use_extended_bbox"] if "use_extended_bbox" in config else False
        cfg.extend_ratio = list(config["extend_ratio"]) if "extend_ratio" in config else [0.2, 0.13]
        cfg.pretrained_rootnet = config["pretrained_rootnet"] if "pretrained_rootnet" in config and config["pretrained_rootnet"] != "None" else None
        cfg.rootnet_depth_loss_weight = config["rootnet_depth_loss_weight"] if "rootnet_depth_loss_weight" in config else 1.0
        cfg.depth_loss_func = config["depth_loss_func"] if "depth_loss_func" in config else "l1"
        cfg.use_rootnet_xy_branch = config["use_rootnet_xy_branch"] if "use_rootnet_xy_branch" in config else False
        cfg.xy_loss_func = config["xy_loss_func"] if "xy_loss_func" in config else "l1"
        cfg.use_offset = config["use_offset"] if "use_offset" in config else False
        # cfg.freeze_rootnet = config["freeze_rootnet"] if "freeze_rootnet" in config else False
        cfg.rootnet_flip = config["rootnet_flip"] if "rootnet_flip" in config else False
        cfg.add_fc = config["add_fc"] if "add_fc" in config else False
        cfg.multi_kp = config["multi_kp"] if "multi_kp" in config else False
        cfg.kps_need_depth = config["kps_need_depth"] if "kps_need_depth" in config else None

        # if "use_rootnet_with_angle" in config.keys():  
        #     cfg.use_rootnet_with_angle = config["use_rootnet_with_angle"]
        # else:
        #     cfg.use_rootnet_with_angle = False

        # if "use_pretrained_integral" in config.keys():
        #     cfg.use_pretrained_integral = config["use_pretrained_integral"]
        #     cfg.pretrained_integral_weights_path = config["pretrained_integral_weights_path"] \
        #                                 if "pretrained_integral_weights_path" in config and config["pretrained_integral_weights_path"] != "None" else None
        # else:
        #     cfg.use_pretrained_integral = False
        #     cfg.pretrained_integral_weights_path = None
 
        # cfg.use_rootnet_with_regression_uv = config["use_rootnet_with_regression_uv"] if "use_rootnet_with_regression_uv" in config else False
        # cfg.use_onlyregression = config["use_onlyregression"] if "use_onlyregression" in config else False
        # cfg.pretrained_onlyregression = config["pretrained_onlyregression"] if "pretrained_onlyregression" in config else None
        # cfg.fine_tune_depth_layer = config["fine_tune_depth_layer"] if "fine_tune_depth_layer" in config else False
        # cfg.tune_rootnet = config["tune_rootnet"] if "tune_rootnet" in config else False
        # cfg.backbone_10percent = config["backbone_10percent"] if "backbone_10percent" in config else False
        
        cfg.use_sim2real = config["use_sim2real"] if "use_sim2real" in config else False
        cfg.use_sim2real_real = config["use_sim2real_real"] if "use_sim2real_real" in config else False
        cfg.pretrained_weight_on_synth = config["pretrained_weight_on_synth"] if "pretrained_weight_on_synth" in config else None
        cfg.mask_loss_func = config["mask_loss_func"] if "mask_loss_func" in config else "mse_mean"
        cfg.use_view = config["use_view"] if "use_view" in config else False
        cfg.mask_loss_weight = config["mask_loss_weight"] if "mask_loss_weight" in config else 0.0
        cfg.scale_loss_weight = config["scale_loss_weight"] if "scale_loss_weight" in config else 0.0
        cfg.iou_loss_weight = config["iou_loss_weight"] if "iou_loss_weight" in config else 0.0
        
        cfg.fix_mask = config["fix_mask"] if "fix_mask" in config else False
        # cfg.fix_rootnet_sim2real = config["fix_rootnet_sim2real"] if "fix_rootnet_sim2real" in config else False
        cfg.known_joint = config["known_joint"] if "known_joint" in config else False
        
        cfg.use_rootnet_with_reg_int_shared_backbone = config["use_rootnet_with_reg_int_shared_backbone"] if "use_rootnet_with_reg_int_shared_backbone" in config else False
        # cfg.use_rootnet_with_reg_with_int_separate_backbone = config["use_rootnet_with_reg_with_int_separate_backbone"] if "use_rootnet_with_reg_with_int_separate_backbone" in config else False
        # cfg.use_rootnet_reg_int_all_shared_backbone = config["use_rootnet_reg_int_all_shared_backbone"] if "use_rootnet_reg_int_all_shared_backbone" in config else False
        
        # Resume
        cfg.resume_run = config["resume_run"]
        cfg.resume_experiment_name = config["resume_experiment_name"]
        # cfg.log_dir = config["log_dir"]

    f.close()
    
    if args.exp_name is not None:
        cfg.exp_name = args.exp_name 
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size 
    if args.gpu is not None:
        cfg.device_id = [int(x) for x in args.gpu]
    if args.n_epochs is not None:
        cfg.n_epochs = args.n_epochs 
    if args.lr is not None:
        cfg.lr = args.lr
    if args.robot is not None:
        cfg.urdf_robot_name is args.robot
    if args.dataset_path is not None:
        cfg.train_ds_names= args.dataset_path
        cfg.val_ds_names = cfg.train_ds_names
    
    return cfg

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training')
    parser.add_argument('--config', '-c', type=str, required=True, default='configs/cfg.yaml', help="hyperparameters path")
    args = parser.parse_args()
    cfg = make_cfg(args)
    
    print("-------------------   config for this experiment   -------------------")
    print(cfg)
    print("----------------------------------------------------------------------")
    
    if cfg.use_rootnet_with_reg_int_shared_backbone:
        print(f"\n pipeline: full network training (JointNet/RotationNet/KeypoinNet/DepthNet) \n")
        train_full(cfg)
    
    elif cfg.use_rootnet:
        print("\n pipeline: training DepthNet only \n")
        train_depthnet(cfg)
        
    elif cfg.use_sim2real:
        print("\n pipeline: self-supervised training on real datasets \n")
        train_sim2real(cfg)
    
    elif cfg.use_sim2real_real:
        print("\n pipeline: self-supervised training on my real datasets \n")
        train_sim2real_real(cfg)
        
        
    
