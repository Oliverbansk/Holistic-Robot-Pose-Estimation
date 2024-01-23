import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml
from lib.config import LOCAL_DATA_DIR
from easydict import EasyDict

def make_default_cfg():
    cfg = EasyDict()
    
    # basic experiment info (must be overwritten)
    cfg.exp_name = "default"
    cfg.config_path = "default"
    
    # training
    cfg.no_cuda = False
    cfg.device_id = 0
    cfg.batch_size = 64
    cfg.epoch_size = 104950 # will get rid of this eventually, but right now let it be
    cfg.n_epochs = 700
    cfg.n_dataloader_workers = int(os.environ.get('N_CPUS', 10)) - 2
    cfg.clip_gradient = 10.0
    
    # data 
    cfg.urdf_robot_name = "panda"
    cfg.train_ds_names = os.path.abspath(LOCAL_DATA_DIR / "panda_synth_train_dr")
    cfg.image_size = 256.0
    
    # augmentation during training
    cfg.jitter = True
    cfg.other_aug = True
    cfg.occlusion = True
    cfg.occlu_p = 0.5
    cfg.padding = False
    cfg.fix_truncation = False
    cfg.truncation_padding = [120,120,120,120]
    cfg.rootnet_flip = False
    
    # pipeline
    cfg.use_rootnet = False
    cfg.use_rootnet_with_reg_int_shared_backbone = False
    cfg.use_sim2real = False
    cfg.use_sim2real_real = False
    cfg.pretrained_rootnet = None
    cfg.pretrained_weight_on_synth = None
    cfg.use_view = False
    cfg.known_joint = False
    
    # optimizer and scheduler
    cfg.lr = 1e-4
    cfg.weight_decay = 0.0
    cfg.use_schedule = False
    cfg.schedule_type = ""
    cfg.n_epochs_warmup = 0
    cfg.start_decay = 100
    cfg.end_decay = 200
    cfg.final_decay = 0.01
    cfg.exponent = 1.0
    cfg.step_decay = 0.1
    cfg.step = 5
    
    # model
    ## basic setting
    cfg.backbone_name = "resnet50"
    cfg.rootnet_backbone_name = "hrnet32"
    cfg.rootnet_image_size = (cfg.image_size, cfg.image_size)
    cfg.other_image_size = (cfg.image_size, cfg.image_size)
    ## Jointnet/RotationNet
    cfg.n_iter = 4
    cfg.p_dropout = 0.5
    cfg.use_rpmg = False
    cfg.reg_joint_map = False
    cfg.joint_conv_dim = []
    cfg.rotation_dim = 6
    cfg.direct_reg_rot = False
    cfg.rot_iterative_matmul = False
    cfg.fix_root = True
    cfg.reg_from_bb_out = False
    cfg.depth_from_bb_out = False
    ## KeypointNet
    cfg.bbox_3d_shape = [1300, 1300, 1300]
    cfg.reference_keypoint_id = 3
    ## DepthNet
    cfg.resample = False
    cfg.use_origin_bbox = False
    cfg.use_extended_bbox = True
    cfg.extend_ratio = [0.2, 0.13]
    cfg.use_offset = False
    cfg.use_rootnet_xy_branch = False
    cfg.add_fc = False
    cfg.multi_kp = False
    cfg.kps_need_depth =  None
    
    # loss
    ## for full network training
    cfg.pose_loss_func = "mse"
    cfg.rot_loss_func = "mse"
    cfg.trans_loss_func = "l2norm"
    cfg.uv_loss_func = "l2norm"
    cfg.depth_loss_func = "l1"
    cfg.kp3d_loss_func = "l2norm"
    cfg.kp2d_loss_func = "l2norm"
    cfg.kp3d_int_loss_func = "l2norm"
    cfg.kp2d_int_loss_func = "l2norm"
    cfg.align_3d_loss_func = "l2norm"
    cfg.pose_loss_weight = 0.0
    cfg.rot_loss_weight = 0.0
    cfg.trans_loss_weight = 0.0
    cfg.uv_loss_weight = 0.0
    cfg.depth_loss_weight = 0.0
    cfg.kp2d_loss_weight = 0.0
    cfg.kp3d_loss_weight = 0.0
    cfg.kp2d_int_loss_weight = 0.0
    cfg.kp3d_int_loss_weight = 0.0
    cfg.align_3d_loss_weight = 0.0
    cfg.joint_individual_weights = None
    cfg.use_joint_valid_mask = False
    cfg.fix_mask = False
    ## for depthnet training
    cfg.rootnet_depth_loss_weight = 1.0
    cfg.depth_loss_func = "l1"
    cfg.xy_loss_func = "l1"
    ## for self-supervised training
    cfg.mask_loss_func = "mse_mean"
    cfg.mask_loss_weight = 0.0
    cfg.scale_loss_weight = 0.0
    cfg.iou_loss_weight = 0.0
    
    # resume
    cfg.resume_run = False
    cfg.resume_experiment_name = "resume_name"
    
    return cfg
    

def make_cfg(args):
    
    cfg = make_default_cfg()
    cfg.config_path = args.config
    
    with open(args.config, encoding="utf-8") as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
        
        for k,v in config.items():
            if k in cfg:
                if k == "n_dataloader_workers":
                    cfg[k] = min(cfg[k], v)
                elif k == "train_ds_names":
                    cfg[k] = os.path.abspath(LOCAL_DATA_DIR / v)
                    if "move" in v:
                        cfg[k] = v
                elif k in ["lr", "exponent"] or k.endswith("loss_weight"):
                    cfg[k] = float(v)
                elif k in ["joint_individual_weights", "pretrained_rootnet", "pretrained_weight_on_synth"]:
                    cfg[k] = None if v == "None" else v
                elif k == "extend_ratio":
                    cfg[k] = list(v)
                else:
                    cfg[k] = v

    f.close()
    
    return cfg