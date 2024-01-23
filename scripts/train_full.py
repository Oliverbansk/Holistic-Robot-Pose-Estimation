import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from lib.core.function import farward_loss, validate
from lib.dataset.const import INITIAL_JOINT_ANGLE
from lib.models.full_net import get_rootNetwithRegInt_model
from lib.utils.urdf_robot import URDFRobot
from lib.utils.utils import set_random_seed, create_logger, get_dataloaders, get_scheduler, resume_run, save_checkpoint
from torchnet.meter import AverageValueMeter
from tqdm import tqdm

    
def train_full(args):
    
    torch.autograd.set_detect_anomaly(True)
    set_random_seed(808)
    
    save_folder, ckpt_folder, log_folder, writer = create_logger(args)
    
    urdf_robot_name = args.urdf_robot_name
    robot = URDFRobot(urdf_robot_name)
 
    device_id = args.device_id
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    ds_iter_train, test_loader_dict = get_dataloaders(args)
    
    init_param_dict = {
        "robot_type" : urdf_robot_name,
        "pose_params": INITIAL_JOINT_ANGLE,
        "cam_params": np.eye(4,dtype=float),
        "init_pose_from_mean": True
    }
    if args.use_rootnet_with_reg_int_shared_backbone:
        print("regression and integral shared backbone, with rootnet 2 backbones in total")
        model = get_rootNetwithRegInt_model(init_param_dict, args)
    else:
        assert 0
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    curr_max_auc = 0.0
    curr_max_auc_4real = { "azure": 0.0, "kinect": 0.0, "realsense": 0.0, "orb": 0.0 }
    start_epoch, last_epoch, end_epoch = 0, -1, args.n_epochs
    if args.resume_run:
        start_epoch, last_epoch, curr_max_auc, curr_max_auc_4real = resume_run(args, model, optimizer, device)
        
    lr_scheduler = get_scheduler(args, optimizer, last_epoch)
 
 
    for epoch in range(start_epoch, end_epoch + 1):
        print('In epoch {},  script: full network training (JointNet/RotationNet/KeypoinNet/DepthNet)'.format(epoch + 1))
        model.train()
        iterator = tqdm(ds_iter_train, dynamic_ncols=True)
        losses = AverageValueMeter()
        losses_pose, losses_rot, losses_trans, losses_uv, losses_depth, losses_error2d, losses_error3d, losses_error2d_int, losses_error3d_int, losses_error3d_align = \
            AverageValueMeter(),AverageValueMeter(),AverageValueMeter(),AverageValueMeter(),AverageValueMeter(),AverageValueMeter(),AverageValueMeter(),AverageValueMeter(),AverageValueMeter(),AverageValueMeter()
        for batchid, sample in enumerate(iterator):
            optimizer.zero_grad()
            loss, loss_dict = farward_loss(args=args, input_batch=sample, model=model, robot=robot, device=device, device_id=device_id, train=True)
            loss.backward()
            if args.clip_gradient is not None:
                clipping_value = args.clip_gradient
                torch.nn.utils.clip_grad_norm_(model.parameters(), clipping_value)
            optimizer.step()
            losses.add(loss.detach().cpu().numpy())
            losses_pose.add(loss_dict["loss_joint"].detach().cpu().numpy())
            losses_rot.add(loss_dict["loss_rot"].detach().cpu().numpy())
            losses_trans.add(loss_dict["loss_trans"].detach().cpu().numpy())
            losses_uv.add(loss_dict["loss_uv"].detach().cpu().numpy())
            losses_depth.add(loss_dict["loss_depth"].detach().cpu().numpy())
            losses_error2d.add(loss_dict["loss_error2d"].detach().cpu().numpy())
            losses_error3d.add(loss_dict["loss_error3d"].detach().cpu().numpy())
            losses_error2d_int.add(loss_dict["loss_error2d_int"].detach().cpu().numpy())
            losses_error3d_int.add(loss_dict["loss_error3d_int"].detach().cpu().numpy())
            losses_error3d_align.add(loss_dict["loss_error3d_align"].detach().cpu().numpy())

            if (batchid+1) % 100 == 0:    # Every 100 mini-batches/iterations
                writer.add_scalar('Train/loss', losses.mean , epoch * len(ds_iter_train) + batchid + 1)
                writer.add_scalar('Train/pose_loss', losses_pose.mean , epoch * len(ds_iter_train) + batchid + 1)
                writer.add_scalar('Train/rot_loss', losses_rot.mean , epoch * len(ds_iter_train) + batchid + 1)
                writer.add_scalar('Train/trans_loss', losses_trans.mean , epoch * len(ds_iter_train) + batchid + 1)
                writer.add_scalar('Train/uv_loss', losses_uv.mean , epoch * len(ds_iter_train) + batchid + 1)
                writer.add_scalar('Train/depth_loss', losses_depth.mean , epoch * len(ds_iter_train) + batchid + 1)
                writer.add_scalar('Train/error2d_loss', losses_error2d.mean, epoch * len(ds_iter_train) + batchid + 1)
                writer.add_scalar('Train/error3d_loss', losses_error3d.mean, epoch * len(ds_iter_train) + batchid + 1)
                writer.add_scalar('Train/error2d_int_loss', losses_error2d_int.mean, epoch * len(ds_iter_train) + batchid + 1)
                writer.add_scalar('Train/error3d_int_loss', losses_error3d_int.mean, epoch * len(ds_iter_train) + batchid + 1)
                writer.add_scalar('Train/error3d_align_loss', losses_error3d_align.mean, epoch * len(ds_iter_train) + batchid + 1)
                losses.reset()
                losses_pose.reset()
                losses_rot.reset()
                losses_trans.reset()
                losses_uv.reset()
                losses_depth.reset()
                losses_error2d.reset()
                losses_error3d.reset()
                losses_error2d_int.reset()
                losses_error3d_int.reset()
                losses_error3d_align.reset()
            writer.add_scalar('LR/learning_rate_opti', optimizer.param_groups[0]['lr'], epoch * len(ds_iter_train) + batchid + 1)
            if len(optimizer.param_groups) > 1:
                for pgid in range(1,len(optimizer.param_groups)):
                    writer.add_scalar(f'LR/learning_rate_opti_{pgid}', optimizer.param_groups[pgid]['lr'], epoch * len(ds_iter_train) + batchid + 1)
        if args.use_schedule:
            lr_scheduler.step()
            
        auc_adds = {}
        for dsname, loader in test_loader_dict.items():
            auc_add = validate(args=args, epoch=epoch, dsname=dsname, loader=loader, model=model, 
                               robot=robot, writer=writer, device=device, device_id=device_id)
            auc_adds[dsname] = auc_add

        save_checkpoint(args=args, auc_adds=auc_adds, 
                        model=model, optimizer=optimizer, 
                        ckpt_folder=ckpt_folder, 
                        epoch=epoch, lr_scheduler=lr_scheduler, 
                        curr_max_auc=curr_max_auc, 
                        curr_max_auc_4real=curr_max_auc_4real)
                  
    print("Training Finished !")
    writer.flush()
