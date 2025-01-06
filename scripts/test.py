import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
from collections import OrderedDict, defaultdict
import numpy as np
import torch
import yaml
from lib.config import (BAXTER_DESCRIPTION_PATH, DREAM_DS_DIR,
                    KUKA_DESCRIPTION_PATH, LOCAL_DATA_DIR, PANDA_DESCRIPTION_PATH)
from lib.dataset.const import INITIAL_JOINT_ANGLE, JOINT_NAMES, JOINT_TO_KP
from lib.dataset.dream import DreamDataset
from lib.dataset.samplers import ListSampler
from lib.utils.urdf_robot import URDFRobot
from lib.models.full_net import get_rootNetwithRegInt_model
from lib.utils.BPnP import BPnP_m3d
from lib.utils.transforms import point_projection_from_3d_tensor
from PIL import Image
from thop import clever_format, profile
from torch.utils.data import DataLoader
from torchnet.meter import AverageValueMeter
from tqdm import tqdm
from lib.utils.geometries import (angle_axis_to_rotation_matrix,
                            compute_euler_angles_from_rotation_matrices,
                            rot6d_to_rotmat, rotmat_to_rot6d)
from lib.utils.metrics import (compute_metrics_batch, draw_add_curve,
                           draw_depth_figure, summary_add_pck)
from lib.utils.vis import vis_joints_3d


def cast(obj, device, dtype=None):
    if isinstance(obj, (dict, OrderedDict)):
        for k, v in obj.items():
            obj[k] = cast(torch.as_tensor(v),device)
            if dtype is not None:
                obj[k] = obj[k].to(dtype)
        return obj
    else:
        return obj.to(device)

def test_network(args):
    
    device_id = args.device_id
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    urdf_robot_name = args.urdf_robot_name
    test_ds_names = args.test_ds_names
    print(f"using {test_ds_names} for testing")
    robot = URDFRobot(urdf_robot_name)
    
    # path
    save_folder = args.exp_path
    model_path = os.path.join(save_folder, f"ckpt/{args.model_name}.pk")
    result_path = os.path.join(save_folder,  'result')
    os.makedirs(result_path, exist_ok=True)

    # make models and set initialization 
    ckpt = torch.load(model_path) 
    init_param_dict = {
        "robot_type" : urdf_robot_name,
        "pose_params": INITIAL_JOINT_ANGLE,
        "cam_params": np.eye(4,dtype=float),
        "init_pose_from_mean": True
    }
    model = get_rootNetwithRegInt_model(init_param_dict, args)
    print("Using rootnet with regression+integral model (2 backbones)")
    model.load_state_dict(ckpt['model_state_dict'])
    print("This model was saved from epoch:", ckpt["epoch"])

    ds_test = DreamDataset(test_ds_names, 
                           rootnet_resize_hw=(args.rootnet_image_size,args.rootnet_image_size), 
                           other_resize_hw=(args.other_image_size,args.other_image_size),
                           color_jitter=False, rgb_augmentation=False, occlusion_augmentation=False)
    ds_iter_test = DataLoader(
        ds_test, batch_size=args.batch_size, num_workers=min(int(os.environ.get('N_CPUS', 10)) - 2, 8)
    )

    def farward_loss(test_args,input_batch, device, model, use_view=False, file_name=None, errors=None, train=True, need_flops=False):
        model.eval() 
        images_id = list(input_batch["image_id"].numpy())
        scene_id = list(input_batch["scene_id"].numpy())
        images_original = cast(input_batch["images_original"],device).float() / 255.
        dtype = torch.float
        root_images = cast(input_batch["root"]["images"],device).float() / 255.
        root_K = cast(input_batch["root"]["K"],device).float()
        
        reg_images = cast(input_batch["other"]["images"],device).float() / 255.
        other_K = cast(input_batch["other"]["K"],device).float()
        
        TCO = cast(input_batch["TCO"],device).float()
        K_original = cast(input_batch["K_original"],device).float()
        gt_jointpose = input_batch["jointpose"]
        gt_keypoints2d_original = cast(input_batch["keypoints_2d_original"],device).float()
        gt_keypoints2d = cast(input_batch["other"]["keypoints_2d"],device).float()
        gt_keypoints3d = cast(input_batch["other"]["keypoints_3d"],device).float()
        valid_mask_crop = cast(input_batch["other"]["valid_mask_crop"],device).float()
                
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
        
        if "synth" not in args.dataset:
            world_3d_pts = robot.get_keypoints_only_fk(gt_pose)
            out = BPnP_m3d.apply(gt_keypoints2d_original, world_3d_pts, K_original[0])
            out_rot = angle_axis_to_rotation_matrix(out[:,0:3])[:,:3,:3]
            out_rot = rotmat_to_rot6d(out_rot)
            gt_rot = cast(out_rot, device).float()

        if args.reference_keypoint_id == 0: # use robot base as root for heatmap
            gt_root_trans = gt_trans
            gt_root_rot = gt_rot
        else:
            gt_root_trans = gt_keypoints3d[:,args.reference_keypoint_id,:]
            gt_root_rot = robot.get_rotation_at_specific_root(gt_pose, gt_rot, gt_trans, root = args.reference_keypoint_id)
        gt_root_depth = gt_root_trans[:,2].unsqueeze(-1)

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
        k_values = torch.tensor([torch.sqrt(fx[n]*fy[n]*real_bbox[0]*real_bbox[1] / area[n]) for n in range(batch_size)]).to(torch.float32).cuda()

        model.float()
        model.to(device)
        model = torch.nn.DataParallel(model, device_ids=device_id, output_device=device_id[0])
        
        pred_pose, pred_rot, pred_trans, pred_root_uv, pred_root_depth, \
            pred_uvd, pred_keypoints3d_int, pred_keypoints3d_fk, times = model(reg_images, root_images, k_values, K=other_K, test_fps=True)

        if cfg.known_joint:
            pred_pose = gt_pose.clone()
        
        image_dis3d_avg, image_dis2d_avg, batch_dis3d_avg, batch_dis2d_avg, \
        batch_l1jointerror_avg, image_l1jointerror_avg, root_depth_error, batch_error_relative, error3d_relative = compute_metrics_batch(
                                                                        robot=robot,
                                                                        gt_keypoints3d=gt_keypoints3d,
                                                                        gt_keypoints2d=gt_keypoints2d_original,
                                                                        K_original=K_original,
                                                                        gt_joint=gt_pose,
                                                                        pred_joint=pred_pose,
                                                                        pred_rot=pred_rot,
                                                                        pred_trans=pred_trans,
                                                                        pred_depth=None,
                                                                        pred_xy=None,
                                                                        pred_xyz_integral=None,
                                                                        reference_keypoint_id=args.reference_keypoint_id
                                                                        )
        gt_rotmat = rot6d_to_rotmat(gt_root_rot)
        pred_rotmat = rot6d_to_rotmat(pred_rot)
        gt_rotang = compute_euler_angles_from_rotation_matrices(gt_rotmat).detach().cpu()
        pred_rotang = compute_euler_angles_from_rotation_matrices(pred_rotmat).detach().cpu()
        mean_rotang = torch.mean(torch.abs(gt_rotang - pred_rotang), dim=1).numpy().reshape(batch_size,)
        mean_rotang = list(mean_rotang)
        
        pred_keypoints2d_reproj_int = point_projection_from_3d_tensor(other_K, pred_keypoints3d_int)
        error2d_reproj = torch.norm(pred_keypoints2d_reproj_int - gt_keypoints2d, dim = 2)
        error2d_reproj = cast(error2d_reproj * valid_mask_crop, device)
        mean_kp2d_distance = torch.sum(error2d_reproj) / torch.sum(valid_mask_crop != 0)
        
        if use_view:
            assert file_name
            assert errors
            vis_path = os.path.join(result_path, "vis")
            os.makedirs(vis_path, exist_ok=True)           
            vis_joints_3d(images_original.detach().cpu().numpy(),
                        pred_keypoints3d_fk.detach().cpu().numpy(), gt_keypoints3d.detach().cpu().numpy(),
                        gt_2d = gt_keypoints2d.detach().cpu().numpy(), K_original=K_original.detach().cpu().numpy(),
                        bbox = bboxes.detach().cpu().numpy(),
                        file_name = file_name, errors = errors, vis_dir = vis_path)
            print('saved')
        
        return scene_id, image_dis3d_avg, image_dis2d_avg, batch_dis3d_avg, batch_dis2d_avg, \
            batch_l1jointerror_avg, image_l1jointerror_avg, images_id, root_depth_error, gt_root_depth, batch_error_relative, error3d_relative, times, mean_rotang, mean_kp2d_distance

    def test():
        model.eval()
        alldis, alldis_relative = defaultdict(list), defaultdict(list)
        losses_pose, losses_rot, losses_trans = AverageValueMeter(),AverageValueMeter(),AverageValueMeter()
        time_root, time_other, time_image = AverageValueMeter(),AverageValueMeter(),AverageValueMeter()
        add_thresholds = [1,5,10,20,40,60,80,100]
        pck_thresholds = [2.5,5.0,7.5,10.0,12.5,15.0,17.5,20.0]
        metric_l1joint = [AverageValueMeter() for i in range(robot.dof)]
        with torch.no_grad():
            for idx, sample in enumerate(tqdm(ds_iter_test, dynamic_ncols=True)):
                need_flops = True if idx == 0 else False
                scene_id, image_dis3d_avg, image_dis2d_avg, batch_dis3d_avg, batch_dis2d_avg, \
                batch_l1jointerror_avg, image_l1jointerror_avg, images_id, error_depth, gt_root_depth, batch_error_relative, error3d_relative, times, mean_rotang, mean_kp2d_distance = \
                farward_loss(test_args=args,input_batch=sample, device=device, model=model, use_view=False, file_name=None, train=False, need_flops=need_flops)
                time_root.add(times[0])
                time_other.add(times[1])
                time_image.add(times[2])
                alldis["id"].extend(images_id)
                alldis["scene_id"].extend(list(scene_id))
                alldis["dis3d"].extend(image_dis3d_avg)
                alldis["dis2d"].extend(image_dis2d_avg)
                alldis["jointerror"].extend(image_l1jointerror_avg)
                alldis["deptherror"].extend(error_depth)
                alldis["gt_root_depth"].extend(gt_root_depth.detach().cpu().numpy())
                alldis["deptherror_relative"].extend(batch_error_relative)
                alldis["mean_rot_angle"].extend(mean_rotang)
                alldis["mean_kp2d_distance"].append(mean_kp2d_distance.item())
                alldis_relative["dis3d"].extend(error3d_relative)
                alldis_relative["dis2d"].extend(image_dis2d_avg)
                for id in range(robot.dof):
                    metric_l1joint[id].add(batch_l1jointerror_avg[id])
        assert len(alldis["scene_id"]) == len(alldis["dis3d"])
        itemid = np.array(alldis["id"])
        ids = np.array(alldis["scene_id"])
        adds = np.array(alldis["dis3d"])
        summary = summary_add_pck(alldis)
        summary_relative = summary_add_pck(alldis_relative)
        draw_add_curve(alldis, result_path, test_ds_names,  auc=summary['ADD/AUC'])
        auc_add = summary['ADD/AUC']
        auc_pck = summary['PCK/AUC']
        mean_joint_error = np.mean(alldis["jointerror"]) / np.pi * 180.0  # degree
        mean_depth_error = np.mean(alldis["deptherror"])
        mean_rotangle_error = np.mean(alldis["mean_rot_angle"]) / np.pi * 180.0  # degree
        relative_depth_error = np.mean(alldis["deptherror_relative"])
        mean_kp2d_distance_error = np.mean(alldis["mean_kp2d_distance"])
        if args.logging:
            with open(os.path.join(result_path, "summary.txt"), 'a') as f:
                f.write("Model metrics summary" + '\n')
                f.write("Dataset for testing: " + args.test_ds_names + '\n')
                f.write("This model was saved from epoch:" + str(ckpt['epoch']) + '\n')
                f.write("Joint_l1_error/mean (degree): " + str(mean_joint_error) + '\n' )
                f.write("Depth_l1_error/mean (m): " + str(mean_depth_error) + '\n')
                f.write("Rotation_l1_error/mean (degree): " + str(mean_rotangle_error) + '\n')
                f.write("Relative_l1_error/mean (m): " + str(relative_depth_error) + '\n')
                f.write("KeypointNet_2d_distance/mean (pixel): " + str(mean_kp2d_distance_error) + '\n')
                f.write("Relative_ADD/AUC: " + str(summary_relative['ADD/AUC']) + '\n')
                f.write("ADD/AUC: " + str(auc_add) + '\n')
                f.write("ADD/mean (m): " + str(summary["ADD/mean"]) + '\n')
                f.write("ADD/median (m): " + str(summary["ADD/median"]) + '\n')
                f.write("PCK/AUC: " + str(auc_pck) + '\n')
                f.write("ADD_2D/mean (pixel): " + str(summary["ADD_2D/mean"]) + '\n')
                f.write("ADD_2D/median (pixel): " + str(summary["ADD_2D/median"]) + '\n')
                for th_mm in add_thresholds:
                    f.write(f'ADD<{th_mm}mm: ' + str(summary[f'ADD_{th_mm}_mm']) + '\n')
                for th_p in pck_thresholds:
                    f.write(f'ADD_2d<{th_p}pixel: ' + str(summary[f'PCK_{th_p}_pixel']) + '\n')
                for k in range(robot.dof):
                    f.write(f'Joint_l1_error/joint_{k+1} (degree): {metric_l1joint[k].mean / np.pi * 180.0} \n')
                f.write("Runtimes:\n")
                f.write(f"Runtime of rootnet: {time_root.mean} \n")
                f.write(f"Runtime of regression+integral: {time_other.mean} \n")
                f.write(f"Runtime of all: {time_image.mean} \n")
                f.write(f"time_image.mean-time_other.mean: {(time_image.mean-time_other.mean)} \n")
                f.write(f"FPS_parallel: {int(1/(time_image.mean-time_other.mean))} \n")
                f.write(f"FPS: {int(1/time_image.mean)} \n")
                f.write(" \n")

        return alldis
               
    alldis = test() 
    
    if args.visualization:
        assert len(alldis["id"]) == len(alldis["dis3d"])
        result = [(alldis["dis3d"][i], alldis["id"][i]) for i in range(len(alldis["id"]))]
        result_ordered = sorted(result)
        errors = [item[0] for item in result_ordered]
        view_ids = [item[1] for item in result_ordered]
        errors_list = [errors[i] for i in np.arange(0,len(errors),10)]
        errors_list.append(errors[-1])
        view_ids_list = [view_ids[i] for i in np.arange(0,len(errors),10)]
        view_ids_list.append(view_ids[-1])
        errors = errors_list
        view_ids = view_ids_list

        view_sampler = ListSampler(view_ids)
        ds_iter_test_view = DataLoader(
            ds_test, batch_size=args.view_batch_size, sampler=view_sampler, num_workers=min(int(os.environ.get('N_CPUS', 10)) - 2, 8)
        )

        # low error, good performance
        for batchid, sample in enumerate(tqdm(ds_iter_test_view, dynamic_ncols=True)):
            error_values = errors[(batchid * args.view_batch_size):((batchid+1) * args.view_batch_size)]
            print(f"3d errors (m): {error_values}")
            file_name = f"Best predictions {batchid+1}"
            scene_id, image_dis3d_avg, image_dis2d_avg, batch_dis3d_avg, batch_dis2d_avg, \
            batch_l1jointerror_avg, image_l1jointerror_avg, images_id, depth_error, gt_root_depth, batch_error_relative, error3d_relative, times, mean_rotang, mean_kp2d_distance = \
            farward_loss(test_args=args,input_batch=sample, device=device, model=model, use_view=True, file_name=file_name, errors=error_values, train=False)
            if batchid == 4:
                break
        
        # high error, bad performance
        view_sampler_r = ListSampler(view_ids[::-1])
        ds_iter_test_view_r = DataLoader(
            ds_test, batch_size=args.view_batch_size, sampler=view_sampler_r, num_workers=min(int(os.environ.get('N_CPUS', 10)) - 2, 8)
        )
        for batchid, sample in enumerate(tqdm(ds_iter_test_view_r, dynamic_ncols=True)):
            error_values = errors[::-1][(batchid * args.view_batch_size):((batchid+1) * args.view_batch_size)]
            print(f"3d errors (m): {error_values}")
            file_name = f"Worst predictions {batchid+1}"
            scene_id , image_dis3d_avg, image_dis2d_avg, batch_dis3d_avg, batch_dis2d_avg, \
            batch_l1jointerror_avg, image_l1jointerror_avg, images_id, depth_error, gt_root_depth, batch_error_relative, error3d_relative, times, mean_rotang, mean_kp2d_distance = \
            farward_loss(test_args=args,input_batch=sample, device=device, model=model, use_view=True, file_name=file_name, errors=error_values, train=False)
            if batchid == 4:
                break     
        

def make_cfg(args):
    cfg = argparse.ArgumentParser('').parse_args([])
    
    cfg.exp_path = args.exp_path
    
    cfg.known_joint = args.known_joint
    print("cfg.known_joint:", cfg.known_joint)
    
    cfg.model_name = args.model_name
    print("cfg.model_name:", cfg.model_name)
    
    exp_cfg_path = os.path.join(cfg.exp_path, "config.yaml")
    
    with open(exp_cfg_path, encoding="utf-8") as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
    f.close()
    
    # Test settings
    cfg.logging = True
    cfg.visualization = args.vis_skeleton
    cfg.view_batch_size = 4
    cfg.view_batches = 15

    cfg.batch_size = 128
    cfg.no_cuda = False
    cfg.device_id = [0]
    cfg.image_size = config["image_size"]
    cfg.rootnet_image_size = int(config["rootnet_image_size"])
    cfg.other_image_size = int(config["other_image_size"])

    # Test dataset
    pre_path = DREAM_DS_DIR
    cfg.dataset = args.dataset
    if args.dataset is not None:
        if "synth" in args.dataset:
            cfg.test_ds_names = os.path.abspath(pre_path / ("synthetic/" + args.dataset))
        else:
            cfg.test_ds_names = os.path.abspath(pre_path / ("real/" + args.dataset))
    else:
        raise ValueError
    
    # Model settings
    cfg.urdf_robot_name = config["urdf_robot_name"]
    cfg.backbone_name= config["backbone_name"]
    cfg.n_iter = config["n_iter"]
    cfg.use_rpmg = config["use_rpmg"]

    # pipeline
    cfg.use_direct_reg_branch = config["use_direct_reg_branch"]
    cfg.reference_keypoint_id = config["reference_keypoint_id"]
    cfg.use_origin_bbox = config["use_origin_bbox"] if "use_origin_bbox" in config else False
    cfg.use_extended_bbox = config["use_extended_bbox"]
    cfg.rootnet_backbone_name = config["rootnet_backbone_name"] if "rootnet_backbone_name" in config else "resnet50"
    cfg.pretrained_rootnet = config["pretrained_rootnet"] if "rootnet_backbone_name" in config else None
    cfg.pretrained_rootnet = None if cfg.pretrained_rootnet == "None" else cfg.pretrained_rootnet
    cfg.add_fc = config["add_fc"] if "add_fc" in config else False
    cfg.rotation_dim = config["rotation_dim"] if "rotation_dim" in config else 6
    
    cfg.reg_joint_map = config["reg_joint_map"] if "reg_joint_map" in config else False
    cfg.joint_conv_dim = config["joint_conv_dim"] if "joint_conv_dim" in config else [128,128,128]
    cfg.p_dropout = config["p_dropout"] if "p_dropout" in config else 0.5
    cfg.direct_reg_rot = config["direct_reg_rot"] if "direct_reg_rot" in config else False
    cfg.rot_iterative_matmul = config["rot_iterative_matmul"] if "rot_iterative_matmul" in config else False
    cfg.pretrained_onlyregression = config["pretrained_onlyregression"] if "pretrained_onlyregression" in config else None

    cfg.fix_root = config["fix_root"] if "fix_root" in config else False
    cfg.bbox_3d_shape = config["bbox_3d_shape"] if "bbox_3d_shape" in config else [1300, 1300, 1300]
    
    cfg.multi_kp = config["multi_kp"] if "multi_kp" in config else False
    cfg.kps_need_depth = config["kps_need_depth"] if "kps_need_depth" in config else None

    return cfg

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Testing')
    parser.add_argument('--exp_path', '-e', type=str, required=True)
    parser.add_argument('--dataset', '-d', type=str, required=True, help= "e.g. panda_synth_test_dr") 
    parser.add_argument('--known_joint', '-k', type=bool, default=False, help= "whether use gt joint for testing")
    parser.add_argument('--model_name', '-m', type=str, default="curr_best_auc(add)_model", help= "model name") 
    parser.add_argument('--vis_skeleton', action="store_true", help= "visualization of pose skeleton") 
    args = parser.parse_args()
    cfg = make_cfg(args)
    test_network(cfg)
