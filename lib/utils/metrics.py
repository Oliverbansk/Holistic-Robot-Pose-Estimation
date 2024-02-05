import torch
import numpy as np
from utils.transforms import point_projection_from_3d
import matplotlib.pyplot as plt
import seaborn as sns
import os

def compute_metrics_batch(robot,gt_keypoints3d,gt_keypoints2d,K_original,gt_joint,**pred_kwargs):
    
    # compute 3d keypoints locations 
    # output shape: (batch_size, keypoints_num, 3)
    pred_joint = pred_kwargs["pred_joint"]
    pred_rot = pred_kwargs["pred_rot"]
    pred_trans = pred_kwargs["pred_trans"]
    if "pred_xy" in pred_kwargs and "pred_depth" in pred_kwargs and pred_kwargs["pred_xy"] is not None and pred_kwargs["pred_depth"] is not None:
        pred_xy = pred_kwargs["pred_xy"]
        pred_depth = pred_kwargs["pred_depth"]
        pred_trans = torch.cat((pred_xy,pred_depth),dim=-1)
    pred_xyz_integral = pred_kwargs["pred_xyz_integral"]
    reference_keypoint_id = pred_kwargs["reference_keypoint_id"]
    
    if pred_joint is None or pred_rot is None or pred_trans is None:
        assert pred_xyz_integral is not None
        pred_keypoints3d = pred_xyz_integral
        batch_size = pred_xyz_integral.shape[0]
    else:
        if reference_keypoint_id == 0:
            pred_keypoints3d = robot.get_keypoints(pred_joint,pred_rot,pred_trans)
            batch_size = pred_joint.shape[0]
            pred_joint = pred_joint.detach().cpu().numpy()
        else:
            pred_keypoints3d = robot.get_keypoints_root(pred_joint,pred_rot,pred_trans,root=reference_keypoint_id)
            batch_size = pred_joint.shape[0]
            pred_joint = pred_joint.detach().cpu().numpy()
        
    keypoints_num = len(robot.link_names)
    dof = robot.dof
    pred_keypoints3d = pred_keypoints3d.detach().cpu().numpy()
    gt_keypoints3d = gt_keypoints3d.detach().cpu().numpy()
    gt_keypoints2d = gt_keypoints2d.detach().cpu().numpy() 
    K_original = K_original.detach().cpu().numpy()
    gt_joint = gt_joint.detach().cpu().numpy()
    pred_keypoints2d = point_projection_from_3d(K_original,pred_keypoints3d)
    assert(pred_keypoints3d.shape == (batch_size,keypoints_num,3)),f"{pred_keypoints3d.shape}"
    assert(gt_keypoints3d.shape == (batch_size,keypoints_num,3)),f"{gt_keypoints3d.shape}"
    assert(pred_keypoints2d.shape == (batch_size,keypoints_num,2)),f"{pred_keypoints2d.shape}"
    assert(gt_keypoints2d.shape == (batch_size,keypoints_num,2)),f"{gt_keypoints2d.shape}"
    
    
    # Thresholds (ADD:mm, PCK:pixel)
    add_thresholds = [1,5,10,20,40,60,80,100]
    pck_thresholds = [2.5,5.0,7.5,10.0,12.5,15.0,17.5,20.0]
    
    # ADD Average distance of detected keypoints within threshold
    error3d_batch = np.linalg.norm(pred_keypoints3d - gt_keypoints3d, ord = 2, axis = 2)
    assert(error3d_batch.shape == (batch_size,keypoints_num))
    error3d = np.mean(error3d_batch, axis = 1)
    # pcts3d = [len(np.where(error3d < th_mm/1000.0)[0])/float(error3d.shape[0]*error3d.shape[1]) for th_mm in add_thresholds]
    
    # PCK percentage of correct keypoints (only keypoints within the camera frame)
    error2d_batch = np.linalg.norm(pred_keypoints2d - gt_keypoints2d, ord = 2, axis = 2)
    assert(error2d_batch.shape == (batch_size,keypoints_num))
    valid = (gt_keypoints2d[:,:,0] <= 640.0) & (gt_keypoints2d[:,:,0] >= 0) & (gt_keypoints2d[:,:,1] <= 480.0) & (gt_keypoints2d[:,:,1] >= 0)
    error2d_all = error2d_batch * valid
    error2d_sum = np.sum(error2d_all, axis = 1)
    valid_sum = np.sum(valid, axis = 1)
    error2d = error2d_sum / valid_sum
    # pcts2d = [len(np.where(error2d < th_p)[0])/float(error2d.shape[0]*error2d.shape[1]) for th_p in pck_thresholds]
    
    # 3D/2D mean distance with gt of each keypoints
    dis3d = list(np.mean(error3d_batch, axis = 0))
    error2d_sum_batch = np.sum(error2d_all, axis = 0)
    valid_sum_batch = np.sum(valid, axis = 0)
    dis2d = error2d_sum_batch / valid_sum_batch
    # dis2d = list(np.mean(error2d_batch, axis = 0))
    
    # mean joint angle L1 error (per joint)
    # mean joint angle L1 error (per image)
    if pred_joint is not None:
        # pred_joint = pred_joint.detach().cpu().numpy()
        assert(gt_joint.shape == pred_joint.shape and gt_joint.shape == (batch_size, dof)), f"{pred_joint.shape},{gt_joint.shape}"
        error_joint = np.abs(gt_joint - pred_joint)
        l1_jointerror = list(np.mean(error_joint, axis = 0))
        if robot.robot_type == "panda":
            mean_jointerror = list(np.mean(error_joint[:,:-1], axis = 1))
        else:
            mean_jointerror = list(np.mean(error_joint, axis = 1))
        assert(len(mean_jointerror) == batch_size), len(mean_jointerror)
    else:
        l1_jointerror = [0] * dof
        mean_jointerror = [0] * batch_size
    
    # depth l1 error
    reference_keypoint_id = pred_kwargs["reference_keypoint_id"]
    error_depth = np.abs(pred_keypoints3d[:,reference_keypoint_id,2] - gt_keypoints3d[:,reference_keypoint_id,2])

    # root relative error
    pred_relatives = pred_keypoints3d[:,:,2] - pred_keypoints3d[:,reference_keypoint_id:reference_keypoint_id+1,2]
    gt_relatives = gt_keypoints3d[:,:,2] - gt_keypoints3d[:,reference_keypoint_id:reference_keypoint_id+1,2]
    error_relative = np.abs(pred_relatives - gt_relatives)
    batch_error_relative = np.mean(error_relative, axis=1)

    # root relative auc
    pred_keypoints3d_relative = pred_keypoints3d.copy()
    pred_keypoints3d_relative[:,:,2] = pred_relatives
    gt_keypoints3d_relative = gt_keypoints3d.copy()
    gt_keypoints3d_relative[:,:,2] = gt_relatives
    error3d_relative_batch = np.linalg.norm(pred_keypoints3d_relative - gt_keypoints3d_relative, ord = 2, axis = 2)
    assert(error3d_relative_batch.shape == (batch_size,keypoints_num))
    error3d_relative = np.mean(error3d_relative_batch, axis = 1)



    return error3d, error2d, dis3d, dis2d, l1_jointerror, mean_jointerror, error_depth, batch_error_relative, error3d_relative


def summary_add_pck(alldis):
    
    dis3d = np.array(alldis['dis3d'])
    dis2d = np.array(alldis['dis2d'])
    assert(dis3d.shape[0] == dis2d.shape[0])
    
    add_threshold_ontb = [1,5,10,20,40,60,80,100]
    pck_threshold_ontb = [2.5,5.0,7.5,10.0,12.5,15.0,17.5,20.0]

    # for ADD
    auc_threshold = 0.1
    delta_threshold = 0.00001
    add_threshold_values = np.arange(0.0, auc_threshold, delta_threshold)
    counts_3d = []
    for value in add_threshold_values:
        under_threshold = (
            np.mean(dis3d <= value)
        )
        counts_3d.append(under_threshold)
    auc_add = np.trapz(counts_3d, dx=delta_threshold) / auc_threshold
    
    # for PCK
    auc_pixel_threshold = 20.0
    delta_pixel = 0.01
    pck_threshold_values = np.arange(0, auc_pixel_threshold, delta_pixel)
    counts_2d = []
    for value in pck_threshold_values:
        under_threshold = (
            np.mean(dis2d <= value)
        )
        counts_2d.append(under_threshold)
    auc_pck = np.trapz(counts_2d, dx=delta_pixel) / auc_pixel_threshold

    summary = {
        'ADD/mean': np.mean(dis3d),
        'ADD/median': np.median(dis3d),
        'ADD/AUC': auc_add.item(),
        'ADD_2D/mean': np.mean(dis2d),
        'ADD_2D/median': np.median(dis2d),
        'PCK/AUC': auc_pck.item()
    }
    for th_mm in add_threshold_ontb:
        summary[f'ADD_{th_mm}_mm'] = np.mean(dis3d <= th_mm * 1e-3)
    for th_p in pck_threshold_ontb:
        summary[f'PCK_{th_p}_pixel'] = np.mean(dis2d <= th_p)
    return summary


def draw_add_curve(alldis, savename, testdsname, auc):
        
    dis3d = np.array(alldis['dis3d'])
    auc_threshold = 0.1
    delta_threshold = 0.00001
    add_threshold_values = np.arange(0.0, auc_threshold, delta_threshold)
    counts_3d = []
    for value in add_threshold_values:
        under_threshold = (
            np.mean(dis3d <= value)
        )
        counts_3d.append(under_threshold)
    plt.figure(figsize=(25,18))
    grid = plt.GridSpec(2,2, wspace=0.1, hspace=0.2)
    plt.subplot(grid[0,0])
    plt.grid()
    plt.plot(add_threshold_values, counts_3d)
    plt.xlim(0,auc_threshold)
    plt.ylim(0,1.0)
    plt.xlabel("add threshold values (unit: m)")
    plt.ylabel("percentages")
    plt.axvline(x=np.mean(dis3d), color='red', linestyle='--', label='mean distance')
    plt.axvline(x=np.median(dis3d), color='green', linestyle='--', label='median distance')
    plt.title("ADD curve")
    plt.text(x=0.001, y=0.9, s="auc="+str(round(auc*100, ndigits=2)))
    plt.legend()
    
    plt.subplot(grid[0,1])
    sns.histplot(dis3d, kde=True)
    plt.title("3d distance distribution, whole range")
    
    plt.subplot(grid[1,0])
    sns.histplot(dis3d, kde=True)
    plt.xlim(0, 0.5)
    plt.title("3d distance distribution, range: 0~0.5m")
    
    plt.subplot(grid[1,1])
    sns.histplot(dis3d, kde=True)
    plt.xlim(0, 0.1)
    plt.xticks(np.arange(0.0,0.101,0.01))
    plt.title("3d distance distribution, range: 0~0.1m")
    plt.axvline(x=np.mean(dis3d), color='red', linestyle='--', label='mean distance')
    plt.axvline(x=np.median(dis3d), color='green', linestyle='--', label='median distance')

    dataset_name = testdsname.split("/")[-1]
    
    plt.savefig(os.path.join(savename, f"add_distribution_curve_{dataset_name}.jpg"))
    print("drawn add curve in folder vis")
    plt.close()
        

def draw_depth_figure(alldis, savename, testdsname):
    if "dr" in testdsname.split("/")[-1]:
        ds = "dr"
    elif "photo" in testdsname.split("/")[-1]:
        ds = "photo"
    else:
        ds = testdsname.split("/")[-1]
    assert len(alldis["deptherror"]) == len(alldis["gt_root_depth"]), (len(alldis["deptherror"]), len(alldis["gt_root_depth"]))
    deptherror = np.array(alldis["deptherror"])
    gtrootdepth = np.array(alldis["gt_root_depth"])
    plt.figure(figsize=(15,15))
    plt.scatter(gtrootdepth, deptherror)
    plt.xlim(0, 2.0)
    plt.ylim(0, 0.2)
    plt.title("root depth error -- gt root depth scatterplot")
    plt.savefig("unit_test/depth_curve/"+savename+"_"+ds+".jpg")
    plt.close()

    plt.close()
