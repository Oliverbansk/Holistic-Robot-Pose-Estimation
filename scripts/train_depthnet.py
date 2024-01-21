import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import defaultdict
import numpy as np
import torch
from lib.dataset.const import JOINT_NAMES
from lib.dataset.dream import DreamDataset
from lib.dataset.multiepoch_dataloader import MultiEpochDataLoader
from lib.dataset.samplers import PartialSampler
from lib.models.depth_net import get_rootnet
from lib.utils.urdf_robot import URDFRobot
from lib.utils.utils import cast, set_random_seed, create_logger, get_scheduler
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchnet.meter import AverageValueMeter
from tqdm import tqdm


def train_depthnet(args):
    
    torch.autograd.set_detect_anomaly(True)
    set_random_seed(808)
    
    save_folder, ckpt_folder, log_folder, writer = create_logger(args)
    
    urdf_robot_name = args.urdf_robot_name
    robot = URDFRobot(urdf_robot_name)
 
    device_id = args.device_id
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    

    train_ds_names = args.train_ds_names
    test_ds_name_dr = train_ds_names.replace("train_dr","test_dr")
    if urdf_robot_name != "baxter":
        test_ds_name_photo = train_ds_names.replace("train_dr","test_photo")
    if urdf_robot_name == "panda":
        test_ds_name_real = [train_ds_names.replace("synthetic/panda_synth_train_dr","real/panda-3cam_azure"),
                            train_ds_names.replace("synthetic/panda_synth_train_dr","real/panda-3cam_kinect360"),
                            train_ds_names.replace("synthetic/panda_synth_train_dr","real/panda-3cam_realsense"),
                            train_ds_names.replace("synthetic/panda_synth_train_dr","real/panda-orb")]
        
    # make train and test(validation) datasets/dataloaders
    ds_train = DreamDataset(train_ds_names, 
                            # rootnet_resize_hw=(int(args.image_size),int(args.image_size)), 
                            color_jitter=args.jitter, rgb_augmentation=args.other_aug, occlusion_augmentation=args.occlusion, 
                            flip = args.rootnet_flip, occlu_p = args.occlu_p,
                            padding=args.padding, extend_ratio=args.extend_ratio)
    ds_test_dr = DreamDataset(test_ds_name_dr, 
                              # rootnet_resize_hw=(int(args.image_size),int(args.image_size)), 
                              color_jitter=False, rgb_augmentation=False, occlusion_augmentation=False, 
                              flip = False,
                              padding=args.padding, extend_ratio=args.extend_ratio) 
    if urdf_robot_name != "baxter":
        ds_test_photo = DreamDataset(test_ds_name_photo, 
                                     # rootnet_resize_hw=(int(args.image_size),int(args.image_size)), 
                                     color_jitter=False, rgb_augmentation=False, occlusion_augmentation=False, 
                                     flip = False,
                                     padding=args.padding, extend_ratio=args.extend_ratio) 
    
    train_sampler = PartialSampler(ds_train, epoch_size=args.epoch_size)
    if args.resample:
        weights_sampler = np.load("unit_test/z_weights.npy")
        train_sampler = WeightedRandomSampler(weights_sampler, num_samples=min(args.epoch_size, len(ds_train))) 
    ds_iter_train = DataLoader(
        ds_train, sampler=train_sampler, batch_size=args.batch_size, num_workers=args.n_dataloader_workers, drop_last=False, pin_memory=True
    )
    ds_iter_train = MultiEpochDataLoader(ds_iter_train)

    test_loader_dict = {}
    ds_iter_test_dr = DataLoader(
        ds_test_dr, batch_size=args.batch_size, num_workers=args.n_dataloader_workers
    )
    test_loader_dict["dr"] = ds_iter_test_dr
    if urdf_robot_name != "baxter":
        ds_iter_test_photo = DataLoader(
            ds_test_photo, batch_size=args.batch_size, num_workers=args.n_dataloader_workers
        )

    if urdf_robot_name == "panda":
        ds_shorts = ["azure", "kinect", "realsense", "orb"]
        for ds_name, ds_short in zip(test_ds_name_real, ds_shorts):
            ds_test_real = DreamDataset(ds_name, 
                                        # rootnet_resize_hw=(int(args.image_size),int(args.image_size)), 
                                        color_jitter=False, rgb_augmentation=False, occlusion_augmentation=False, 
                                        flip = False,
                                        padding=args.padding, extend_ratio=args.extend_ratio) 
            ds_iter_test_real = DataLoader(
                ds_test_real, batch_size=args.batch_size, num_workers=args.n_dataloader_workers
            )
            test_loader_dict[ds_short] = ds_iter_test_real
    
    print("len(ds_iter_train): ",len(ds_iter_train))
    print("len(ds_iter_test_dr): ", len(ds_iter_test_dr))
    if urdf_robot_name != "baxter":
        print("len(ds_iter_test_photo): ", len(ds_iter_test_photo))
    if urdf_robot_name == "panda":
        for ds_short in ds_shorts:
            print(f"len(ds_iter_test_{ds_short}): ", len(test_loader_dict[ds_short]))
    model = get_rootnet(args.backbone_name, 
                    pred_xy=args.use_rootnet_xy_branch,
                    add_fc=args.add_fc,
                    use_offset=args.use_offset)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    curr_min_loss = 1e10
    curr_min_loss_4real = { "azure": 1e10, "kinect": 1e10, "realsense": 1e10, "orb": 1e10 }
    curr_min_loss_allreal = 1e10
    
    if args.resume_run:
        resume_dir =  os.path.join("experiments" , args.resume_experiment_name)
        path = os.path.join(resume_dir, 'ckpt/curr_best_root_depth_model.pk')
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
        curr_min_loss = checkpoint["loss"]
        if urdf_robot_name == "panda":
            for postfix in ['ckpt/curr_best_root_depth_azure_model.pk', 'ckpt/curr_best_root_depth_kinect_model.pk', 'ckpt/curr_best_root_depth_realsense_model.pk', 'ckpt/curr_best_root_depth_orb_model.pk']:   
                model_path = os.path.join(resume_dir, postfix)
                ckpt = torch.load(model_path)
                curr_min_loss_onreal = ckpt["loss"]
                for real_name in curr_min_loss_4real.keys():
                    if real_name in postfix:
                        curr_min_loss_4real[real_name] = curr_min_loss_onreal
                        break
            model_path = os.path.join(resume_dir, "ckpt/curr_best_root_depth_allreal_model.pk")
            ckpt_allreal = torch.load(model_path)
            curr_min_loss_allreal = ckpt_allreal["loss"]
        
    else:
        start_epoch = 0
        last_epoch = -1
    end_epoch = args.n_epochs
    
    lr_scheduler = get_scheduler(args, optimizer, last_epoch)
    
    
    # for loop
    for epoch in range(start_epoch, end_epoch + 1):
        print('In epoch {} ----------------- (script: training rootnet)'.format(epoch + 1))
        
        def farward_loss(args,input_batch, device, model, train=True):
            if train:
                model.train()
            else:
                model.eval() 
            
            dtype = torch.float
        
            # Get data from the batch and cast to GPU
            images = cast(input_batch["root"]["images"],device).float() / 255.
            TCO = cast(input_batch["TCO"],device).float()
            K = cast(input_batch["root"]["K"],device).float()
            K_original = cast(input_batch["K_original"],device).float()
            if args.use_origin_bbox:
                bboxes = cast(input_batch["bbox_strict_bounded_original"], device).float()
            else:
                bboxes = cast(input_batch["root"]["bbox_strict_bounded"], device).float()
            if args.use_extended_bbox:
                bboxes = cast(input_batch["root"]["bbox_gt2d_extended"], device).float()
            gt_jointpose = input_batch["jointpose"]
            gt_keypoints3d = cast(input_batch["root"]["keypoints_3d"],device).float()
            valid_mask = cast(input_batch["valid_mask"], device).float()
            valid_mask_crop = cast(input_batch["root"]["valid_mask_crop"], device).float()
            batch_size = images.shape[0]
            robot_type = urdf_robot_name

            gt_pose = []
            gt_trans = []
            for n in range(batch_size):
                jointpose = torch.as_tensor([gt_jointpose[k][n] for k in JOINT_NAMES[robot_type]])
                jointpose = cast(jointpose, device,dtype=dtype)
                trans = TCO[n,:3,3]
                gt_pose.append(jointpose)
                gt_trans.append(trans)
            gt_pose = torch.stack(gt_pose, 0).to(torch.float32)
            gt_trans = torch.stack(gt_trans, 0).to(torch.float32)
            if args.reference_keypoint_id == 0: # use robot base as root for heatmap
                gt_root_trans = gt_trans
            else:
                assert(args.reference_keypoint_id < len(robot.link_names)), args.reference_keypoint_id
                gt_root_trans = gt_keypoints3d[:,args.reference_keypoint_id,:]
            assert(gt_root_trans.shape == (batch_size, 3)), gt_root_trans
            gt_root_depth = gt_root_trans[:,2].unsqueeze(-1)
            
            if args.multi_kp:
                gt_kp_depths = gt_keypoints3d[:,args.kps_need_depth,2]
            
            root_relatives = gt_keypoints3d[:,:,2].clone()
            root_relatives = (root_relatives - gt_root_depth) / (float(args.bbox_3d_shape[2]) * 1e-3)

            real_bbox = torch.tensor([1000.0, 1000.0]).to(torch.float32)
            if args.use_origin_bbox:
                fx, fy = K_original[:,0,0], K_original[:,1,1]
            else:
                fx, fy = K[:,0,0], K[:,1,1]
            if args.use_extended_bbox:
                fx, fy = K[:,0,0], K[:,1,1]
            assert(fx.shape == fy.shape and fx.shape == (batch_size,)), (fx.shape,fy.shape)
            area = torch.max(torch.abs(bboxes[:,2]-bboxes[:,0]), torch.abs(bboxes[:,3]-bboxes[:,1])) ** 2
            assert(area.shape == (batch_size,)), area.shape
            k_values = torch.tensor([torch.sqrt(torch.abs(fx[n])*torch.abs(fy[n])*real_bbox[0]*real_bbox[1] / area[n]) for n in range(batch_size)]).to(torch.float32)
            assert torch.sum(torch.isnan(k_values)) == 0, torch.sum(torch.isnan(k_values))
                
            # Cast model to the GPU
            model.to(device)
            model.float()
            model = torch.nn.DataParallel(model, device_ids=device_id, output_device=device_id[0])

            # Forward
            if args.use_rootnet_xy_branch:
                coord = model(images, k_values)
                pred_root_depth = coord[:,2] / 1000.0 # unit: mm -> m
            else:
                if args.multi_kp:
                    pred_depths = model(images, k_values)
                    pred_depths = pred_depths / 1000.0 # unit: mm -> m
                    root_index = args.kps_need_depth.index(args.reference_keypoint_id)
                    pred_root_depth = pred_depths[:,root_index].reshape(-1,1)
                else:
                    pred_root_depth = model(images, k_values)
                    pred_root_depth = pred_root_depth / 1000.0 # unit: mm -> m

            # metrics: depth/joint
            if not train:
                error_depth = torch.abs(pred_root_depth.reshape((batch_size)) - gt_root_depth.reshape((batch_size))).detach().cpu().numpy()
                error_x, error_y = torch.zeros_like(gt_root_depth).detach().cpu().numpy(), \
                    torch.zeros_like(gt_root_depth).detach().cpu().numpy()
                if args.use_rootnet_xy_branch:
                    error_x, error_y = torch.abs(coord[:,0] - gt_root_trans[:,0]).detach().cpu().numpy(), \
                        torch.abs(coord[:,1] - gt_root_trans[:,1]).detach().cpu().numpy()
                
            MSELoss = torch.nn.MSELoss()
            SmoothL1Loss = torch.nn.SmoothL1Loss()
            L1Loss = torch.nn.L1Loss()

            mask = valid_mask_crop[:, args.reference_keypoint_id].unsqueeze(-1)
            if not args.multi_kp:
                if args.depth_loss_func == "l1":
                    loss = L1Loss(pred_root_depth, gt_root_depth)
                elif args.depth_loss_func == "mse":
                    loss = MSELoss(pred_root_depth, gt_root_depth)
                else:
                    raise NotImplementedError
                if args.use_rootnet_xy_branch:
                    if args.xy_loss_func == "l1":
                        loss += L1Loss(coord[:,0:2] * mask , gt_root_trans[:,0:2] * mask)
                    elif args.xy_loss_func == "mse":
                        loss += MSELoss(coord[:,0:2] * mask , gt_root_trans[:,0:2] * mask)
                    else:
                        raise NotImplementedError
            else:
                if args.depth_loss_func == "l1":
                    loss = L1Loss(pred_depths, gt_kp_depths)
                elif args.depth_loss_func == "mse":
                    loss = MSELoss(pred_depths, gt_kp_depths)
                else:
                    raise NotImplementedError
            
            if train:
                return loss
            else:
                return loss, error_depth, error_x, error_y
            

        def validate(ds):
            if ds == "dr":
                loader = ds_iter_test_dr
            elif ds == "photo" and urdf_robot_name != "baxter":
                loader = ds_iter_test_photo
            elif ds in ["azure", "kinect", "realsense", "orb"] and urdf_robot_name == "panda":
                loader = test_loader_dict[ds]
            ds = "_"+ds
            model.eval()
            loss_val = AverageValueMeter()
            losses_depth = AverageValueMeter()
            alldis = defaultdict(list)
            with torch.no_grad():
                for idx, sample in enumerate(tqdm(loader, dynamic_ncols=True)):
                    vloss, error_depth, error_x, error_y = farward_loss(args=args,input_batch=sample, device=device, model=model, train=False)
                    loss_val.add(vloss.detach().cpu().numpy())
                    alldis["deptherror"].extend(list(error_depth))
                    alldis["xerror"].extend(error_x)
                    alldis["yerror"].extend(error_y)

            mean_depth_error = np.mean(alldis["deptherror"])
            mean_x_error, mean_y_error = np.mean(alldis["xerror"]), np.mean(alldis["yerror"])
            writer.add_scalar('Val/rootz_loss'+ds, loss_val.mean , epoch)
            writer.add_scalar('Val/mean_depth_error'+ds, mean_depth_error, epoch)
            writer.add_scalar('Val/mean_x_error'+ds, mean_x_error, epoch)
            writer.add_scalar('Val/mean_y_error'+ds, mean_y_error, epoch)
            model.train()
            return mean_depth_error
        
        
        # train one epoch
        model.train()
        iterator = tqdm(ds_iter_train, dynamic_ncols=True)
        losses = AverageValueMeter()
        nowid = 0

        for batchid, sample in enumerate(iterator):
            optimizer.zero_grad()
            loss = farward_loss(args=args,input_batch=sample, device=device, model=model, train=True)
            loss.backward()
            if args.clip_gradient is not None:
                clipping_value = args.clip_gradient
                torch.nn.utils.clip_grad_norm_(model.parameters(), clipping_value)
            optimizer.step()
            losses.add(loss.detach().cpu().numpy())
            nowid = batchid
            
            if (batchid+1) % 100 == 0:    # Every 100 mini-batches/iterations
                writer.add_scalar('Train/loss', losses.mean , epoch * len(ds_iter_train) + batchid + 1)
                losses.reset()
                
            writer.add_scalar('LR/learning_rate_opti', optimizer.param_groups[0]['lr'],epoch * len(ds_iter_train) + batchid + 1)
            
        if (nowid+1) % 100 != 0:
            writer.add_scalar('Train/loss', losses.mean, epoch * len(ds_iter_train) + nowid)
            losses.reset()

        if args.use_schedule:
            lr_scheduler.step()
            
        mean_depth_error_dr = validate("dr")
        if urdf_robot_name != "baxter":
            mean_depth_error_photo = validate("photo")
        if urdf_robot_name == "panda":
            mean_depth_error_4real = {}
            mean_depth_error_allreal = 0.0
            for ds_short in ds_shorts:
                mean_depth_error_real = validate(ds_short)
                mean_depth_error_4real[ds_short] = mean_depth_error_real
                if ds_short in ["orb", "realsense"]:
                    mean_depth_error_allreal += mean_depth_error_allreal
                else:
                    mean_depth_error_allreal += 0.4 * mean_depth_error_allreal
    
        save_path_dr = os.path.join(ckpt_folder, 'curr_best_root_depth_model.pk')
        # save_path_real = os.path.join(ckpt_folder, 'curr_best_root_depth_onreal_model.pk')
        save_path_azure = os.path.join(ckpt_folder, 'curr_best_root_depth_azure_model.pk')
        save_path_kinect = os.path.join(ckpt_folder, 'curr_best_root_depth_kinect_model.pk')
        save_path_realsense = os.path.join(ckpt_folder, 'curr_best_root_depth_realsense_model.pk')
        save_path_orb = os.path.join(ckpt_folder, 'curr_best_root_depth_orb_model.pk')
        save_path = {"azure":save_path_azure, "kinect":save_path_kinect, "realsense":save_path_realsense, "orb":save_path_orb}
        save_path_allreal = os.path.join(ckpt_folder, 'curr_best_root_depth_allreal_model.pk')
        
        saves = {"dr":True, "azure":True, "kinect":True, "realsense":True, "orb":True ,"allreal":True}
        if os.path.exists(save_path_dr): 
            ckpt = torch.load(save_path_dr)
            if epoch <= ckpt["epoch"]: # prevent better model got covered during cluster rebooting 
                saves["dr"] = False
        for real_name in ["azure", "kinect", "realsense", "orb"]:
            if os.path.exists(save_path[real_name]): 
                ckpt_real = torch.load(save_path[real_name])
                if epoch <= ckpt_real["epoch"]: # prevent better model got covered during cluster rebooting 
                    saves[real_name] = False
        if os.path.exists(save_path_allreal): 
            ckpt_real = torch.load(save_path_allreal)
            if epoch <= ckpt_real["epoch"]: # prevent better model got covered during cluster rebooting 
                saves["allreal"] = False
        
        if saves["dr"]:
            if mean_depth_error_dr < curr_min_loss:
                curr_min_loss = mean_depth_error_dr
                if args.use_schedule:
                    last_epoch = lr_scheduler.last_epoch
                else:
                    last_epoch = -1
                torch.save({
                            'epoch': epoch,
                            'loss': mean_depth_error_dr,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'lr_scheduler_last_epoch':last_epoch,
                            }, save_path_dr)
        
        if urdf_robot_name == "panda":
            for real_name in ["azure", "kinect", "realsense", "orb"]:
                if saves[real_name]:
                    if mean_depth_error_4real[real_name] < curr_min_loss_4real[real_name]:
                        curr_min_loss_4real[real_name] = mean_depth_error_4real[real_name]
                        if args.use_schedule:
                            last_epoch = lr_scheduler.last_epoch
                        else:
                            last_epoch = -1
                        torch.save({
                                    'epoch': epoch,
                                    'loss': mean_depth_error_4real[real_name],
                                    'model_state_dict': model.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict(),
                                    'lr_scheduler_last_epoch':last_epoch,
                                    }, save_path[real_name])
            if saves["allreal"]:
                if mean_depth_error_allreal < curr_min_loss_allreal:
                    curr_min_loss_allreal = mean_depth_error_allreal
                    if args.use_schedule:
                        last_epoch = lr_scheduler.last_epoch
                    else:
                        last_epoch = -1
                    torch.save({
                                'epoch': epoch,
                                'loss': mean_depth_error_allreal,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'lr_scheduler_last_epoch':last_epoch,
                                }, save_path_allreal)
                  
    print("Training Finished !")
    writer.flush()