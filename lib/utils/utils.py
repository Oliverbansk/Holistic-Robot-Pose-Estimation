import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import random
import shutil
from collections import OrderedDict, defaultdict
from lib.dataset.multiepoch_dataloader import MultiEpochDataLoader
from lib.dataset.samplers import PartialSampler
from pathlib import Path
import numpy as np
import torch
from lib.dataset.dream import DreamDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchnet.meter import AverageValueMeter
from tqdm import tqdm

def cast(obj, device, dtype=None):
    
    if isinstance(obj, (dict, OrderedDict)):
        for k, v in obj.items():
            if v is None:
                continue
            obj[k] = cast(torch.as_tensor(v),device)
            if dtype is not None:
                obj[k] = obj[k].to(dtype)
        return obj
    else:
        return obj.to(device)
    

def set_random_seed(seed):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    
def copy_and_rename(src_path, dest_path, new_filename):
    
    src_path = Path(src_path)
    dest_path = Path(dest_path)
    shutil.copy(src_path, dest_path)
    src_filename = src_path.name
    dest_filepath = dest_path / new_filename
    (dest_path / src_filename).replace(dest_filepath)
    
    
def create_logger(args):
    
    save_folder = os.path.join('experiments',  args.exp_name)
    ckpt_folder = os.path.join(save_folder,  'ckpt')
    log_folder = os.path.join(save_folder,  'log')
    os.makedirs(ckpt_folder, exist_ok=True)
    os.makedirs(log_folder, exist_ok=True)
    writer = SummaryWriter(log_dir=log_folder)
    copy_and_rename(args.config_path, save_folder, "config.yaml")
    
    return save_folder, ckpt_folder, log_folder, writer


def get_dataloaders(args):
    
    urdf_robot_name = args.urdf_robot_name
    train_ds_names = args.train_ds_names
    test_ds_name_dr = train_ds_names.replace("train_dr","test_dr")
    if urdf_robot_name != "baxter":
        test_ds_name_photo = train_ds_names.replace("train_dr","test_photo")
    if urdf_robot_name == "panda":
        
        test_ds_name_real = [train_ds_names.replace("synthetic/panda_synth_train_dr","real/panda-3cam_azure"),
                            train_ds_names.replace("synthetic/panda_synth_train_dr","real/panda-3cam_kinect360"),
                            train_ds_names.replace("synthetic/panda_synth_train_dr","real/panda-3cam_realsense"),
                            train_ds_names.replace("synthetic/panda_synth_train_dr","real/panda-orb")]
        
    rootnet_hw = (int(args.rootnet_image_size),int(args.rootnet_image_size))
    other_hw = (int(args.other_image_size),int(args.other_image_size))
    ds_train = DreamDataset(train_ds_names,
                            rootnet_resize_hw=rootnet_hw, 
                            other_resize_hw=other_hw, 
                            color_jitter=args.jitter, rgb_augmentation=args.other_aug, 
                            occlusion_augmentation=args.occlusion, occlu_p=args.occlu_p)
    ds_test_dr = DreamDataset(test_ds_name_dr,
                              rootnet_resize_hw=rootnet_hw, 
                              other_resize_hw=other_hw, 
                              color_jitter=False, rgb_augmentation=False, occlusion_augmentation=False) 
    if urdf_robot_name != "baxter":
        ds_test_photo = DreamDataset(test_ds_name_photo, 
                                     rootnet_resize_hw=rootnet_hw, 
                                     other_resize_hw=other_hw, 
                                     color_jitter=False, rgb_augmentation=False, occlusion_augmentation=False) 
    
    train_sampler = PartialSampler(ds_train, epoch_size=args.epoch_size)
    ds_iter_train = DataLoader(
        ds_train, 
        sampler=train_sampler, 
        batch_size=args.batch_size, 
        num_workers=args.n_dataloader_workers, 
        drop_last=False,
        pin_memory=True
    )
    ds_iter_train = MultiEpochDataLoader(ds_iter_train)

    test_loader_dict = {}
    ds_iter_test_dr = DataLoader(
        ds_test_dr, 
        batch_size=args.batch_size, 
        num_workers=args.n_dataloader_workers
    )
    test_loader_dict["dr"] = ds_iter_test_dr
    
    if urdf_robot_name != "baxter":
        ds_iter_test_photo = DataLoader(
            ds_test_photo, 
            batch_size=args.batch_size,
            num_workers=args.n_dataloader_workers
        )
        test_loader_dict["photo"] = ds_iter_test_dr
        
    if urdf_robot_name == "panda":
        ds_shorts = ["azure", "kinect", "realsense", "orb"]
        for ds_name, ds_short in zip(test_ds_name_real, ds_shorts):
            ds_test_real = DreamDataset(ds_name, 
                                        rootnet_resize_hw=rootnet_hw, 
                                        other_resize_hw=other_hw, 
                                        color_jitter=False, rgb_augmentation=False, occlusion_augmentation=False, 
                                        process_truncation=args.fix_truncation) 
            ds_iter_test_real = DataLoader(
                ds_test_real, 
                batch_size=args.batch_size, 
                num_workers=args.n_dataloader_workers
            )
            test_loader_dict[ds_short] = ds_iter_test_real
    
    print("len(ds_iter_train): ", len(ds_iter_train))
    print("len(ds_iter_test_dr): ", len(ds_iter_test_dr))
    if urdf_robot_name != "baxter":
        print("len(ds_iter_test_photo): ", len(ds_iter_test_photo))
    if urdf_robot_name == "panda":
        for ds_short in ds_shorts:
            print(f"len(ds_iter_test_{ds_short}): ", len(test_loader_dict[ds_short]))
    
    return ds_iter_train, test_loader_dict


def get_scheduler(args, optimizer, last_epoch):
    
    def lr_lambda_linear(epoch):
        if epoch < args.n_epochs_warmup:
            ratio = float(epoch+1)/float(args.n_epochs_warmup)
        elif epoch <= args.start_decay:
            ratio = 1.0
        elif epoch <= args.end_decay:
            ratio = (float(args.end_decay - args.final_decay * args.start_decay) - (float(1-args.final_decay) * epoch)) / float(args.end_decay - args.start_decay)
        else:
            ratio = args.final_decay
        return ratio
        
    def lr_lambda_exponential(epoch):
        base_ratio = 1.0
        ratio = base_ratio
        if epoch < args.n_epochs_warmup:
            ratio = float(epoch+1)/float(args.n_epochs_warmup)
        elif epoch <= args.start_decay:
            ratio = base_ratio
        elif epoch <= args.end_decay:
            ratio = (args.exponent)**(epoch-args.start_decay)
        else:
            ratio = (args.exponent)**(args.end_decay-args.start_decay)
        return ratio
    
    def lr_lambda_everyXepoch(epoch):
        ratio = (args.step_decay)**(epoch // args.step)
        if epoch >= args.end_decay:
            ratio = (args.step_decay)**(args.end_decay // args.step)
        return ratio
    
    if args.use_schedule:
        if args.schedule_type == "linear":
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda_linear, last_epoch=last_epoch)
        elif args.schedule_type == "exponential":
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda_exponential, last_epoch=last_epoch)
        elif args.schedule_type == "everyXepoch":
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda_everyXepoch, last_epoch=last_epoch)
    else:
        lr_scheduler = None
            
    return lr_scheduler


def resume_run(args, model, optimizer, device):

    curr_max_auc_4real = { "azure": 0.0, "kinect": 0.0, "realsense": 0.0, "orb": 0.0 }
    template = 'ckpt/curr_best_auc(add)_DATASET_model.pk'
    ckpt_paths = [template.replace("DATASET", name) for name in curr_max_auc_4real.keys()]
    
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
    curr_max_auc = checkpoint["auc_add"]
    
    for postfix, dsname in zip(ckpt_paths, curr_max_auc_4real.keys()):   
        model_path = os.path.join(resume_dir, postfix)
        ckpt = torch.load(model_path)
        curr_max_auc_onreal = ckpt["auc_add"]
        curr_max_auc_4real[dsname] = curr_max_auc_onreal
    
    return start_epoch, last_epoch, curr_max_auc, curr_max_auc_4real


def save_checkpoint(args, auc_adds, model, optimizer, ckpt_folder, epoch, lr_scheduler, curr_max_auc, curr_max_auc_4real):

    save_path_dr = os.path.join(ckpt_folder, 'curr_best_auc(add)_model.pk')
    save_path_azure = os.path.join(ckpt_folder, 'curr_best_auc(add)_azure_model.pk')
    save_path_kinect = os.path.join(ckpt_folder, 'curr_best_auc(add)_kinect_model.pk')
    save_path_realsense = os.path.join(ckpt_folder, 'curr_best_auc(add)_realsense_model.pk')
    save_path_orb = os.path.join(ckpt_folder, 'curr_best_auc(add)_orb_model.pk')
    save_path = {"azure":save_path_azure, "kinect":save_path_kinect, "realsense":save_path_realsense, "orb":save_path_orb}
    saves = {"dr":True, "azure":True, "kinect":True, "realsense":True, "orb":True }
    if os.path.exists(save_path_dr): 
        ckpt = torch.load(save_path_dr)
        if epoch <= ckpt["epoch"]: # prevent better model got covered during cluster rebooting 
            saves["dr"] = False
    for real_name in ["azure", "kinect", "realsense", "orb"]:
        if os.path.exists(save_path[real_name]): 
            ckpt_real = torch.load(save_path[real_name])
            if epoch <= ckpt_real["epoch"]: # prevent better model got covered during cluster rebooting 
                saves[real_name] = False
    
    if saves["dr"]:        
        if auc_adds["dr"] > curr_max_auc:
            curr_max_auc = auc_adds["dr"]
            last_epoch = lr_scheduler.last_epoch if args.use_schedule else -1
            torch.save({
                        'epoch': epoch,
                        'auc_add': curr_max_auc,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'lr_scheduler_last_epoch':last_epoch,
                        }, save_path_dr)
                    
    if args.urdf_robot_name == "panda":
        for real_name in ["azure", "kinect", "realsense", "orb"]:
            if saves[real_name]:
                if auc_adds[real_name] > curr_max_auc_4real[real_name]:
                    curr_max_auc_4real[real_name] = auc_adds[real_name]
                    last_epoch = lr_scheduler.last_epoch if args.use_schedule else -1
                    torch.save({
                                'epoch': epoch,
                                'auc_add': curr_max_auc_4real[real_name],
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'lr_scheduler_last_epoch':last_epoch,
                                }, save_path[real_name])