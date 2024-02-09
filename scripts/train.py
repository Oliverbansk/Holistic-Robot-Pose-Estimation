import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import yaml
from lib.config import LOCAL_DATA_DIR
from lib.core.config import make_cfg
from scripts.train_depthnet import train_depthnet
from scripts.train_sim2real import train_sim2real
# from scripts.train_sim2real_real import train_sim2real_real
from scripts.train_full import train_full
    

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
        # train_sim2real_real(cfg)
        
        
    
