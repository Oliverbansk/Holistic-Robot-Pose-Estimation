# Real-time Holistic Robot Pose Estimation with Unknown States

<img src="asset/teaser.png" width="800"/>

## Introduction
This is the official PyTorch implementation of our paper "Real-time Holistic Robot Pose Estimation with Unknown States".

The overall framework is presented below.

<img src="asset/framework.jpg" width="800"/>

## Installation
This project's dependencies include python 3.9, pytorch 1.13 and CUDA 11.7.
The code is developed and tested on Ubuntu 20.04.

```bash
    pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117
    pip install -r requirements.txt
```

## Data and Model Preparation

In our work, we use the following data and pretrained model:
* The [DREAM datasets](https://drive.google.com/drive/folders/1uNK2n9wU4tRE07sM_r640wDhwmOwuxx6) consisting of both real and synthetic subsets, placed under `${ROOT}/data/dream/$`.
* The [URDF](https://drive.google.com/drive/folders/1x8WDx1uF4DovTH1HaB5CBziz70Zb4SnJ) (Unified Robotics Description Format) of robot Panda, Kuka and Baxter, placed under `${ROOT}/data/deps/$`.
* The [HRnet backbone weights](https://drive.google.com/file/d/1eqIftq1T_oIGhmCfkVYSM245Wj5xZaUo/view?) for pose estimation, placed under `${ROOT}/models/$`.
* The openly available [foreground segmentation model](https://drive.google.com/drive/folders/1PpXe3p5dJt9EOM-fwvJ9TNStTWTQFDNK?) of 4 real datasets of Panda from [CtRNet](https://github.com/ucsdarclab/CtRNet-robot-pose-estimation), placed under `${ROOT}/models/panda_segmentation/$`.

You can download the data and models through provided links. 
When finished, the directory tree should look like this. 
```
${ROOT}
|-- data
    |-- dream
    |   |-- real
    |   |   |-- panda-3cam_azure  
    |   |   |-- panda-3cam_kinect360
    |   |   |-- panda-3cam_realsense
    |   |   |-- panda-orb
    |   |-- synthetic
    |   |   |-- baxter_synth_test_dr
    |   |   |-- baxter_synth_train_dr
    |   |   |-- kuka_synth_test_dr
    |   |   |-- kuka_synth_test_photo
    |   |   |-- kuka_synth_train_dr
    |   |   |-- panda_synth_test_dr
    |   |   |-- panda_synth_test_photo
    |   |   |-- panda_synth_train_dr
    |-- deps
    |   |-- baxter-description
    |   |-- kuka-description
    |   |-- panda-description
|-- models
    |-- panda_segmentation
    |   |-- azure.pth
    |   |-- kinect.pth
    |   |-- orb.pth
    |   |-- realsense.pth
    |-- hrnet_w32-36af842e_roc.pth
```

## Train
We train our final model in a multi-stage fashion. All model is trained using a single NVIDIA V100 with 32GB GPU. Distributed training is also supported.

We use config files in `./config` to specify the training process. We recommend filling in the `exp_name` field in the config files with a unique name, as the checkpoints and event logs produced during training will be saved under `experiments/{exp_name}`.

### Synthetic Datasets

Firstly, pretrain the depthnet (root depth estimator) for 90 epochs for each robot arm:
```bash
python scripts/train.py --config configs/panda/depthnet.yaml
python scripts/train.py --config configs/kuka/depthnet.yaml
python scripts/train.py --config configs/baxter/depthnet.yaml
```

With depthnet pretrained, we can train the full network for 90 epochs:
```bash
python scripts/train.py --config configs/panda/full.yaml
python scripts/train.py --config configs/kuka/full.yaml
python scripts/train.py --config configs/baxter/full.yaml
```
To save your time when reproducing results of our paper, we provide readily-pretrained [depthnet model weights](https://drive.google.com/drive/folders/1rWC2bbA3U0IiZ7oDoKIVsWK_m4JkVarA?) for full network training. To use them, you can modify the `configs/{robot}/full.yaml` file by filling in the `pretrained_rootnet` field with the path of the downloaded `.pk` file. 

### Real Datasets of Panda

We employ self-supervised training for the 4 real datasets of Panda.

Firstly, train the model on synthetic dataset using `configs/panda/self_supervised/synth.yaml` for 90 epochs. Besure to fill in the `pretrained_rootnet` field with the path of the pretrained Panda depthnet weight in advance.

```bash
python scripts/train.py --config configs/panda/self_supervised/synth.yaml
```
The checkpoints generated will include checkpoints for further self-supervised training (e.g. `experiments/{exp_name}/ckpt/curr_best_auc(add)_azure_model.pk`). Besure to modify the `configs/panda/self_supervised/{real_dataset}.yaml` file by filling in the `pretrained_weight_on_synth` field with the path of the correspondent checkpoint in advance. 

```bash
python scripts/train.py --config configs/panda/self_supervised/azure.yaml
python scripts/train.py --config configs/panda/self_supervised/kinect360.yaml
python scripts/train.py --config configs/panda/self_supervised/realsense.yaml
python scripts/train.py --config configs/panda/self_supervised/orb.yaml
```

## Test
Each model generated in training is combined with its correspondent config file, which is automatically copied into the `experiments/{exp_name}/` directory as training starts. To evaluate models, simply run:
```bash
python scripts/test.py --exp_name {exp_name}
```

## Model Zoo
You can download our final models from Google Drive and evaluate them yourself.
|  Datasets |  Model Weight |
|-----------|-------------|
|  Panda  |               | 
|   Kuka  |               |
|  Baxter |               |


## Acknowledgment
This repo is built on the excellent work [RoboPose](https://github.com/ylabbe/robopose) and [CtRNet](https://github.com/ucsdarclab/CtRNet-robot-pose-estimation). Thank the authors for releasing their codes.