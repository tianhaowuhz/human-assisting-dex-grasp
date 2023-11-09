# Human-assist Dexterous Grasping

## [Note] currently only available for testing
[[Website](https://sites.google.com/view/graspgf)] [[Arxiv](https://arxiv.org/abs/2309.06038)]


<img src="Demos/graspgf_mug.gif" align="middle" width="700"/> 

In this paper, we propose a novel task called **human-assisting dexterous grasping** that aims to train a policy for controlling a robotic hand's fingers to assist users in grasping diverse objects with diverse grasping poses.

<img src="Demos/pipeline.png" align="middle" width="700"/> 

We address this challenge by proposing an approach consisting of **two sub-modules**: a hand-object-conditional grasping primitive called Grasping Gradient Field (**GraspGF**), and a history-conditional **residual policy**. 

This repo is the official implementation of [GraspGF](https://arxiv.org/abs/2309.06038). Contents of this repo are as follows:

- [Installation](#installation)
  - [Requirements](#requirements)
- [Dataset](#dataset)
  - [Asset](#asset)
  - [ExpertDataset](#expert-dataset)
- [Training](#training)
  - [GraspGF](#training-the-graspgf)
  - [RL+GraspGF](#training-the-rl-with-graspgf)
- [Evaluation](#evaluation)
- [Acknowledgement](#acknowledgement)
- [Citation](#citation)
- [Contact](#contact)
- [License](#license)


## Installation

### Requirements
The code has been tested on Ubuntu 20.04 with Python 3.8. 

### Install

**IsaacGym**:

You can refer installation of IsaacGym [here](https://developer.nvidia.com/isaac-gym). We currently support the Preview Release 4 version of IsaacGym.


**Human-assisting Dexterous Grasping Environment**:
```
cd ConDexEnv
pip install -e .
```

**Pointnet2**:
```
cd Networks/pointnet2
pip install -e .
```

**Other Dependencies**

python package:
``` 
ipdb
tqdm
opencv-python
matplotlib
transforms3d
open3d
```

## Dataset

### Asset
you can download filterd grasping dataset from [Human-assisting Dexterous Grasping/Asset](https://drive.google.com/drive/folders/1fUKlUu1ifYQJ5_7NkmC1jpkEysLR9v8A) and put on following directopry.
```
ConDexEnv/assets
```
There are three sets in our dataset: train, seencategory and unseencategory. You can choose to only download one of them for simple demonstration.
### ExpertDataset
you can download filterd grasping dataset from [Human-assisting Dexterous Grasping/ExpertDatasets](https://drive.google.com/drive/folders/1fUKlUu1ifYQJ5_7NkmC1jpkEysLR9v8A) and put on current directory.

There are three types of data: 
1. Human trajectories: sampled from [Handover-Sim](https://github.com/NVlabs/handover-sim), which has to be downloaded.
```
ExpertDatasets/human_traj_200_all.npy
```
2. Pointcloud Buffer: contains pointcloud of objects we are using, you can download according to object dataset you download in asset.
```
ExpertDatasets/pcl_buffer_4096_train.pkl
ExpertDatasets/pcl_buffer_4096_seencategory.pkl
ExpertDatasets/pcl_buffer_4096_unseencategory.pkl
```
3. Human Grasp Pose: contains human grasp pose which can also be downloaded according to object dataset you download in asset.
```
ExpertDatasets/grasp_data/ground/*_oti.pth
ExpertDatasets/grasp_data/ground/*_rc_ot.pth
ExpertDatasets/grasp_data/ground/*_rc.pth
```


## Training 

### Training the GraspGF
[TODO]

### Training the RL with GraspGF
[TODO]



## Evaluation
```
sh ./rl_eval.sh
```
you can download pretrained checkpoint from [Human-assisting Dexterous Grasping/Ckpt](https://drive.google.com/drive/folders/17nWw5PZKNdpAPdlUKzHWY4aKWK4L8vus?usp=drive_link), and put on current directory for evluating pretrained model by adding following in rl_eval.sh.
```
--score_model_path="Ckpt/gf" \
--model_dir="Ckpt/gfppo.pt" \
```

## Acknowledgement
The code and dataset used in this project is built from these repository:

Environment: 

[NVIDIA-Omniverse/IsaacGymEnvs](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs)

[PKU-MARL/DexterousHands](https://github.com/PKU-MARL/DexterousHands)

Dataset: 

[PKU-EPIC/DexGraspNet](https://github.com/PKU-EPIC/DexGraspNet)

[PKU-EPIC/UniDexGrasp](https://github.com/PKU-EPIC/UniDexGrasp)

[NVlabs/handover-sim](https://github.com/NVlabs/handover-sim)

Diffusion:

[yang-song/score_sde_pytorch](https://github.com/yang-song/score_sde_pytorch)

[AaronAnima/TarGF](https://github.com/AaronAnima/TarGF)

Pointnet:

[hwjiang1510/GraspTTA](https://github.com/hwjiang1510/GraspTTA/tree/master)

Pointnet2:

[ck-kai/SGPA](https://github.com/ck-kai/SGPA)

## Citation
[TODO]
<!-- ```
@inproceedings{
wu2022targf,
title={Tar{GF}: Learning Target Gradient Field for Object Rearrangement},
author={Mingdong Wu and Fangwei Zhong and Yulong Xia and Hao Dong},
booktitle={Advances in Neural Information Processing Systems},
editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
year={2022},
url={https://openreview.net/forum?id=Euv1nXN98P3}
}
``` -->

## Contact
If you have any suggestion or questions, please get in touch at [thwu@stu.pku.edu.cn](thwu@stu.pku.edu.cn).

## LICENSE
<!-- GraspGF has an MIT license, as found in the [LICENSE](./LICENSE) file. -->
[TODO]




