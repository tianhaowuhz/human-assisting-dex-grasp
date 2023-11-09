# Human-assist Dexterous Grasping

## Requirements
Details regarding installation of IsaacGym can be found here. We currently support the Preview Release 4 version of IsaacGym.

Pre-requisites
The code has been tested on Ubuntu 18.04/20.04 with Python 3.7/3.8. The minimum recommended NVIDIA driver version for Linux is 470.74 (dictated by support of IsaacGym).

It uses Anaconda to create virtual environments. To install Anaconda, follow instructions here.

Ensure that Isaac Gym works on your system by running one of the examples from the python/examples directory, like joint_monkey.py. Please follow troubleshooting steps described in the Isaac Gym Preview Release 4 install instructions if you have any trouble running the samples.

Once Isaac Gym is installed and samples work within your current python environment, install this repo:

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
  - [Install Global Dependencies](#install-global-dependencies)
  - [Install *Ball Rearrangement* Dependencies](#install-ball-rearrangement-environment)
  - [Install *Room Rearrangement* Dependencies](#install-room-rearrangement-environment)
- [Training](#training)
  - [Target Score Network](#training-the-target-score-network)
  - [Learning to control with RL and TarGF (Optional)](#learning-to-control-with-rl-and-targf-optional)
- [Evaluation](#evaluation)
- [Citation](#citation)
- [Contact](#contact)
- [License](#license)


## Installation

### Requirements
- Ubuntu >= 18.04
- Anaconda3 
- python >= 3.9
- pytorch >= 1.11.0
- pytorch_geometric >= 2.0.0
- pybullet >= 3.1.0
- tensorboard >= 2.6.0
- pytorch_fid >= 0.2.0
- imageio >= 2.9.0

### Install Global Dependencies

```
git clone https://github.com/AaronAnima/TarGF

cd TarGF

conda create -n targf python=3.9

conda activate targf

conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch

conda install pyg==2.0.4 -c pyg

pip install opencv-python tensorboard pytorch_fid ipdb imageio 
```

### Install *Ball* Rearrangement Environment

```
pip install gym pybullet

cd envs

git clone https://github.com/AaronAnima/EbOR # install Example-based Object Rearrangement (EbOR) environments

cd EbOR

pip install -e .

cd ../../
```


### Install *Room Rearrangement* Environment

Please follow the README in [this page](https://github.com/AaronAnima/TarGF/tree/main/envs/Room).

If you do not need to run this experiment, you can skip this procedure. 


## Training 

### Training the GraspGF
[TODO]

### Training the RL with GraspGF
[TODO]


## Evaluation
```
python main.py --config configs/targf_circlecluster.py --workdir CircleCluster_SAC_eval --mode test_policy
```




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




