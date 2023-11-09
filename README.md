# Human-assist Dexterous Grasping

## Requirements
Details regarding installation of IsaacGym can be found here. We currently support the Preview Release 4 version of IsaacGym.

Pre-requisites
The code has been tested on Ubuntu 18.04/20.04 with Python 3.7/3.8. The minimum recommended NVIDIA driver version for Linux is 470.74 (dictated by support of IsaacGym).

It uses Anaconda to create virtual environments. To install Anaconda, follow instructions here.

Ensure that Isaac Gym works on your system by running one of the examples from the python/examples directory, like joint_monkey.py. Please follow troubleshooting steps described in the Isaac Gym Preview Release 4 install instructions if you have any trouble running the samples.

Once Isaac Gym is installed and samples work within your current python environment, install this repo:

## Installation
First clone this repo and install the environments.
```
git clone https://gitee.com/thwu17/ConDex.git
cd ConDex
git clone https://gitee.com/thwu17/ConDexEnv.git
cd ConDexEnv
pip install -e .
```
## Data 
put objects.tar.gz to ConDexEnv/assets/objects
```
cd objects
python data_process.py
```
## Running
```
cd Runners
python test_env.py
```
