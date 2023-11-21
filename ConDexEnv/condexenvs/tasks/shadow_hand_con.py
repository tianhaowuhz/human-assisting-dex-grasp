# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from multiprocessing import managers
import numpy as np
import random
from collections import deque
from scipy.stats import gamma

import os
import sys
import time
from ipdb import set_trace
import cv2
import torch.multiprocessing as mp
from scipy import rand
import torch
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity
import open3d as o3d
import transforms3d
import pickle
from gym import spaces
from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from condexenvs.utils.torch_jit_utils import *
from condexenvs.tasks.base.vec_task import VecTask

from condexenvs.utils.utils import rodrigues_to_rotation, sample_from_circle, angle_from_vector
from condexenvs.utils.utils import farthest_point_sample, index_points, matrix_to_quaternion, plane2euler

from condexenvs.utils.misc import RewardNormalizer

from ipdb import set_trace
import warnings
warnings.filterwarnings("ignore", category=Warning)

translation_names = ['WRJTx', 'WRJTy', 'WRJTz']
rot_names = ['WRJRx', 'WRJRy', 'WRJRz']
joint_names = [
    'robot0:FFJ3',
    'robot0:FFJ2',
    'robot0:FFJ1',
    # 'robot0:FFJ0',
    'robot0:MFJ3',
    'robot0:MFJ2',
    'robot0:MFJ1',
    # 'robot0:MFJ0',
    'robot0:RFJ3',
    'robot0:RFJ2',
    'robot0:RFJ1',
    # 'robot0:RFJ0',
    'robot0:LFJ4',
    'robot0:LFJ3',
    'robot0:LFJ2',
    'robot0:LFJ1',
    # 'robot0:LFJ0',
    'robot0:THJ4',
    'robot0:THJ3',
    'robot0:THJ2',
    'robot0:THJ1',
    'robot0:THJ0'
]

inverse_joint_names = [
    'robot0:FFJ3',
    'robot0:MFJ3',
    'robot0:RFJ3',
    'robot0:LFJ3',
    'robot0:THJ4',
    'robot0:THJ2',
    'robot0:THJ1',
    'robot0:THJ0'
]

test_gf = False
local = False
debug = False
allow_obj_scale = False
add_obs_noise = False

invalid_object_types = [
'bigbird_crayola_24_crayons_scaled', 'bigbird_nutrigrain_apple_cinnamon_scaled', 'bigbird_vo5_extra_body_volumizing_shampoo_scaled', 'bigbird_band_aid_clear_strips_scaled', 'bigbird_chewy_dipps_peanut_butter_scaled', 'bigbird_nutrigrain_harvest_blueberry_bliss_scaled', 'bigbird_nutrigrain_chotolatey_crunch_scaled', 'bigbird_haagen_dazs_butter_pecan_scaled', 'bigbird_nature_valley_gluten_free_roasted_nut_crunch_almond_crunch_scaled', 'bigbird_quaker_chewy_chocolate_chip_scaled', 'bigbird_nutrigrain_cherry_scaled',
'PUNCH_DROP', 'CHILDREN_BEDROOM_CLASSIC', 'MIRACLE_POUNDING', 'MIRACLE_POUNDING', 'OVAL_XYLOPHONE', 'STEAK_SET', 'Super_Mario_3D_World_Deluxe_Set_yThuvW9vZed', 'TOOL_BELT', 'KID_ROOM_FURNITURE_SET_1', 'SORTING_BUS', 'PARENT_ROOM_FURNITURE_SET_1', 'Squirtin_Barnyard_Friends_4pk', 'HAMMER_BALL', 'SPEED_BOAT', 'POUNDING_MUSHROOMS', 'HAMMER_PEG', 'STACKING_BEAR_V04KKgGBn2A', 'CHILDRENS_ROOM_NEO', 'KITCHEN_FURNITURE_SET_1',
'pistol-664579680dc09267e1f2a1daf140ac9f',
]

episode_length = 51
class ShadowHandCon(VecTask):
    #####################################################################
    ###=========================init functions========================###
    #####################################################################
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render, seed=0):

        torch.manual_seed(seed) #cpu
        random.seed(seed)
        np.random.seed(seed)

        self.cfg = cfg

        self.mode = self.cfg["task"]["mode"]
        self.eval_times = self.cfg["task"]["eval_times"]
        self.randomize = self.cfg["task"]["randomize"]
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        '''
        camera configuration
        '''
        self.enable_camera_sensors = self.cfg["env"]["enableCameraSensors"]
        # if self.enable_camera_sensors:
        # set camera configuration
        self.cam_props = gymapi.CameraProperties()
        self.cam_props.width = 512
        self.cam_props.height = 512
        self.cam_pos = gymapi.Vec3(0.3, 0.0, 0.3)
        self.cam_target = gymapi.Vec3(0.0, 0.0, 0.0)
        self.vis_states_input = False
        '''
        hand configuration
        '''
        self.simple_hand = self.cfg["env"]["simpleHand"]
        self.has_base = self.cfg["env"]["hasBase"]
        self.fake_tendon = self.cfg["env"]["fakeTendon"]
        self.rot_obs_type = self.cfg["env"]["rotObsType"]
        self.constrained = self.cfg["env"]["constrained"]
        self.hand_rotation = self.cfg["env"]["handRotation"]
        self.use_human_rot = False
        self.start_joint_noise = self.cfg["env"]["startJointNoise"]
        self.traj_len = self.cfg["env"]["trajLen"]
        self.human_hand_speed = [0.05, 0.05]
        self.action_speed = 0.05
        self.detection_freq = 10
        self.custom_gen_data = self.cfg["env"]["customGenData"]

        self.gen_pcl_with_ground = self.cfg["env"]["genPclWithGround"]

        self.method = self.cfg["env"]["method"]        

        self.fingertips = ["robot0:ffdistal", "robot0:mfdistal", "robot0:rfdistal", "robot0:lfdistal", "robot0:thdistal"]
        self.fingertips_center = ["robot0:fftip_center", "robot0:mftip_center", "robot0:rftip_center", "robot0:lftip_center", "robot0:thtip_center"]
        self.fingertips_center_back = ["robot0:fftip_center_back", "robot0:mftip_center_back", "robot0:rftip_center_back", "robot0:lftip_center_back", "robot0:thtip_center_back"]
        self.hand_mount = ["robot0:palm"]
        self.num_fingertips = len(self.fingertips)

        self.collect_contact_info = False
        self.use_contact_sensor = False
            
        if self.use_contact_sensor:
            self.contact_sensor_names = ["robot0:ffdistal_fsr", "robot0:mfdistal_fsr", "robot0:rfdistal_fsr", "robot0:lfdistal_fsr", "robot0:thdistal_fsr"]
            self.contact_outside_sensor_names = ["robot0:ffdistal_fsr_outside", "robot0:mfdistal_fsr_outside", "robot0:rfdistal_fsr_outside", "robot0:lfdistal_fsr_outside", "robot0:thdistal_fsr_outside"]

        self.joint_range_type = 'dgn'

        '''
        environment parameter configuration
        '''
        self.reset_position_noise = self.cfg["env"]["resetPositionNoise"]
        self.reset_rotation_noise = self.cfg["env"]["resetRotationNoise"]
        self.reset_dof_pos_noise = self.cfg["env"]["resetDofPosRandomInterval"]
        self.reset_dof_vel_noise = self.cfg["env"]["resetDofVelRandomInterval"]

        # random force to object
        self.force_scale = self.cfg["env"].get("forceScale", 0.0)
        self.force_prob_range = self.cfg["env"].get("forceProbRange", [0.001, 0.1])
        self.force_decay = self.cfg["env"].get("forceDecay", 0.99)
        self.force_decay_interval = self.cfg["env"].get("forceDecayInterval", 0.08)

        self.friction = self.cfg["env"]["friction"]

        self.shadow_hand_dof_speed_scale = self.cfg["env"]["dofSpeedScale"]
        self.use_relative_control = self.cfg["env"]["useRelativeControl"]
        self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.reset_time = self.cfg["env"].get("resetTime", -1.0)
        self.print_success_stat = self.cfg["env"]["printNumSuccesses"]
        self.max_consecutive_successes = self.cfg["env"]["maxConsecutiveSuccesses"]
        self.av_factor = self.cfg["env"].get("averFactor", 0.1)
        self.reward_type = self.cfg["env"]["rewardType"]
        self.reward_normalize = self.cfg["env"]["rewardNormalize"]
        if self.reward_normalize:
            self.sim_reward_normalizer = RewardNormalizer(is_norm=True,name='sim')
        self.similarity_reward = self.cfg["env"]["similarityReward"]
        self.similarity_reward_freq = self.cfg["env"]["similarityRewardFreq"]
        self.contact_reward = self.cfg["env"]["contactReward"]
        self.any_contact_reward = self.cfg["env"]["anyContactReward"]
        self.any_contact_threshold = self.cfg["env"]["anyContactThreshold"]
        self.contact_time = self.cfg["env"]["contactTime"]
        self.collision_reward = self.cfg["env"]["collisionReward"]
        self.moving_reward = self.cfg["env"]["movingReward"]
        self.success_reward = self.cfg["env"]["successReward"]
        self.sample_goal = self.cfg["env"]["sampleGoal"]
        if 'goaldist' in self.reward_type:
            self.sample_goal = True
        self.sample_nearest_goal = False
        self.grad_scale = 1.0
        self.close_dis = self.cfg["env"]["closeDis"]
        print(f'close dis:{self.close_dis}')

        self.angle_thres = None

        self.up_axis = 'z'
        self.table_setting = True
        self.use_human_trajs = True
        self.traj_noise = 0.0
        self.env_mode = 'train'
        self.stack_frame_numbers = self.cfg["env"]["stackFrameNumber"]
        self.use_abs_stack_frame = self.cfg["env"]["absStackFrame"]
        self.wrist_gen = (self.method=='wristgen')
        if self.wrist_gen or (self.custom_gen_data):
            self.table_setting = True

        self.has_goal = False
        self.disable_collision = self.cfg["env"]["disableCollision"]
        self.robot_done = self.cfg["env"]["robotDone"]
        if self.method=='gf+rl' and not self.constrained:
            self.robot_done=True
        
        self.reset_sim_every_time = True
        self.diff_obj_scale = True
        self.data_gen = False
        self.cem_traj_gen = self.cfg["env"]["cemTrajGen"]
        
        if (self.method == 'filter' or self.method == 'gf') and self.data_gen:
            self.custom_gen_data = False
            self.reset_sim_every_time = False
            self.diff_obj_scale = False
        
        if self.cem_traj_gen:
            print('cem gen traj')
            self.custom_gen_data = False
        '''
        Object configuration
        '''
        # dataset setup
        self.obj_scale_number = 0.15
        self.dataset_type = self.cfg["env"]["asset"]["assetType"]
        if self.mode == 'train':
            self.sub_dataset_type = 'train'
            self.random_sample_grasp_label = True
            if self.data_gen:
                self.sub_dataset_type = 'seencategory'
                self.random_sample_grasp_label = False
        elif self.mode == 'eval':
            self.sub_dataset_type = self.cfg["env"]["asset"]["assetSubType"]
            self.obj_scale_number = 0.06
            self.random_sample_grasp_label = False

        if self.dataset_type == 'dexgraspnet':
            self.label_dir = os.path.join(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')),f'dexgraspnet_dataset/datasetv4.1/{self.sub_dataset_type}')
            self.data_dir = os.path.join(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')),f'dexgraspnet_dataset/datasetv4.1/{self.sub_dataset_type}')
            self.mesh_dir = os.path.join(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')),f'dexgraspnet_dataset/meshdata/{self.sub_dataset_type}')

        if "object_shapes" not in self.cfg['env'].keys() or self.cfg['env']['object_shapes'] is None:
            print('load object from dataset !!!!!!!!!!!!!!!!!!!!!!!!!!!')
            if self.mode == 'train':
                if self.sub_dataset_type == 'train':
                    self.shapes_all = [shape for shape in os.listdir(self.data_dir)] # Training Set
                elif self.sub_dataset_type == 'unseencategory':
                    self.shapes_all = [shape for shape in os.listdir(self.data_dir)] # Training Set
                elif self.sub_dataset_type == 'seencategory':
                    self.shapes_all = [shape for shape in os.listdir(self.data_dir)] # Training Set
            elif self.mode == 'eval':
                if self.sub_dataset_type == 'train':
                    self.shapes_all = [shape for shape in os.listdir(self.data_dir)] # Training Set
                elif self.sub_dataset_type == 'unseencategory':
                    self.shapes_all = [shape for shape in os.listdir(self.data_dir)] # Training Set
                elif self.sub_dataset_type == 'seencategory':
                    self.shapes_all = [shape for shape in os.listdir(self.data_dir)] # Training Set
        else:
            print('predfeined object !!!!!!!!!!!!!!!!!!!!!!!!!!!')
            self.shapes_all = self.cfg["env"]["object_shapes"]
        
        if self.custom_gen_data:
            self.custom_gen_data_dir = os.path.join(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../ExpertDatasets')),'grasp_data/ground/')
            if self.mode == 'train':
                if self.dataset_type == 'dexgraspnet':
                    if self.sub_dataset_type == 'train':
                        if self.method == 'filter':
                            self.custom_gen_data_name = 'train_eval'
                        else:
                            self.custom_gen_data_name = 'train'
            else:
                if self.sub_dataset_type == 'train':
                    self.custom_gen_data_name = 'train_eval'
                elif self.sub_dataset_type == 'unseencategory':
                    self.custom_gen_data_name = 'unseencategory'
                elif self.sub_dataset_type == 'seencategory':
                    self.custom_gen_data_name = 'seencategory'
                    
            with open(os.path.join(self.custom_gen_data_dir,f'{self.custom_gen_data_name}_rc_ot.pth'), 'rb') as f:
                self.shapes_all = pickle.load(f)
                self.valid_object_nums = len(self.shapes_all)
            print(self.custom_gen_data_name)

        for invalid_object in invalid_object_types:
            if invalid_object in self.shapes_all:
                print(invalid_object)
                self.shapes_all.pop(self.shapes_all.index(invalid_object))

        # print(len(self.shapes_all), self.sub_dataset_type)
        # pointcloud setup
        self.max_points_per_object = self.cfg["env"]["max_points_per_object"]
        self.points_per_object = self.cfg["env"]["points_per_object"]

        # for reset
        self.objects_minz = torch.zeros(self.cfg["env"]["numEnvs"], device=rl_device)
        '''
        observation and action dim setup
        '''
        if self.simple_hand:
            hand_dofs = 22
            num_actions = 18
        else:
            hand_dofs = 24
            num_actions = 20

        ### action ###
        # for base
        num_actions += 6
        self.cfg["env"]["numActions"] = num_actions

        ### observation ###
        # for "full_state" observation
        if self.rot_obs_type == 'quat':
            self.rot_obs_dim = 4
        else:
            self.rot_obs_dim = 3
        hand_full_state_dim = hand_dofs*3 + 13*5 + 6*5 + 3 + self.rot_obs_dim + num_actions + 7 + 3 + 3

        # get observation type
        if self.method == 'gf' or self.method == 'gf+rl' or self.method == 'filter' or self.method == 'wristgen':
            self.cfg["env"]["observationType"] = 'gf'

        self.obs_type = self.cfg["env"]["observationType"]
        self.sub_obs_type = self.cfg["env"]["subObservationType"]

        if not (self.obs_type in ["full_state", "gf"]):
            raise Exception(
                "Unknown type of observations!\nobservationType should be one of: [full_state, gf]")

        print("Obs type:", self.obs_type)

        # get observation dims according observation type
        if self.method == 'gf+rl':
            gf_observation_number = 0
            if "joint" in self.sub_obs_type:
                gf_observation_number += 18
            if "wrist" in self.sub_obs_type:
                gf_observation_number += 7*self.stack_frame_numbers
            if "pcl" in self.sub_obs_type:
                gf_observation_number += self.points_per_object*3
            if "gf" in self.sub_obs_type:
                gf_observation_number += 18
            if "absfingertip" in self.sub_obs_type:
                gf_observation_number += self.num_fingertips * 3
            if "disfingertip" in self.sub_obs_type:
                gf_observation_number += self.num_fingertips
            if "fingertipjoint" in self.sub_obs_type:
                gf_observation_number += self.num_fingertips - 1
            if "objpose" in self.sub_obs_type:
                gf_observation_number += 7
            if "diso2o" in self.sub_obs_type:
                gf_observation_number += 1
            if "goal" in self.sub_obs_type:
                gf_observation_number += 18
        else:
            # dof, base pose, pcl, obj pose, envid, fingertip pos, obj scale
            gf_observation_number = 18+7+self.points_per_object*3+7+1+self.num_fingertips*3+1

        self.gfrl_relative = (self.method == 'gf+rl')

        self.num_obs_dict = {
            "full_state": hand_full_state_dim,
            "gf": gf_observation_number,
        }

        self.asymmetric_obs = self.cfg["env"]["asymmetric_observations"]

        num_states = 0
        if self.asymmetric_obs:
            num_states = 211

        self.cfg["env"]["numObservations"] = self.num_obs_dict[self.obs_type]
        self.cfg["env"]["numStates"] = num_states

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)
        
        self._init_meta()
        self._reset_simulator(env_init=True)
        # redefine action space
        if self.method == 'gf+rl':
            if self.robot_done:
                if self.constrained:
                    self.act_space = spaces.Box(np.ones(self.num_actions-5)*-1, np.ones(self.num_actions-5))
                else:
                    self.act_space = spaces.Box(np.ones(self.num_actions+1)*-1, np.ones(self.num_actions+1))
            else:
                if self.constrained:
                    self.act_space = spaces.Box(np.ones(self.num_actions-6)*-1, np.ones(self.num_actions-6))
                else:
                    self.act_space = spaces.Box(np.ones(self.num_actions+1)*-1, np.ones(self.num_actions+1))
        else:
            self.act_space = spaces.Box(np.ones(self.num_actions) * -self.action_speed, np.ones(self.num_actions) * self.action_speed)
                
    def _init_meta(self):
        self.dt = self.cfg["sim"]["dt"]
        self.up_axis_idx = 2 if self.up_axis == 'z' else 1 # index of up axis: Y=1, Z=2
        self._init_meta_data()   

    def get_h2o_state(self, dex_data):
        hand_dof = dex_data[:,:18].clone()
        hand_pos_2_w = dex_data[:,18:21].clone()
        hand_quat_2_w = dex_data[:,21:25].clone()
        obj_pos_2_w = dex_data[:,self.points_per_object*3+25:self.points_per_object*3+25+3].clone()
        obj_quat_2_w = dex_data[:,self.points_per_object*3+25+3:self.points_per_object*3+25+7].clone()
        h2o_pos, h2o_quat = self.transform_target2source(obj_quat_2_w, obj_pos_2_w, hand_quat_2_w, hand_pos_2_w)
        dex_data[:,18:21] = h2o_pos
        dex_data[:,21:25] = h2o_quat
        return dex_data

    def _reset_simulator(self, env_init=False, pose_id=None, env_ids=None, obj_scales=None):
        # set_trace()
        if not env_init:
            # destory sim
            if self.headless == False:
                self.gym.destroy_viewer(self.viewer)
            
            if len(self.cameras_handle) > 0:
                for (cam_id, cam) in enumerate(self.cameras_handle):
                    self.gym.destroy_camera_sensor(self.sim, self.envs[cam_id], cam)
                
            for env in self.envs:
                self.gym.destroy_env(env)

            self.gym.destroy_sim(self.sim)
        else:
            self.test_times = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        
        # common reset
        self.envs = []
        # if self.enable_camera_sensors:
        self.cameras_handle = []
        self.object_init_state = []
        self.hand_start_states = []
        self.hand_indices = []
        self.fingertip_indices = []
        self.object_indices = []
        if self.has_goal:
            self.goal_object_indices = []
            self.goal_states = []
        
        #reset object pclbuffer type id
        if self.method == 'gf' or self.method == 'gf+rl' or self.method == 'filter' or self.constrained or self.method == 'wristgen':
            self.obj_pcl_buf = torch.zeros((self.num_envs, self.max_points_per_object, 3), device=self.device, dtype=torch.float)
        
        if (self.method=='gf+rl') and self.mode=='train':
            self.object_type_per_env = random.sample(self.object_types, self.num_envs)
        elif self.mode=='eval':
            self.object_type_per_env = self.object_types[:self.num_envs]
        else:
            self.object_type_per_env = self.object_types
            
        self.object_ids_per_env = []
        for object_type in self.object_type_per_env:
            self.object_ids_per_env.append(self.obj_type_id[object_type])

        # reset object scale
        if self.diff_obj_scale:
            if pose_id!=None:
                for env_id in env_ids:
                    # take object type
                    tmp_object_type = self.object_types[env_id]
                    tmp_object_id = self.obj_type_id[tmp_object_type]
                    tmp_object_grasp_pose_num = self.grasp_poses_num[tmp_object_id]
                    assert pose_id[env_id] <= tmp_object_grasp_pose_num , f"grasp pose id invalid, total grasp poses number is: {tmp_object_grasp_pose_num}"
                    tmp_grasp_id = pose_id[env_id]
                    obj_scale = self.obj_scales[tmp_object_id][tmp_grasp_id].copy()
                    self.cur_obj_scales[env_id] = obj_scale
            elif obj_scales!=None:
                self.cur_obj_scales = obj_scales
            elif self.method=='gf+rl':
                self.random_move = torch_rand_float(-0.5, 0.5, (self.num_envs, 2), device=self.device)
                if self.random_sample_grasp_label:
                    # TODO not use pbj grasp pose num
                    self.meta_data['data']['obj_grasp_pose_num'] = np.ones(len(self.meta_data['data']['obj_grasp_pose_num']))
                    env_id_idx = 25+self.points_per_object*3+7
                    self.grasp_poses = torch.zeros((len(self.object_types),1,3121),device=self.device, dtype=torch.float32)
                    for (i,object_type) in enumerate(self.object_types):
                        obj_id_in_ori = self.filtered_data_oti[object_type]
                        grasp_poses_id = (self.filtered_data[:,env_id_idx]==obj_id_in_ori).nonzero(as_tuple=False).squeeze().reshape(-1)
                        tmp_grasp_poses = self.filtered_data[grasp_poses_id,:].clone()
                        tmp_grasp_poses = self.get_h2o_state(tmp_grasp_poses)
                        far_grasp_idxs = farthest_point_sample(tmp_grasp_poses[:,:25].reshape(1,-1,25),npoint=2,device=self.device,init=self.grasp_poses_ids[object_type])
                        
                        if int(far_grasp_idxs[0][1]) in self.far_sample_hist[object_type]:
                            new_random_init = np.random.choice(len(grasp_poses_id))
                            grasp_poses_id = grasp_poses_id[new_random_init]
                            self.grasp_poses_ids[object_type] = new_random_init
                            self.far_sample_hist[object_type] = [new_random_init]
                        else:
                            self.grasp_poses_ids[object_type] = int(far_grasp_idxs[0][1])
                            self.far_sample_hist[object_type].append(int(far_grasp_idxs[0][1]))
                            grasp_poses_id = grasp_poses_id[far_grasp_idxs[0][1]]
                        self.grasp_poses[i,:,:] = self.filtered_data[grasp_poses_id].reshape(1,1,-1)
                    print(object_type, self.grasp_poses_ids[object_type])
                candidate_grasps_num = self.meta_data['data']['obj_grasp_pose_num']
                if self.reset_sim_every_time:
                    candidate_grasps_num = np.array(candidate_grasps_num)[self.object_ids_per_env]

                self.cur_grasp_pose_idx = np.zeros(self.num_envs, dtype=int)
                if self.mode=='eval':
                    self.cur_grasp_pose_idx = self.test_times.cpu().numpy()
                else:
                    self.cur_grasp_pose_idx = np.floor((candidate_grasps_num-self.cur_grasp_pose_idx)*np.random.rand(self.num_envs) + self.cur_grasp_pose_idx)
                
                point_cloud_idx = 25 + self.points_per_object * 3
                self.cur_obj_scales = self.grasp_poses[self.object_ids_per_env,self.cur_grasp_pose_idx,point_cloud_idx+7+15+1:point_cloud_idx+7+15+2].clone()
            else:
                self.random_move = torch_rand_float(-0.5, 0.5, (self.num_envs, 2), device=self.device)
                self.cur_obj_scales = torch.ones(self.num_envs, device=self.device, dtype=torch.float)*0.1
        else:
            self.cur_obj_scales = torch.ones(self.num_envs, device=self.device, dtype=torch.float)*self.obj_scale_number

        self.shadow_hands = []

        if env_init:
            '''
            init buffer for create envs
            '''
            
            '''
            start pose for hand, object, goal object
            '''
            # hand pose
            self.shadow_hand_start_pose = gymapi.Transform()
            if not self.has_base:
                self.shadow_hand_start_pose.p = gymapi.Vec3(*get_axis_params(0.2, self.up_axis_idx))

            # object pose
            self.object_start_pose = gymapi.Transform()
            self.object_start_pose.p = gymapi.Vec3()
            self.object_start_pose.p.x = 0
            self.object_start_pose.p.y = -0.2
            self.object_start_pose.p.z = 0

            pose_dy, pose_dz = -0.39, 0.10  
            if not self.has_base:
                self.shadow_hand_start_pose.p.y = self.object_start_pose.p.y - pose_dy

            if self.has_goal:
                # goal pose
                self.goal_displacement = gymapi.Vec3(-0.2, -0.06, 0.12)
                self.goal_displacement_tensor = to_torch(
                    [self.goal_displacement.x, self.goal_displacement.y, self.goal_displacement.z], device=self.device)
                self.goal_start_pose = gymapi.Transform()
                self.goal_start_pose.p = self.object_start_pose.p + self.goal_displacement
                self.goal_start_pose.p.z = 5

        self._create_sim(env_init=env_init)

    def _create_sim(self, env_init=False):
        # create envs, sim and viewer
        self.sim_initialized = False

        # create sim
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        if env_init:
            self._config_asset_options()
        
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)), env_init=env_init)

        self.gym.prepare_sim(self.sim)
        self.sim_initialized = True

        self.set_viewer()
        if env_init:
            self._allocate_buffers()

        self._configure_viewer()
        self._init_wrappers()
    
    def _configure_viewer(self):
        '''
        viewer setup
        '''
        if self.viewer != None:
            cam_pos = gymapi.Vec3(1.0, 0.0, 0.2)
            cam_target = gymapi.Vec3(0.0, 0.0, 0.2)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def _init_wrappers(self):
        '''
        init wrapper tensors 
        '''
        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        if self.use_contact_sensor:
            contact_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)

        if self.obs_type == "full_state" or self.asymmetric_obs or self.method=='gf+rl':
            sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
            self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, self.num_fingertips * 6)

            dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
            self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self.num_shadow_hand_dofs)
            
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        if self.use_contact_sensor:
            self.gym.refresh_net_contact_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.shadow_hand_default_dof_pos = torch.zeros(self.num_shadow_hand_dofs, dtype=torch.float, device=self.device)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.shadow_hand_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_shadow_hand_dofs]
        self.shadow_hand_dof_pos = self.shadow_hand_dof_state[..., 0]
        self.shadow_hand_dof_vel = self.shadow_hand_dof_state[..., 1]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]

        # actually state for all actor in sim
        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)
        self.hand_positions = self.root_state_tensor[:, 0:3]
        self.hand_orientations = self.root_state_tensor[:, 3:7]
        self.hand_linvels = self.root_state_tensor[:, 7:10]
        self.hand_angvels = self.root_state_tensor[:, 10:13]
        self.saved_root_tensor = self.root_state_tensor.clone() 
        if self.use_contact_sensor:
            self.contact_tensor = gymtorch.wrap_tensor(contact_tensor).view(self.num_envs, -1)

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        self.global_indices = torch.arange(self.num_envs * 3, dtype=torch.int32, device=self.device).view(self.num_envs, -1)
        self.x_unit_tensor = to_torch([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = to_torch([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = to_torch([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        self.av_factor = to_torch(self.av_factor, dtype=torch.float, device=self.device)

        # object apply random forces parameters
        self.force_decay = to_torch(self.force_decay, dtype=torch.float, device=self.device)
        self.force_prob_range = to_torch(self.force_prob_range, dtype=torch.float, device=self.device)
        self.random_force_prob = torch.exp((torch.log(self.force_prob_range[0]) - torch.log(self.force_prob_range[1]))
                                           * torch.rand(self.num_envs, device=self.device) + torch.log(self.force_prob_range[1]))
        self.rb_forces = torch.zeros((self.num_envs, self.num_bodies, 3), dtype=torch.float, device=self.device)

        '''
        init extra buffer
        '''
        self.trajs = torch.ones((self.num_envs,100,25),device=self.device)* -100
        self.trajs_len = torch.zeros(self.num_envs,device=self.device)
        
        self.reset_goal_buf = self.reset_buf.clone()
        self.lift_buf = torch.zeros_like(self.reset_buf)
        self.lift_successes = torch.zeros_like(self.reset_buf)
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        

        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]

    def _config_asset_options(self):
        ''' 
        load shadow hand asset
        '''
        self.shadow_hand_asset_file = os.path.normpath("mjcf/open_ai_assets/hand/shadow_hand.xml")
        if "asset" in self.cfg["env"]:
            self.shadow_hand_asset_file = os.path.normpath(self.cfg["env"]["asset"].get("assetFileName", self.shadow_hand_asset_file))
        
        if self.has_base:
            if self.use_contact_sensor:
                self.shadow_hand_asset_file = os.path.normpath("mjcf/open_ai_assets/hand/shadow_hand_simple_with_base_sensor.xml")
            else:
                self.shadow_hand_asset_file = os.path.normpath("mjcf/open_ai_assets/hand/shadow_hand_simple_with_base.xml")
        
        if self.dataset_type=='dexgraspnet':
            if self.has_base:
                self.shadow_hand_asset_file = os.path.normpath("mjcf/open_ai_assets/hand/shadow_hand_simple_with_base_dgn.xml")
            else:
                self.shadow_hand_asset_file = os.path.normpath("mjcf/open_ai_assets/hand/shadow_hand_simple_dgn.xml")
                
        self.shadowhand_asset_options = gymapi.AssetOptions()
        self.shadowhand_asset_options.flip_visual_attachments = False
        # TODO maybe add dof later
        if self.has_base:
            self.shadowhand_asset_options.fix_base_link = True
        else:
            self.shadowhand_asset_options.fix_base_link = False
        self.shadowhand_asset_options.collapse_fixed_joints = True
        self.shadowhand_asset_options.disable_gravity = True
        self.shadowhand_asset_options.thickness = 0.001
        # TODO seems can help robot not flying
        self.shadowhand_asset_options.angular_damping = 100
        self.shadowhand_asset_options.linear_damping = 100

        if self.physics_engine == gymapi.SIM_PHYSX:
            self.shadowhand_asset_options.use_physx_armature = True
        # Note - DOF mode is set in the MJCF file and loaded by Isaac Gym
        self.shadowhand_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        '''
        object and goal asset option setup
        '''
        ### object ###
        self.object_asset_options = gymapi.AssetOptions()
        self.object_asset_options.density = 500
        if (self.method == 'gf' or not self.table_setting):
            self.object_asset_options.disable_gravity = True
        # object_asset_options.use_mesh_materials = True
        self.object_asset_options.override_com = True
        self.object_asset_options.override_inertia = True
        # object_asset_options.vhacd_enabled = True
        # object_asset_options.vhacd_params = gymapi.VhacdParams()
        # object_asset_options.vhacd_params.resolution = 100000
        self.object_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        self.object_asset_options.convex_decomposition_from_submeshes = True

        if self.has_goal:
            ### goal ###
            self.goal_object_asset_options = gymapi.AssetOptions()
            self.goal_object_asset_options.disable_gravity = True
            self.goal_object_asset_options.override_com = True
            self.goal_object_asset_options.override_inertia = True
            self.goal_object_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

    def _load_hand_asset(self):
        self.shadow_hand_asset = self.gym.load_asset(self.sim, self.asset_root, self.shadow_hand_asset_file, self.shadowhand_asset_options)
        if not self.fake_tendon:
            self.num_shadow_hand_tendons = self.gym.get_asset_tendon_count(self.shadow_hand_asset)
            # tendon set up
            limit_stiffness = 30
            t_damping = 0.1
            relevant_tendons = ["robot0:T_FFJ1c", "robot0:T_MFJ1c", "robot0:T_RFJ1c", "robot0:T_LFJ1c"]
            tendon_props = self.gym.get_asset_tendon_properties(self.shadow_hand_asset)

            for i in range(self.num_shadow_hand_tendons):
                for rt in relevant_tendons:
                    if self.gym.get_asset_tendon_name(self.shadow_hand_asset, i) == rt:
                        tendon_props[i].limit_stiffness = limit_stiffness
                        tendon_props[i].damping = t_damping
            self.gym.set_asset_tendon_properties(self.shadow_hand_asset, tendon_props)

        # create fingertip force sensors, if needed
        self.fingertip_handles = [self.gym.find_asset_rigid_body_index(self.shadow_hand_asset, name) for name in self.fingertips]
        if self.obs_type == "full_state" or self.asymmetric_obs or self.method=='gf+rl':
            sensor_pose = gymapi.Transform()
            for ft_handle in self.fingertip_handles:
                self.gym.create_asset_force_sensor(self.shadow_hand_asset, ft_handle, sensor_pose)
        
        # num body shape dof actuator change
        self.shadow_hand_rigid_body_names = self.gym.get_asset_rigid_body_names(self.shadow_hand_asset)
        self.num_shadow_hand_bodies = self.gym.get_asset_rigid_body_count(self.shadow_hand_asset)
        self.num_shadow_hand_shapes = self.gym.get_asset_rigid_shape_count(self.shadow_hand_asset)
        self.num_shadow_hand_dofs = self.gym.get_asset_dof_count(self.shadow_hand_asset)
        self.num_shadow_hand_actuators = self.gym.get_asset_actuator_count(self.shadow_hand_asset)
        self.shadow_hand_rb_count = self.gym.get_asset_rigid_body_count(self.shadow_hand_asset)
        # compute aggregate size
        self.max_agg_bodies = self.num_shadow_hand_bodies + 200
        self.max_agg_shapes = self.num_shadow_hand_shapes + 200

        # get shadow_hand dof properties, loaded by Isaac Gym from the MJCF file
        self.shadow_hand_dof_props = self.gym.get_asset_dof_properties(self.shadow_hand_asset)

        self.shadow_hand_dof_props['stiffness'] = 10000.0
        self.shadow_hand_dof_props['damping'] = 0.0

        self.shadow_hand_dof_lower_limits = []
        self.shadow_hand_dof_upper_limits = []
        self.shadow_hand_dof_default_pos = []
        self.shadow_hand_dof_default_vel = []

        for i in range(self.num_shadow_hand_dofs):
            self.shadow_hand_dof_lower_limits.append(self.shadow_hand_dof_props['lower'][i])
            self.shadow_hand_dof_upper_limits.append(self.shadow_hand_dof_props['upper'][i])
            self.shadow_hand_dof_default_pos.append(0.0)
            self.shadow_hand_dof_default_vel.append(0.0)
    
    def _get_hand_asset_info(self):
        self.actuated_dof_names = ['robot0:FFJ3', 'robot0:FFJ2', 'robot0:FFJ1', 'robot0:MFJ3', 'robot0:MFJ2', 'robot0:MFJ1', 'robot0:RFJ3', 'robot0:RFJ2', 'robot0:RFJ1', 'robot0:LFJ4', 'robot0:LFJ3', 'robot0:LFJ2', 'robot0:LFJ1', 'robot0:THJ4', 'robot0:THJ3', 'robot0:THJ2', 'robot0:THJ1', 'robot0:THJ0']
        self.unactuated_dof_names = ['robot0:FFJ0', 'robot0:MFJ0', 'robot0:RFJ0', 'robot0:LFJ0']
        close_dof_names = ['robot0:FFJ2', 'robot0:FFJ1', 'robot0:FFJ0', 'robot0:MFJ2', 'robot0:MFJ1', 'robot0:MFJ0', 'robot0:RFJ2', 'robot0:RFJ1', 'robot0:RFJ0', 'robot0:LFJ2', 'robot0:LFJ1', 'robot0:LFJ0', 'robot0:THJ3', 'robot0:THJ1', 'robot0:THJ0']
        tip_actuated_dof_names = ['robot0:FFJ1', 'robot0:MFJ1', 'robot0:RFJ1','robot0:LFJ1', 'robot0:THJ1', 'robot0:THJ0']
        if self.has_base:
            hand_base_dof_names = ['robot0:baseJX', 'robot0:baseJY', 'robot0:baseJZ', 'robot0:baseJROLL', 'robot0:baseJPITCH', 'robot0:baseJYAW']

        self.actuated_dof_indices = [self.gym.find_asset_dof_index(self.shadow_hand_asset, name) for name in self.actuated_dof_names]
        self.unactuated_dof_indices = [self.gym.find_asset_dof_index(self.shadow_hand_asset, name) for name in self.unactuated_dof_names]
        self.close_dof_indices = [self.gym.find_asset_dof_index(self.shadow_hand_asset, name) for name in close_dof_names]
        self.tip_actuated_dof_indices = [self.gym.find_asset_dof_index(self.shadow_hand_asset, name) for name in tip_actuated_dof_names]
        if self.has_base:
            self.hand_base_dof_indices = [self.gym.find_asset_dof_index(self.shadow_hand_asset, name) for name in hand_base_dof_names]

        self.actuated_dof_indices = to_torch(self.actuated_dof_indices, dtype=torch.long, device=self.device)
        self.unactuated_dof_indices = to_torch(self.unactuated_dof_indices, dtype=torch.long, device=self.device)
        self.close_dof_indices = to_torch(self.close_dof_indices, dtype=torch.long, device=self.device)
        self.tip_actuated_dof_indices = to_torch(self.tip_actuated_dof_indices, dtype=torch.long, device=self.device)
        
        self.tip_actuated_dof_indices_in_states = []
        for i in range(self.num_shadow_hand_dofs):
            if i in self.tip_actuated_dof_indices:
                self.tip_actuated_dof_indices_in_states.append(int(torch.where(self.actuated_dof_indices==i)[0]))
        self.tip_actuated_dof_indices_in_states = to_torch(self.tip_actuated_dof_indices_in_states, dtype=torch.long, device=self.device)

        if self.has_base:
            self.hand_base_dof_indices = to_torch(self.hand_base_dof_indices, dtype=torch.long, device=self.device)

        self.knuckle_dof_indices = to_torch([0,3,6,9,10,13,14,15], dtype=torch.long, device=self.device)
        self.middle_dof_indices = to_torch([1,4,7,11,16], dtype=torch.long, device=self.device)
        self.distal_dof_indices = to_torch([2,5,8,12,17], dtype=torch.long, device=self.device)

        self.shadow_hand_dof_lower_limits = to_torch(self.shadow_hand_dof_lower_limits, device=self.device)
        self.shadow_hand_dof_upper_limits = to_torch(self.shadow_hand_dof_upper_limits, device=self.device)
        self.shadow_hand_dof_range = self.shadow_hand_dof_upper_limits - self.shadow_hand_dof_lower_limits
        self.shadow_hand_dof_default_pos = to_torch(self.shadow_hand_dof_default_pos, device=self.device)
        self.shadow_hand_dof_default_vel = to_torch(self.shadow_hand_dof_default_vel, device=self.device)

        self.fingertip_center_handles = [self.gym.find_asset_rigid_body_index(self.shadow_hand_asset, name) for name in self.fingertips_center]
        self.fingertip_center_back_handles = [self.gym.find_asset_rigid_body_index(self.shadow_hand_asset, name) for name in self.fingertips_center_back]
        self.hand_mount_handle = self.gym.find_asset_rigid_body_index(self.shadow_hand_asset, 'robot0:hand mount')
        self.shadow_hand_rigid_body_handles = [self.gym.find_asset_rigid_body_index(self.shadow_hand_asset, name) for name in self.shadow_hand_rigid_body_names]

        self.fingertip_handles = to_torch(self.fingertip_handles, dtype=torch.long, device=self.device)
        self.fingertip_center_handles = to_torch(self.fingertip_center_handles, dtype=torch.long, device=self.device)
        self.fingertip_center_back_handles = to_torch(self.fingertip_center_back_handles, dtype=torch.long, device=self.device)
        self.hand_mount_handle = to_torch(self.hand_mount_handle, dtype=torch.long, device=self.device)
        self.shadow_hand_rigid_body_handles = to_torch(self.shadow_hand_rigid_body_handles, dtype=torch.long, device=self.device)

    def _create_envs(self, num_envs, spacing, num_per_row, env_init=False):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)
        
        self._load_hand_asset()

        # if env_init:
        self._get_hand_asset_info()
        '''
        collision setup
        '''
        if self.method == 'gf' or self.disable_collision:
            hand_collision_bitmask = 1
            object_collision_bitmask = 1
        else:
            hand_collision_bitmask = -1
            object_collision_bitmask = 0
        
        st_time = time.time()
        '''
        start create
        '''
        for i in range(num_envs):
            # set_trace()
            ### decide object type ###
            object_type = self.object_type_per_env[i]
            # print(i, object_type)

            ### load object pointcloud for sampling ###
            if self.method == 'gf' or self.method == 'gf+rl' or self.method == 'filter' or self.method == 'wristgen':
                if self.dataset_type=='dexgraspnet':
                    if self.diff_obj_scale:
                        self.obj_pcl_buf[i,:,:] = to_torch(self.obj_pcl_buf_all[object_type],device=self.device) * self.cur_obj_scales[i]
                    else:
                        self.obj_pcl_buf[i,:,:] = to_torch(self.obj_pcl_buf_all[object_type],device=self.device) * self.obj_scale_number
                else:
                    object_pcl_path = os.path.join(self.data_dir, object_type, 'pcd', '600.pcd')
                    object_pcl = o3d.io.read_point_cloud(object_pcl_path)
                    # scale object 0.1, and set center to 0, to match obj
                    object_pcl = object_pcl.scale(0.1, [0,0,0])
                    object_pcl_points = np.asarray(object_pcl.points)
                    object_pcl_points = to_torch(object_pcl_points, device=self.device)
                    total_object_pcl_numer = object_pcl_points.size()[0]
                    for num_point in range(self.max_points_per_object):
                        self.obj_pcl_buf[i,num_point,:] = object_pcl_points[num_point%total_object_pcl_numer,:]

            ### object asset setup ###
            if self.dataset_type == 'dexgraspnet':
                if self.diff_obj_scale:
                    object_asset_file = os.path.join(f'dexgraspnet_dataset/meshdata/{self.sub_dataset_type}', object_type, f"coacd/decomposed-{str(int(self.cur_obj_scales[i]*100)).rjust(3,'0')}.urdf")
                else:
                    object_asset_file = os.path.join(f'dexgraspnet_dataset/meshdata/{self.sub_dataset_type}', object_type, f"coacd/decomposed-{str(int(self.obj_scale_number*100)).rjust(3,'0')}.urdf")

            object_asset = self.gym.load_asset(self.sim, self.asset_root, object_asset_file, self.object_asset_options)

            if self.has_goal:
                goal_asset = self.gym.load_asset(self.sim, self.asset_root, object_asset_file, self.goal_object_asset_options)

            object_rb_count = self.gym.get_asset_rigid_body_count(object_asset)
            self.object_rb_handles = list(range(self.shadow_hand_rb_count, self.shadow_hand_rb_count + object_rb_count))


            ### create env instance ###
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )

            # aggregate_mode
            if self.aggregate_mode >= 1:
                self.gym.begin_aggregate(env_ptr, self.max_agg_bodies, self.max_agg_shapes, True)

            ### add hand - collision filter = -1 to use asset collision filters set in mjcf loader  ###
            shadow_hand_actor = self.gym.create_actor(env_ptr, self.shadow_hand_asset, self.shadow_hand_start_pose, "hand", i, hand_collision_bitmask, 0)

            self.hand_start_states.append([self.shadow_hand_start_pose.p.x, self.shadow_hand_start_pose.p.y, self.shadow_hand_start_pose.p.z,
                                           self.shadow_hand_start_pose.r.x, self.shadow_hand_start_pose.r.y, self.shadow_hand_start_pose.r.z, self.shadow_hand_start_pose.r.w,
                                           0, 0, 0, 0, 0, 0])
            self.gym.set_actor_dof_properties(env_ptr, shadow_hand_actor, self.shadow_hand_dof_props)
            hand_idx = self.gym.get_actor_index(env_ptr, shadow_hand_actor, gymapi.DOMAIN_SIM)
            self.hand_indices.append(hand_idx)

            # hand friction reset
            hand_actor_handle = self.gym.get_actor_handle(env_ptr, shadow_hand_actor)
            hand_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, hand_actor_handle)
            for hand_rigid_id in range(len(hand_shape_props)):
                hand_shape_props[hand_rigid_id].friction = self.friction
            self.gym.set_actor_rigid_shape_properties(env_ptr, hand_actor_handle, hand_shape_props)

            # enable DOF force sensors, if needed
            if self.obs_type == "full_state" or self.asymmetric_obs or self.method=='gf+rl':
                self.gym.enable_actor_dof_force_sensors(env_ptr, shadow_hand_actor)

            ### add object ###
            object_handle = self.gym.create_actor(env_ptr, object_asset, self.object_start_pose, "object", i, object_collision_bitmask, 0)
            # rigid shape props
            object_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, object_handle)
            for object_rigid_id in range(len(object_shape_props)):
                object_shape_props[object_rigid_id].friction = self.friction
            self.gym.set_actor_rigid_shape_properties(env_ptr, object_handle, object_shape_props)
            # body props
            object_body_props = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
            for object_rigid_id in range(len(object_body_props)):
                object_body_props[object_rigid_id].mass = 0.1
            self.gym.set_actor_rigid_body_properties(env_ptr, object_handle, object_body_props)
            if self.dataset_type=='dexgraspnet' and allow_obj_scale:
                self.gym.set_actor_scale(env_ptr,object_handle,0.1)

            self.object_init_state.append([self.object_start_pose.p.x, self.object_start_pose.p.y, self.object_start_pose.p.z,
                                           self.object_start_pose.r.x, self.object_start_pose.r.y, self.object_start_pose.r.z, self.object_start_pose.r.w,
                                           0, 0, 0, 0, 0, 0])
            object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
            self.object_indices.append(object_idx)

            if self.has_goal:
                ### add goal object ###
                goal_handle = self.gym.create_actor(env_ptr, goal_asset, self.goal_start_pose, "goal_object", i + self.num_envs, object_collision_bitmask, 0)
                if self.dataset_type=='dexgraspnet' and allow_obj_scale:
                    self.gym.set_actor_scale(env_ptr,goal_handle,0.1)
                goal_object_idx = self.gym.get_actor_index(env_ptr, goal_handle, gymapi.DOMAIN_SIM)
                self.goal_object_indices.append(goal_object_idx)

            ### add camera ###
            if self.enable_camera_sensors:
                # create camera actor
                camera_handle = self.gym.create_camera_sensor(env_ptr, self.cam_props)
                self.cameras_handle.append(camera_handle)
                if self.vis_states_input:
                    point_cloud_idx = 25 + self.points_per_object * 3
                    cur_obj_pos = self.vis_states[i][point_cloud_idx:point_cloud_idx+3]
                    self.cam_pos = gymapi.Vec3(0.3+cur_obj_pos[0], 0.0+cur_obj_pos[1], 0.3+cur_obj_pos[2])
                    self.cam_target = gymapi.Vec3(0.0+cur_obj_pos[0], 0.0+cur_obj_pos[1], 0.0+cur_obj_pos[2])
                else:
                    cur_obj_pos = self.random_move[i].clone()
                    self.cam_pos = gymapi.Vec3(0.3+cur_obj_pos[0], 0.0+cur_obj_pos[1], 0.3)
                    self.cam_target = gymapi.Vec3(0.0+cur_obj_pos[0], 0.0+cur_obj_pos[1], 0.0)
                self.gym.set_camera_location(camera_handle, env_ptr, self.cam_pos, self.cam_target)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)
            
            ### save handles ###
            self.envs.append(env_ptr)
            self.shadow_hands.append(shadow_hand_actor)

        et_time = time.time()
        print(f'create env in:{et_time-st_time}')

        '''
        init handles to tensor
        '''
        # we are not using new mass values after DR when calculating random forces applied to an object,
        # which should be ok as long as the randomization range is not too big
        object_rb_props = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
        self.object_rb_masses = [prop.mass for prop in object_rb_props]
        if self.use_contact_sensor:
            sensor_handles = [self.gym.find_actor_rigid_body_handle(env_ptr, shadow_hand_actor, sensor_name) for sensor_name in
                            self.contact_sensor_names]
            sensor_outside_handle = [self.gym.find_actor_rigid_body_handle(env_ptr, shadow_hand_actor, sensor_name) for sensor_name in
                            self.contact_outside_sensor_names]

            self.sensor_handle_indices = to_torch(sensor_handles, device=self.device, dtype=torch.int64)
            self.sensor_outside_handle_indices = to_torch(sensor_outside_handle, device=self.device, dtype=torch.int64)

        self.object_init_state = to_torch(self.object_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        if self.has_goal:
            self.goal_states = self.object_init_state.clone()
            # self.goal_states[:, self.up_axis_idx] -= 0.04
            self.goal_states[:, self.up_axis_idx] -= 10
            self.goal_init_state = self.goal_states.clone()
        self.hand_start_states = to_torch(self.hand_start_states, device=self.device).view(self.num_envs, 13)

        self.object_rb_handles = to_torch(self.object_rb_handles, dtype=torch.long, device=self.device)
        self.object_rb_masses = to_torch(self.object_rb_masses, dtype=torch.float, device=self.device)

        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)
        self.object_indices = to_torch(self.object_indices, dtype=torch.long, device=self.device)

        if self.has_goal:
            self.goal_object_indices = to_torch(self.goal_object_indices, dtype=torch.long, device=self.device)

    def _destroy(self):
        self.gym.destroy_sim(self.sim)
        if self.has_viewer:
            self.gym.destroy_viewer(self.sim)

    def _allocate_buffers(self):
        """Allocate the observation, states, etc. buffers.

        These are what is used to set observations and states in the environment classes which
        inherit from this one, and are read in `step` and other related functions.

        """

        # allocate buffers
        self.obs_buf = torch.zeros(
            (self.num_envs, self.num_obs), device=self.device, dtype=torch.float)
        self.states_buf = torch.zeros(
            (self.num_envs, self.num_states), device=self.device, dtype=torch.float)
        

        self.rew_buf = torch.zeros(
                self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(
            self.num_envs, device=self.device, dtype=torch.long)
        self.timeout_buf = torch.zeros(
             self.num_envs, device=self.device, dtype=torch.long)
        self.progress_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self.randomize_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self.extras = {}

        self.frames = deque([], maxlen=self.stack_frame_numbers)

        if self.sample_goal:
            self.rl_goal_pose = torch.zeros(
                (self.num_envs, 18), device=self.device, dtype=torch.float)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)

        if self.table_setting:
            pass
        elif self.method == 'gf'  or self.method == 'filter' or not self.table_setting:
            plane_params.distance = 1.0
        self.gym.add_ground(self.sim, plane_params)
        
    def train(self):
        self.env_mode = 'train'
        self.graphics_device_id = -1
        self.enable_camera_sensors = False

    def eval(self, vis=False):
        self.env_mode = 'eval'
        if vis:
            self.graphics_device_id = 1
            self.enable_camera_sensors = True

    def _init_meta_data(self):
        self.asset_root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets'))

        self.meta_data = self.cfg
        # add meta data for data
        self.meta_data['data']={}
        self.meta_data['data']['data_path'] = os.path.join(self.asset_root, self.cfg["env"]["asset"]["assetFileNameObjects"])
        self.meta_data['data']['obj_num'] = len(self.shapes_all)
        self.meta_data['data']['obj_types'] = self.shapes_all

        self.object_types = []
        if self.reset_sim_every_time:
            for i in range(len(self.shapes_all)):
                ### decide object type ###
                object_type = self.shapes_all[i%len(self.shapes_all)]
                self.object_types.append(object_type)
        else:
            for i in range(self.num_envs):
                ### decide object type ###
                object_type = self.shapes_all[i%len(self.shapes_all)]
                self.object_types.append(object_type)
        
        self.obj_type_id = {}
        for (env_id,shape) in enumerate(self.shapes_all):
            self.obj_type_id[shape] = env_id
        
        obj_pcl_buf_all_path = os.path.join(self.asset_root, f'../../ExpertDatasets/pcl_buffer_4096_{self.sub_dataset_type}.pkl')
        with open(obj_pcl_buf_all_path, 'rb') as f:
                self.obj_pcl_buf_all = pickle.load(f)
                
        self.obj_scales = []
        '''
        gras pose meta data
        '''
        if self.custom_gen_data:
            self.grasp_poses_ids = {}
            self.far_sample_hist = {}

            filtered_data_dir = os.path.join(self.asset_root, f'../../ExpertDatasets/grasp_data/ground/{self.custom_gen_data_name}_rc.pth')
            oti_dir = os.path.join(self.asset_root, f'../../ExpertDatasets/grasp_data/ground/{self.custom_gen_data_name}_oti.pth')
            with open(filtered_data_dir, 'rb') as f:
                filtered_data = pickle.load(f)
                self.filtered_data = to_torch(filtered_data, device=self.device)
            with open(oti_dir, 'rb') as f:
                self.filtered_data_oti = pickle.load(f)

            if self.random_sample_grasp_label:
                env_id_idx = 25+self.points_per_object*3+7
                self.grasp_poses = torch.zeros((len(self.object_types),1,3121),device=self.device, dtype=torch.float32)
                self.grasp_poses_num = []
                for (i,object_type) in enumerate(self.object_types):
                    obj_id_in_ori = self.filtered_data_oti[object_type]
                    grasp_poses_id = (self.filtered_data[:,env_id_idx]==obj_id_in_ori).nonzero(as_tuple=False).squeeze().reshape(-1)
                    self.grasp_poses_num.append(len(self.filtered_data[grasp_poses_id]))
                    grasp_poses_id_local = np.random.choice(len(grasp_poses_id))
                    grasp_poses_id = grasp_poses_id[grasp_poses_id_local]
                    # set_trace()
                    self.grasp_poses[i,:,:] = self.filtered_data[grasp_poses_id].reshape(1,1,-1)
                    self.grasp_poses_ids[object_type] = grasp_poses_id_local
                    self.far_sample_hist[object_type] = []
                    # print(object_type,self.grasp_poses_ids[object_type])
            else:
                env_id_idx = 25+self.points_per_object*3+7
                self.grasp_pose_per_object = int(self.filtered_data.size(0)/self.valid_object_nums)
                self.grasp_poses = torch.zeros((len(self.object_types),self.grasp_pose_per_object,3121),device=self.device, dtype=torch.float32)
                self.grasp_poses_num = []
                for (i,object_type) in enumerate(self.object_types):
                    obj_id_in_ori = self.filtered_data_oti[object_type]
                    grasp_poses_id = (self.filtered_data[:,env_id_idx]==obj_id_in_ori).nonzero(as_tuple=False).squeeze()
                    # self.grasp_poses.append(filtered_data[grasp_poses_id])
                    self.grasp_poses[i,:,:] = self.filtered_data[grasp_poses_id].reshape(1,len(self.filtered_data[grasp_poses_id]),-1)
                    self.grasp_poses_num.append(len(self.filtered_data[grasp_poses_id]))
                # self.grasp_poses = filtered_data.reshape(self.num_envs,40,-1)
        else:
            self.grasp_poses = []
            self.grasp_poses_num = []
            self.obj_poses = []
            for (obj_number,object_type) in enumerate(self.object_types):
                if self.dataset_type=='dexgraspnet':
                    grasp_pose_dir = os.path.join(self.label_dir, object_type)
                    grasp_poses = []
                    obj_poses = []
                    obj_scales = []

                    for grasp_pose in np.sort(os.listdir(grasp_pose_dir)):
                        data_dict = np.load(os.path.join(self.label_dir,object_type, grasp_pose), allow_pickle=True)
                        qpos = data_dict['qpos'].item()
                        rot = np.array(transforms3d.euler.euler2quat(*[qpos[name] for name in rot_names]))
                        rot = np.array([*rot[1:], rot[0]])
                        trans = np.array([qpos[name] for name in translation_names])

                        hand_qpos = []
                        for name in joint_names:
                            if name in inverse_joint_names:
                                hand_qpos.append(-qpos[name])
                            else:
                                hand_qpos.append(qpos[name])
                        hand_qpos = np.array(hand_qpos)

                        grasp_pose = np.concatenate((trans,rot,hand_qpos),axis=0) 
                        obj_scale = round(1/data_dict['scale'],2)
                        if (not self.diff_obj_scale) and (obj_scale!=self.obj_scale_number):
                            continue
                        plane = data_dict['plane']
                        obj_translation, obj_euler = plane2euler(plane, axes='sxyz')
                        obj_quat = transforms3d.euler.euler2quat(obj_euler[0], obj_euler[1], (obj_euler[2]+0.1234567890), axes='sxyz')
                        obj_quat = np.array([*obj_quat[1:], obj_quat[0]])
                        obj_pose = np.concatenate((obj_translation,obj_quat),axis=0)

                        grasp_poses.append(grasp_pose)
                        obj_scales.append(obj_scale)
                        obj_poses.append(obj_pose)

                        if self.cem_traj_gen:
                            break

                    self.grasp_poses.append(grasp_poses)
                    self.grasp_poses_num.append(len(grasp_poses))
                    self.obj_scales.append(obj_scales)
                    self.obj_poses.append(obj_poses)

            grasp_example_006_total = []
            grasp_example_008_total = []
            grasp_example_010_total = []
            grasp_example_012_total = []
            grasp_example_015_total = []
            object_typee = []
            for (i,op) in enumerate(self.obj_poses):
                grasp_example_006 = np.sum(np.array(self.obj_scales[i])==0.06)
                grasp_example_008 = np.sum(np.array(self.obj_scales[i])==0.08)
                grasp_example_010 = np.sum(np.array(self.obj_scales[i])==0.1)
                grasp_example_012 = np.sum(np.array(self.obj_scales[i])==0.12)
                grasp_example_015 = np.sum(np.array(self.obj_scales[i])==0.15)

                object_typee.append(i)
                grasp_example_006_total.append(grasp_example_006)
                grasp_example_008_total.append(grasp_example_008)
                grasp_example_010_total.append(grasp_example_010)
                grasp_example_012_total.append(grasp_example_012)
                grasp_example_015_total.append(grasp_example_015)
                if self.data_gen:
                    print('-----------------------')
                    print(f'0.06scale:{grasp_example_006}')
                    print(f'0.08scale:{grasp_example_008}')
                    print(f'0.1scale:{grasp_example_010}')
                    print(f'0.12scale:{grasp_example_012}')
                    print(f'0.15scale:{grasp_example_015}')
                assert(grasp_example_006+grasp_example_008+grasp_example_010+grasp_example_012+grasp_example_015==len(self.obj_scales[i]))
        self.meta_data['data']['obj_grasp_pose_num'] =  self.grasp_poses_num
        self.meta_data['data']['obj_grasp_pose_candidates'] =  self.grasp_poses

        '''
        trajctory velocity
        '''
        if self.use_human_trajs:
            self.human_pattern = {}

            vel_data_dir = os.path.join(self.asset_root, '../../ExpertDatasets/human_traj_200_all.npy')
            all_human_trajs= np.load(vel_data_dir,allow_pickle=True).tolist()
            human_trajs = {}
            for (i,ht) in enumerate(all_human_trajs):
                if self.mode=='train' and i < 150:
                    human_trajs[ht] = all_human_trajs[ht]
                    self.human_traj_pattern_num = 150
                elif self.mode=='eval' and i >= 150:
                    human_trajs[ht] = all_human_trajs[ht]
                    self.human_traj_pattern_num = 50

            traj_len = episode_length * 2
            self.human_traj_len = episode_length * 2

            delta_rot_roll = []
            delta_rot_pitch = []
            delta_rot_yaw = []
            
            # get human traj
            resampled_human_traj = {}
            for (i,human_traj) in enumerate(human_trajs):
                human_traj_len = len(human_trajs[human_traj]['x_pos'])
                final_pos = to_torch([human_trajs[human_traj]['x_pos'][-1],human_trajs[human_traj]['y_pos'][-1],human_trajs[human_traj]['z_pos'][-1]],device=self.device)
                for traj_step in range(human_traj_len):
                    cur_pos =  to_torch([human_trajs[human_traj]['x_pos'][traj_step],human_trajs[human_traj]['y_pos'][traj_step],human_trajs[human_traj]['z_pos'][traj_step]],device=self.device)
                    if F.pairwise_distance(final_pos, cur_pos) < 0.175:
                        if human_traj not in resampled_human_traj:
                            resampled_human_traj[human_traj] = {}
                            resampled_human_traj[human_traj]['x_pos'] = [human_trajs[human_traj]['x_pos'][traj_step]]
                            resampled_human_traj[human_traj]['y_pos'] = [human_trajs[human_traj]['y_pos'][traj_step]]
                            resampled_human_traj[human_traj]['z_pos'] = [human_trajs[human_traj]['z_pos'][traj_step]]
                            resampled_human_traj[human_traj]['roll_pos'] = [human_trajs[human_traj]['roll_pos'][traj_step]]
                            resampled_human_traj[human_traj]['pitch_pos'] = [human_trajs[human_traj]['pitch_pos'][traj_step]]
                            resampled_human_traj[human_traj]['yaw_pos'] = [human_trajs[human_traj]['yaw_pos'][traj_step]]
                        else:                    
                            resampled_human_traj[human_traj]['x_pos'].append(human_trajs[human_traj]['x_pos'][traj_step])
                            resampled_human_traj[human_traj]['y_pos'].append(human_trajs[human_traj]['y_pos'][traj_step])
                            resampled_human_traj[human_traj]['z_pos'].append(human_trajs[human_traj]['z_pos'][traj_step])
                            resampled_human_traj[human_traj]['roll_pos'].append(human_trajs[human_traj]['roll_pos'][traj_step])
                            resampled_human_traj[human_traj]['pitch_pos'].append(human_trajs[human_traj]['pitch_pos'][traj_step])
                            resampled_human_traj[human_traj]['yaw_pos'].append(human_trajs[human_traj]['yaw_pos'][traj_step])

            human_trajs = resampled_human_traj

            # get human trajectory pattern
            for axis in ['x_pos', 'y_pos', 'z_pos', 'roll_pos', 'pitch_pos', 'yaw_pos']:
                human_patterns_axis = torch.zeros((self.human_traj_pattern_num,traj_len),device=self.device)
                valid_order = 0
                for (i,human_traj) in enumerate(human_trajs):
                    pattern = np.array(human_trajs[human_traj][axis])
                    ori_traj_len = len(pattern)
                    if ori_traj_len >= traj_len:
                        sample_interval = np.floor(ori_traj_len/traj_len)
                        sample_traj_steps = np.arange(0, ori_traj_len, sample_interval, dtype=np.int32)[:traj_len]
                        sample_traj_steps += ori_traj_len - (sample_traj_steps[-1]+1)
                        sampled_traj = pattern[sample_traj_steps]

                        if self.use_human_rot:
                            if axis == 'roll_pos':
                                delta_rot_roll.append(sampled_traj[-1]-sampled_traj[0])
                            elif axis == 'pitch_pos':
                                delta_rot_pitch.append(sampled_traj[-1]-sampled_traj[0])
                            elif axis == 'yaw_pos':
                                delta_rot_yaw.append(sampled_traj[-1]-sampled_traj[0])
                        else:
                            if axis == 'roll_pos':
                                delta_rot_roll.append(abs(sampled_traj[-1]-sampled_traj[0]))
                            elif axis == 'pitch_pos':
                                delta_rot_pitch.append(abs(sampled_traj[-1]-sampled_traj[0]))
                            elif axis == 'yaw_pos':
                                delta_rot_yaw.append(abs(sampled_traj[-1]-sampled_traj[0]))

                        if sampled_traj[0] < sampled_traj[-1]:
                            sampled_traj = sampled_traj[::-1]
                        max_pos = max(sampled_traj)
                        min_pos = min(sampled_traj)
                        
                        sampled_traj = (sampled_traj - min_pos)/(max_pos-min_pos)
                        human_patterns_axis[valid_order,:] = to_torch(sampled_traj, device=self.device)
                        valid_order+=1
                # assert (human_patterns_axis[:,0]==1).all() and (human_patterns_axis[:,-1]==0).all()
                self.human_pattern[axis] = human_patterns_axis

        self.rotation_dist_params = gamma.fit(delta_rot_roll+delta_rot_pitch+delta_rot_yaw)
        print('meta data generated')
    def get_meta_data(self):
        return self.meta_data

    #####################################################################
    ###========================reward functions=======================###
    #####################################################################
    def compute_reward(self):
        if self.method == 'gf+rl':
            self.compute_gf_hand_reward()

    # gf reward function
    def compute_gf_hand_reward(self):
        self.rew_buf[:] = 0
        if self.constrained:
            self.reset_buf = torch.where(self.progress_buf >= self.trajs_len, torch.ones_like(self.reset_buf), self.reset_buf)
        else:
            self.reset_buf = torch.where(self.progress_buf >= self.max_episode_length, torch.ones_like(self.reset_buf), self.reset_buf)

        if self.mode == 'eval':
            self.reset_buf[self.eval_finish_env_ids] *=0
        
        time_step_increase = self.progress_buf / self.trajs_len
        final_dist = F.pairwise_distance(self.con_final_hand_pose[:,:3],self.con_final_object_pose[:,:3])
        cur_dis_o2h =  F.pairwise_distance(self.cur_hand_pose[:,:3],self.object_pos)
        
        self.extras['action_value'] = torch.mean(abs(self.scale),-1)
        self.extras['action_value_std'] = torch.std(self.scale,dim=1)

        '''
        similarity reward
        '''
        if "similarity" in self.reward_type:
            next_hand_states = scale(self.obs_buf[:,:18].clone(),self.shadow_hand_dof_lower_limits[self.actuated_dof_indices], self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
            # state change according to grad
            delta_hand_dof_state = next_hand_states - self.cur_hand_states
            if "ori_similarity" in self.reward_type:
                similarity_reward = torch.sum(delta_hand_dof_state * self.grad/(abs(self.grad)+1e-5), -1)/(self.action_speed*18)
                similarity_reward *= self.similarity_reward

                direction = delta_hand_dof_state * self.grad

                if self.reward_normalize:
                    similarity_reward = torch.tensor(self.sim_reward_normalizer.update(similarity_reward.cpu().numpy()) * self.similarity_reward, device=self.device)
                    self.extras['RunningMean_sim'] = torch.tensor([self.sim_reward_normalizer.reward_mean],device=self.device)
                    self.extras['RunningStd_sim'] = torch.tensor([self.sim_reward_normalizer.reward_std], device=self.device)

                self.extras['ori_similarity'] = similarity_reward
                self.extras['diff_direction'] = torch.sum((direction<0),1)

            if self.progress_buf[0]%self.similarity_reward_freq==0:
                self.rew_buf[:] += similarity_reward
        else:
            next_hand_states = scale(self.obs_buf[:,:18].clone(),self.shadow_hand_dof_lower_limits[self.actuated_dof_indices], self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
            delta_hand_dof_state = next_hand_states - self.cur_hand_states
            direction = delta_hand_dof_state * self.grad
            self.extras['diff_direction'] = torch.sum((direction<0),1)
        
        '''
        obj moving and rotation
        '''
        next_obj_states = self.object_pose.clone()
        # moving
        init_env_ids = (self.progress_buf<(self.trajs_len*0.2)).nonzero(as_tuple=False).squeeze(0)
        delta_obj_moving = F.pairwise_distance(next_obj_states[:,:3],self.cur_obj_states[:,:3])
        delta_obj_moving[init_env_ids] = 0
        delta_no_move_ids = (delta_obj_moving < 1e-3).nonzero()
        delta_obj_moving[delta_no_move_ids] = 0
        self.extras['delta_objmoving'] = delta_obj_moving
        target_obj_moving = F.pairwise_distance(next_obj_states[:,:3],self.init_object_pose[:,:3])
        target_obj_moving[init_env_ids] = 0
        target_no_move_ids = (target_obj_moving < 1e-2).nonzero()
        target_obj_moving[target_no_move_ids] = 0
        self.extras['target_objmoving'] = target_obj_moving
        # rotation
        cur_obj_axis = quat_axis(next_obj_states[:,3:7],0)
        previous_obj_axis = quat_axis(self.cur_obj_states[:,3:7],0)
        init_obj_axis = quat_axis(self.init_object_pose[:,3:7],0)
        delta_dot = (cur_obj_axis*previous_obj_axis).sum(dim=-1)
        target_dot = (cur_obj_axis*init_obj_axis).sum(dim=-1)

        '''
        distance reward
        '''        
        if 'fingertip_dis' in self.reward_type:
            # compute each finger distance to object
            dist_finger1_to_closest_point_on_object = torch.min(F.pairwise_distance(self.sampled_pcl_abs,self.fingertip_pos[:, 0, :].reshape(-1,1,3).repeat(1,self.points_per_object,1)),-1)[0].reshape(self.num_envs,1)
            dist_finger2_to_closest_point_on_object = torch.min(F.pairwise_distance(self.sampled_pcl_abs,self.fingertip_pos[:, 1, :].reshape(-1,1,3).repeat(1,self.points_per_object,1)),-1)[0].reshape(self.num_envs,1)
            dist_finger3_to_closest_point_on_object = torch.min(F.pairwise_distance(self.sampled_pcl_abs,self.fingertip_pos[:, 2, :].reshape(-1,1,3).repeat(1,self.points_per_object,1)),-1)[0].reshape(self.num_envs,1)
            dist_finger4_to_closest_point_on_object = torch.min(F.pairwise_distance(self.sampled_pcl_abs,self.fingertip_pos[:, 3, :].reshape(-1,1,3).repeat(1,self.points_per_object,1)),-1)[0].reshape(self.num_envs,1)
            dist_finger5_to_closest_point_on_object = torch.min(F.pairwise_distance(self.sampled_pcl_abs,self.fingertip_pos[:, 4, :].reshape(-1,1,3).repeat(1,self.points_per_object,1)),-1)[0].reshape(self.num_envs,1)
            cur_fingertip_dis= torch.cat([dist_finger1_to_closest_point_on_object, dist_finger2_to_closest_point_on_object, 
            dist_finger3_to_closest_point_on_object, dist_finger4_to_closest_point_on_object, dist_finger5_to_closest_point_on_object],-1)
            delta_fingertip_dis = torch.mean(cur_fingertip_dis,-1) / 12.5
            fingertip_dis_reward = -delta_fingertip_dis
            self.rew_buf[:]+=fingertip_dis_reward    
            self.extras['fingertip_dis'] = fingertip_dis_reward
            self.previous_fingertip_pos = cur_fingertip_dis.clone()

        '''
        pose matching reward
        '''
        if 'pose_matching' in self.reward_type:
            satisfy_env_ids = (cur_dis_o2h <= final_dist*1.1).nonzero(as_tuple=False).squeeze(-1)
            if len(satisfy_env_ids)>0:
                dof_distance = self.get_dof_distance(self.cur_hand_states, self.target_hand_dof, weighted=True)/4.8744
                self.rew_buf[satisfy_env_ids] += (0.5-dof_distance[satisfy_env_ids])*0.1
        
        if 'goaldist' in self.reward_type:
            dof_distance = self.get_dof_distance(self.cur_hand_states, self.rl_goal_pose, weighted=False)/5.8852
            self.rew_buf[:] += -dof_distance/50
            self.extras['goaldist_penatly'] = dof_distance/50

        '''
        lift test
        '''
        # lift test
        lift_env_ids = (self.actions[:,-1]==1).nonzero(as_tuple=False).squeeze(-1)
        
        if len(lift_env_ids) > 0:
            target_obj_moving = F.pairwise_distance(next_obj_states[:,:3],self.init_object_pose[:,:3])
            cur_obj_axis = quat_axis(next_obj_states[:,3:7],0)
            init_obj_axis = quat_axis(self.init_object_pose[:,3:7],0)
            self.extras['obj_translation'] = torch.mean(target_obj_moving).unsqueeze(-1)
            self.extras['obj_cosine_similarity'] = torch.mean(cosine_similarity(cur_obj_axis,init_obj_axis)).unsqueeze(-1)
            
            self.extras['diso2h'] = cur_dis_o2h
            self.extras['target_dist'] = final_dist

            self.final_hand_dof = self.shadow_hand_dof_pos[:, self.actuated_dof_indices]

            self.extras['gt_dist'] = torch.mean(F.pairwise_distance(self.final_hand_dof, self.target_hand_dof)).unsqueeze(-1)
            delta_object_pose = self.lift_test(lift_env_ids, flist=None, close_dis=self.close_dis)

            # height reward
            if "height" in self.reward_type:
                self.rew_buf[lift_env_ids]+=delta_object_pose[lift_env_ids]*self.success_reward*0.5
            
            lifted_env_ids = (delta_object_pose > 0.05).nonzero(as_tuple=False).squeeze(-1)
            self.extras['lifted_object_ratio'] = to_torch((len(lifted_env_ids)/len(lift_env_ids)), dtype=torch.float, device=self.device).unsqueeze(-1)
            self.extras['lifted_object_height'] = delta_object_pose[lifted_env_ids]

            self.reset_buf[lift_env_ids] = 1

            self.extras['lift_nums'] =  self.lift_successes.clone()
            lift_success_env_ids = self.lift_successes.nonzero(as_tuple=False).squeeze(-1)

            self.rew_buf[lift_success_env_ids] = self.success_reward

            # reset buffer
            self.lift_successes[lift_env_ids] = 0
            self.successes[lift_success_env_ids] += 1

            success_rate = to_torch(len(lift_success_env_ids)/len(lift_env_ids), dtype=torch.float, device=self.device)
            self.extras['success_num'] = to_torch(len(lift_success_env_ids), dtype=torch.float, device=self.device).unsqueeze(-1)
            self.extras['success_rate'] = success_rate.unsqueeze(-1)

        return

    def get_dof_distance(self, cur_dof, target_dof, weighted=False):
        if weighted:
            dof_distance = F.pairwise_distance(cur_dof[:,self.knuckle_dof_indices],target_dof[:,self.knuckle_dof_indices])*0.75 + \
                                F.pairwise_distance(cur_dof[:,self.middle_dof_indices],target_dof[:,self.middle_dof_indices])*0.5 + \
                                F.pairwise_distance(cur_dof[:,self.distal_dof_indices],target_dof[:,self.distal_dof_indices])*0.25
        else:
            dof_distance = F.pairwise_distance(cur_dof,target_dof)
        return dof_distance

    #####################################################################
    ###=====================observation functions=====================###
    #####################################################################
    def compute_observations(self, reset=True):
        self.refresh_env_states()

        if self.obs_type == "gf":
            self.compute_gf_state(reset)
        else:
            print("Unknown observations type!")

    # gf observation function
    def compute_gf_state(self, reset=True, relative=False, gf_state=False):
        # get obj pcl 2 world
        sampled_pcl = self.transform_obj_pcl_2_world(gen_pcl_with_ground=self.gen_pcl_with_ground)
        self.sampled_pcl_abs = sampled_pcl.clone()
        if add_obs_noise:
            wrist_pos_estimation_nosie = torch.clamp(torch.randn_like(self.rigid_body_states[:, self.hand_mount_handle][:,:3])*np.sqrt(0.0004), -0.02, 0.02)
            writs_quat_estimation_noise = np.sqrt(2/57.3)
        # get world 2 hand quat
        if relative or self.gfrl_relative or self.wrist_gen:
            if self.has_base:
                h2w_pos = self.rigid_body_states[:, self.hand_mount_handle][:,:3].clone()
                h2w_quat = self.rigid_body_states[:, self.hand_mount_handle][:,3:7].clone()
            else:
                h2w_pos = self.hand_positions[self.hand_indices, :].clone()
                h2w_quat = self.hand_orientations[self.hand_indices, :].clone()
            if add_obs_noise:
                h2w_pos += wrist_pos_estimation_nosie
                h2w_quat = random_orientation_within_angle(h2w_quat.size(0), self.device, h2w_quat, writs_quat_estimation_noise)
            h_dof = self.shadow_hand_dof_pos[:,self.actuated_dof_indices].clone()
            w2h_quat, w2h_pos = transform_world2target(h2w_quat, h2w_pos)

            sampled_pcl_2_hand = self.transform_obj_pcl_2_hand(sampled_pcl, w2h_quat, w2h_pos)
            sampled_pcl_2_hand = sampled_pcl_2_hand.reshape(self.num_envs,self.points_per_object*3)
            sampled_pcl = sampled_pcl_2_hand
        else:
            h_dof = self.shadow_hand_dof_pos[:,self.actuated_dof_indices].clone()
            sampled_pcl = sampled_pcl.reshape(self.num_envs,self.points_per_object*3)
    
        # unscale means normalization
        self.obs_buf[:,:18] = unscale(h_dof,self.shadow_hand_dof_lower_limits[self.actuated_dof_indices], self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
        if self.has_base:
            self.obs_buf[:,18:21] = self.rigid_body_states[:, self.hand_mount_handle][:,:3].clone()
            self.obs_buf[:,21:25] = self.rigid_body_states[:, self.hand_mount_handle][:,3:7].clone()
        else:
            self.obs_buf[:,18:21] = self.hand_positions[self.hand_indices, :].clone()
            self.obs_buf[:,21:25] = self.hand_orientations[self.hand_indices, :].clone()
        if add_obs_noise:
            self.obs_buf[:,18:21] = h2w_pos.clone()
            self.obs_buf[:,21:25] = h2w_quat.clone()
        point_cloud_idx = 25+self.points_per_object*3
        self.obs_buf[:,25:point_cloud_idx] = sampled_pcl
        if self.method!='gf+rl' or gf_state:
            self.obs_buf[:,point_cloud_idx:point_cloud_idx+3] = self.object_pos
            self.obs_buf[:,point_cloud_idx+3:point_cloud_idx+7] = self.object_rot
            self.obs_buf[:,point_cloud_idx+7:point_cloud_idx+8] = torch.arange(start=0, end=self.num_envs, device=self.device, dtype=torch.long).unsqueeze(1)
            self.obs_buf[:,point_cloud_idx+8:point_cloud_idx+23] = self.fingertip_pos.reshape(self.num_envs, self.num_fingertips*3)
            self.obs_buf[:,point_cloud_idx+23:point_cloud_idx+24] = self.cur_obj_scales.reshape(self.num_envs, 1)

        if reset:
            self.init_object_pose = self.object_pose.clone()
            self.init_hand_pose = self.obs_buf[:,18:25].clone()
            self.previous_object_pose = self.object_pose.clone()
            self.previous_hand_pose = self.obs_buf[:,18:25].clone()
            cur_hand_pose = self.obs_buf[:,18:25].clone()
            if 'fingertip_dis' in self.reward_type:
                dist_finger1_to_closest_point_on_object = torch.min(F.pairwise_distance(self.sampled_pcl_abs,self.fingertip_pos[:, 0, :].reshape(-1,1,3).repeat(1,self.points_per_object,1)),-1)[0].reshape(self.num_envs,1)
                dist_finger2_to_closest_point_on_object = torch.min(F.pairwise_distance(self.sampled_pcl_abs,self.fingertip_pos[:, 1, :].reshape(-1,1,3).repeat(1,self.points_per_object,1)),-1)[0].reshape(self.num_envs,1)
                dist_finger3_to_closest_point_on_object = torch.min(F.pairwise_distance(self.sampled_pcl_abs,self.fingertip_pos[:, 2, :].reshape(-1,1,3).repeat(1,self.points_per_object,1)),-1)[0].reshape(self.num_envs,1)
                dist_finger4_to_closest_point_on_object = torch.min(F.pairwise_distance(self.sampled_pcl_abs,self.fingertip_pos[:, 3, :].reshape(-1,1,3).repeat(1,self.points_per_object,1)),-1)[0].reshape(self.num_envs,1)
                dist_finger5_to_closest_point_on_object = torch.min(F.pairwise_distance(self.sampled_pcl_abs,self.fingertip_pos[:, 4, :].reshape(-1,1,3).repeat(1,self.points_per_object,1)),-1)[0].reshape(self.num_envs,1)

                self.previous_fingertip_pos = torch.cat([dist_finger1_to_closest_point_on_object, dist_finger2_to_closest_point_on_object, 
                dist_finger3_to_closest_point_on_object, dist_finger4_to_closest_point_on_object, dist_finger5_to_closest_point_on_object],-1)
        else:
            self.cur_hand_pose = self.obs_buf[:,18:25].clone()
            cur_hand_pose = self.obs_buf[:,18:25].clone()

        if self.method=='gf+rl' and not gf_state:
            # for stack frame
            if self.use_abs_stack_frame:
                if reset:
                    for _ in range(self.stack_frame_numbers):
                        self.frames.append(self.obs_buf[:,18:25].clone())
                else:
                    self.frames.append(self.obs_buf[:,18:25].clone())
            else:
                if reset:
                    o2h_pos, o2h_quat = multiply_transform(w2h_pos, w2h_quat, self.object_pos, self.object_rot)
                    for _ in range(self.stack_frame_numbers):
                        self.frames.append(torch.cat([o2h_pos,o2h_quat],-1))
                else:
                    o2h_pos, o2h_quat = multiply_transform(w2h_pos, w2h_quat, self.object_pos, self.object_rot)
                    self.frames.append(torch.cat([o2h_pos,o2h_quat],-1))

            if self.stack_frame_numbers > 1:
                hand_dof = self.obs_buf[:,:18]
                hand_wrist_traj = torch.cat(list(self.frames),1)
                fingertip_idx = 18 + self.stack_frame_numbers*7 + self.points_per_object*3
                self.obs_buf[:,:fingertip_idx] = torch.cat([hand_dof, hand_wrist_traj, sampled_pcl],-1)
            else:
                fingertip_idx = 18 + self.stack_frame_numbers*7 + self.points_per_object*3
            
            # fingertip obs
            if "fingertipjoint" in self.sub_obs_type:
                self.obs_buf[:,fingertip_idx:fingertip_idx+len(self.unactuated_dof_indices)] = unscale(self.shadow_hand_dof_pos[:,self.unactuated_dof_indices],self.shadow_hand_dof_lower_limits[self.unactuated_dof_indices], self.shadow_hand_dof_upper_limits[self.unactuated_dof_indices])
                fingertip_idx = fingertip_idx+len(self.unactuated_dof_indices)

            if "absfingertip" in self.sub_obs_type:
                self.obs_buf[:,fingertip_idx:fingertip_idx+self.num_fingertips*3] = (self.fingertip_pos - h2w_pos.reshape(self.num_envs,1,3).repeat(1,self.num_fingertips,1)).reshape(self.num_envs, self.num_fingertips*3)
                obj_pose_idx = fingertip_idx+self.num_fingertips*3
            elif "disfingertip" in self.sub_obs_type:
                dist_finger1_to_closest_point_on_object = torch.min(F.pairwise_distance(self.sampled_pcl_abs,self.fingertip_pos[:, 0, :].reshape(-1,1,3).repeat(1,self.points_per_object,1)),-1)[0].reshape(self.num_envs,1)
                dist_finger2_to_closest_point_on_object = torch.min(F.pairwise_distance(self.sampled_pcl_abs,self.fingertip_pos[:, 1, :].reshape(-1,1,3).repeat(1,self.points_per_object,1)),-1)[0].reshape(self.num_envs,1)
                dist_finger3_to_closest_point_on_object = torch.min(F.pairwise_distance(self.sampled_pcl_abs,self.fingertip_pos[:, 2, :].reshape(-1,1,3).repeat(1,self.points_per_object,1)),-1)[0].reshape(self.num_envs,1)
                dist_finger4_to_closest_point_on_object = torch.min(F.pairwise_distance(self.sampled_pcl_abs,self.fingertip_pos[:, 3, :].reshape(-1,1,3).repeat(1,self.points_per_object,1)),-1)[0].reshape(self.num_envs,1)
                dist_finger5_to_closest_point_on_object = torch.min(F.pairwise_distance(self.sampled_pcl_abs,self.fingertip_pos[:, 4, :].reshape(-1,1,3).repeat(1,self.points_per_object,1)),-1)[0].reshape(self.num_envs,1)
                self.obs_buf[:,fingertip_idx:fingertip_idx+self.num_fingertips] = torch.cat([dist_finger1_to_closest_point_on_object, dist_finger2_to_closest_point_on_object, 
                dist_finger3_to_closest_point_on_object, dist_finger4_to_closest_point_on_object, dist_finger5_to_closest_point_on_object],-1)
                # print(self.obs_buf[:,fingertip_idx:fingertip_idx+self.num_fingertips])
                obj_pose_idx = fingertip_idx+self.num_fingertips
            else:
                obj_pose_idx = fingertip_idx

            if "objpose" in self.sub_obs_type:
                if "absobjpose" in self.sub_obs_type:
                    self.obs_buf[:,obj_pose_idx:obj_pose_idx+3] = self.object_pos
                    self.obs_buf[:,obj_pose_idx+3:obj_pose_idx+7] = self.object_rot
                elif "deltaobjpose" in self.sub_obs_type:
                    self.obs_buf[:,obj_pose_idx:obj_pose_idx+3] = self.object_pos - self.previous_object_pose[:,:3]
                    self.obs_buf[:,obj_pose_idx+3:obj_pose_idx+7] = self.object_rot - self.previous_object_pose[:,3:7]
                    self.previous_object_pose = self.object_pose.clone()
                elif "targetobjpose" in self.sub_obs_type:
                    self.obs_buf[:,obj_pose_idx:obj_pose_idx+3] = self.object_pos - self.init_object_pose[:,:3]
                    self.obs_buf[:,obj_pose_idx+3:obj_pose_idx+7] = self.object_rot - self.init_object_pose[:,3:7]
                diso2o_idx = obj_pose_idx + 7
            else:
                diso2o_idx = obj_pose_idx
            
            if "diso2o" in self.sub_obs_type:
                if "deltadiso2o" in self.sub_obs_type:
                    self.obs_buf[:,diso2o_idx:diso2o_idx+1] = F.pairwise_distance(self.object_pos, self.previous_object_pose[:,:3]).reshape(self.num_envs,1)
                elif "targetdiso2o" in self.sub_obs_type:
                    self.obs_buf[:,diso2o_idx:diso2o_idx+1] = F.pairwise_distance(torch.mean(self.sampled_pcl_abs,1), cur_hand_pose[:,:3]).reshape(self.num_envs,1)
                goal_idx = diso2o_idx + 1
            else:
                goal_idx = diso2o_idx
            
            if 'goal' in self.sub_obs_type:
                self.goal_idx = goal_idx
                self.obs_buf[:,goal_idx:goal_idx+18] = self.dof_norm(self.rl_goal_pose)
    
    def transform_obj_pcl_2_world(self, points_num=None, gen_pcl_with_ground=False):
        o2w_pos = self.object_pos[:,:3].clone()
        o2w_pos = o2w_pos.resize(o2w_pos.size(0),1,3)
        o2w_quat = self.object_rot.clone()
        o2w_quat = o2w_quat.resize(o2w_quat.size(0),1,4)
        if self.method == 'gf+rl':
            # for rl sample same point cloud in one episode
            sampled_point_idxs = self.sampled_point_idxs.clone()
        else:
            if points_num == None:
                sampled_point_idxs = farthest_point_sample(self.obj_pcl_buf, self.points_per_object, self.device)
            else:
                sampled_point_idxs = farthest_point_sample(self.obj_pcl_buf, points_num, self.device)
        sampled_pcl = index_points(self.obj_pcl_buf, sampled_point_idxs, self.device)
        if points_num == None:
            append_pos = torch.zeros([sampled_pcl.size(0),self.points_per_object,1]).to(self.device)
        else:
            append_pos = torch.zeros([sampled_pcl.size(0),points_num,1]).to(self.device)
        sampled_pcl = torch.cat([sampled_pcl,append_pos],2)

        o2w_quat = o2w_quat.expand_as(sampled_pcl)
        sampled_pcl = transform_points(o2w_quat, sampled_pcl)
        o2w_pos = o2w_pos.expand_as(sampled_pcl)
        sampled_pcl = sampled_pcl + o2w_pos

        if gen_pcl_with_ground:
            gt_points_num = int(self.points_per_object/4)
            random_point_idx = torch.randint(0, self.points_per_object, (self.num_envs ,gt_points_num), dtype=torch.long, device=self.device)
            gt_pcl = index_points(sampled_pcl, random_point_idx, self.device)
            
            gt_pcl[:,:,2] = torch.min(sampled_pcl,1)[0][:,2:3].repeat(1,gt_points_num)

            radius = (torch.max(torch.max(sampled_pcl,1)[0],-1)[0] - torch.min(torch.min(sampled_pcl,1)[0],-1)[0])/4
            min_bound = torch.min(sampled_pcl,1)[0] - 0.01
            max_bound = torch.max(sampled_pcl,1)[0] + 0.01
            center_point = torch.mean(sampled_pcl,1)
            gt_pcl_x = torch_random_sample(min_bound[:,0].reshape(self.num_envs,1).repeat(1,gt_points_num),max_bound[:,0].reshape(self.num_envs,1).repeat(1,gt_points_num),(self.num_envs,int(self.points_per_object/4)),device=self.device)
            gt_pcl_y = torch_random_sample(min_bound[:,1].reshape(self.num_envs,1).repeat(1,gt_points_num),max_bound[:,1].reshape(self.num_envs,1).repeat(1,gt_points_num),(self.num_envs,int(self.points_per_object/4)),device=self.device)

            gt_pcl[:,:,0] = gt_pcl_x
            gt_pcl[:,:,1] = gt_pcl_y

            B = sampled_pcl.size()[0]
            view_shape = list(random_point_idx.size())
            view_shape[1:] = [1] * (len(view_shape) - 1)
            repeat_shape = list(random_point_idx.size())
            repeat_shape[0] = 1
            batch_indices = torch.arange(B, dtype=torch.long).to(self.device).view(view_shape).repeat(repeat_shape)
            sampled_pcl[batch_indices, random_point_idx,:] = gt_pcl
    
        return sampled_pcl
    
    def transform_obj_pcl_2_hand(self, sampled_pcl, w2h_quat, w2h_pos):
        B = sampled_pcl.size()[0]
        padding = torch.zeros([B, self.points_per_object, 1]).to(sampled_pcl.device)
        pt_input = torch.cat([sampled_pcl,padding],2)
        w2h_quat = w2h_quat.resize(w2h_quat.size(0),1,4)
        w2h_quat = w2h_quat.expand_as(pt_input)
        pt_new = transform_points(w2h_quat, pt_input)[:,:,:3]
        w2h_pos = w2h_pos.resize(w2h_pos.size(0),1,3)
        w2h_pos.expand_as(pt_new)
        pt_new += w2h_pos
        return pt_new

    #####################################################################
    ###========================reset functions========================###
    #####################################################################
    def reset_target_pose(self, env_ids, apply_reset=False):
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), 4), device=self.device)

        new_rot = randomize_rotation(rand_floats[:, 0], rand_floats[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids])

        if self.has_goal:
            self.goal_states[env_ids, 0:3] = self.goal_init_state[env_ids, 0:3]
            self.goal_states[env_ids, 3:7] = new_rot
            self.root_state_tensor[self.goal_object_indices[env_ids], 0:3] = self.goal_states[env_ids, 0:3] + self.goal_displacement_tensor
            self.root_state_tensor[self.goal_object_indices[env_ids], 3:7] = self.goal_states[env_ids, 3:7]
            self.root_state_tensor[self.goal_object_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.goal_object_indices[env_ids], 7:13])

            if apply_reset:
                goal_object_indices = self.goal_object_indices[env_ids].to(torch.int32)
                self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                            gymtorch.unwrap_tensor(self.root_state_tensor),
                                                            gymtorch.unwrap_tensor(goal_object_indices), len(env_ids))
        self.reset_goal_buf[env_ids] = 0

    def reset_object_pose(self, env_ids, object_pose=None, hand_also=True, move_left=False):
        if self.has_goal:
            goal_env_ids = env_ids
        hand_indices = self.hand_indices[env_ids].to(torch.int32)

        if hand_also:
            # reset hand z avoid collision
            if self.has_base:
                self.cur_targets[env_ids,:] *= 0
                self.cur_targets[env_ids,self.hand_base_dof_indices[2]] = 1.0
                self.shadow_hand_dof_pos[env_ids,:] *= 0
                self.shadow_hand_dof_pos[env_ids,self.hand_base_dof_indices[2]] = 1.0
                self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                            gymtorch.unwrap_tensor(self.cur_targets),
                                                            gymtorch.unwrap_tensor(hand_indices), len(env_ids))
                self.gym.set_dof_state_tensor_indexed(self.sim,
                                                gymtorch.unwrap_tensor(self.dof_state),
                                                gymtorch.unwrap_tensor(hand_indices), len(env_ids))
            else:
                self.root_state_tensor[self.hand_indices[env_ids], 2] = 1.0
                self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                            gymtorch.unwrap_tensor(self.root_state_tensor),
                                                            gymtorch.unwrap_tensor(hand_indices), len(hand_indices))

        self.refresh_env_states(simulate_gym=True)                                             
        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)

        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), 4), device=self.device)
        new_rot = randomize_rotation_xyz(rand_floats[:, 0], rand_floats[:, 1], rand_floats[:, 2], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids], self.z_unit_tensor[env_ids])

        if object_pose is not None:
            if move_left:
                object_pose_left = object_pose.clone()
                object_pose_left[:,0] = object_pose_left[:,0]+0.5
                self.root_state_tensor[self.object_indices[env_ids],:7] = object_pose_left[env_ids]
            else:
                self.root_state_tensor[self.object_indices[env_ids],:7] = object_pose[env_ids]
        else:
            self.root_state_tensor[self.object_indices[env_ids],:2] *= 0 
            self.root_state_tensor[self.object_indices[env_ids],2] = self.objects_minz[env_ids]+0.1
            self.root_state_tensor[self.object_indices[env_ids],3:7] = new_rot

        if self.has_goal:
            object_indices = torch.unique(torch.cat([self.object_indices[env_ids],
                                                    self.goal_object_indices[env_ids],
                                                    self.goal_object_indices[goal_env_ids]]).to(torch.int32))
        else:
            object_indices = torch.unique(self.object_indices[env_ids].to(torch.int32))
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                    gymtorch.unwrap_tensor(self.root_state_tensor),
                                                    gymtorch.unwrap_tensor(object_indices), len(object_indices))
        
        self.reset_obj_vel(env_ids=env_ids)
        self.refresh_env_states(simulate_gym=True)

    def reset_hand_pose(self, env_ids, hand_pose=None, set_pcl=True, joint_only=False):
        if hand_pose is not None:
            hand_pos, hand_rot, hand_dof = hand_pose
                
        for (i, env_id) in enumerate(env_ids):
            self.hand_new_dof_state[env_id, self.actuated_dof_indices] = hand_dof[i]

        if self.fake_tendon:
            # set underactuated dof
            self.set_underactuated_dof(env_ids=env_ids, target='hand_new_dof_state')
        
        if self.has_base and (not joint_only):
            hand_euler = self.get_hand_euler_from_quat(hand_rot)
            self.hand_new_dof_state[env_ids.reshape(-1,1).repeat(1,len(self.hand_base_dof_indices)),self.hand_base_dof_indices.reshape(1,-1).repeat(len(env_ids),1)] = torch.cat([hand_pos, hand_euler],-1)

        # set dof state
        self.shadow_hand_dof_pos[env_ids, :] = self.hand_new_dof_state[env_ids]
        self.shadow_hand_dof_vel[env_ids, :] = self.shadow_hand_dof_default_vel
        self.prev_targets[env_ids, :] = self.hand_new_dof_state[env_ids]
        self.cur_targets[env_ids, :] = self.hand_new_dof_state[env_ids]

        hand_indices = self.hand_indices[env_ids].to(torch.int32)

        if (not self.has_base) and (not joint_only):
            # reset hand root state
            self.root_state_tensor[self.hand_indices[env_ids], :3] = hand_pos
            self.root_state_tensor[self.hand_indices[env_ids], 3:7] = hand_rot

            self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.root_state_tensor),
                                                        gymtorch.unwrap_tensor(hand_indices), len(hand_indices))
        
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.cur_targets),
                                                        gymtorch.unwrap_tensor(hand_indices), len(env_ids))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(hand_indices), len(env_ids))
        
        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)
        
        self.refresh_env_states(simulate_gym=True)
    
    # main reset
    def reset(self, env_ids=None, mode=None, states=None, is_random=True, pose_id=None, env_init=False):
        if self.method=='gf+rl' or self.method=='filter':
            self._reset_simulator(pose_id=pose_id, env_ids=env_ids)
            # print(self.test_times)

        if mode != None:
            self.mode = mode
            
        if env_ids is None:
            # reset all environments
            env_ids = torch.arange(start=0, end=self.num_envs, device=self.device, dtype=torch.long)
        else:
            if 'int' in str(type(env_ids)):
                env_ids = torch.tensor([env_ids], device=self.device, dtype=torch.long)
            else:
                env_ids = env_ids.nonzero(as_tuple=False).squeeze(-1)

        if self.method == 'gf' and self.mode!='demo_gen' and states is not None:
            self.reset_gf_env(states=states, is_random=is_random)
        else:
            self.reset_idx(env_ids, env_ids, is_random, pose_id, env_init=env_init)
            
        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)

        # compute obs after reset
        self.compute_observations()

        if debug:
            self.previous_dof = self.shadow_hand_dof_pos[:,self.actuated_dof_indices].clone()
        # TODO gf+rl not clip like gf
        if self.obs_type == "gf" and self.method!='gf+rl':
            point_cloud_idx = self.points_per_object*3 + 25
            pcl_obs = torch.clamp(self.obs_buf[:,:point_cloud_idx+7], -self.clip_obs, self.clip_obs).to(self.rl_device)
            envid_obs = self.obs_buf[:,point_cloud_idx+7:point_cloud_idx+8].to(self.rl_device)
            fingertip_obs = torch.clamp(self.obs_buf[:,point_cloud_idx+8:point_cloud_idx+23], -self.clip_obs, self.clip_obs).to(self.rl_device)
            objscale_obs = self.obs_buf[:,point_cloud_idx+23:point_cloud_idx+24].to(self.rl_device)
            self.obs_dict["obs"] = torch.cat([pcl_obs, envid_obs, fingertip_obs, objscale_obs],-1)
        else:        
            self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

        # asymmetric actor-critic
        if self.num_states > 0:
            self.obs_dict["states"] = self.get_state()

        return self.obs_dict

    def reset_idx(self, env_ids, goal_env_ids, is_random=True, pose_id=None, env_init=False):
        if self.mode == 'eval':
            if not env_init:
                self.test_times[env_ids] += 1
            self.eval_finish_env_ids = torch.where(self.test_times > self.eval_times)[0]

        # randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        # generate random values
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), self.num_shadow_hand_dofs * 2 + 5), device=self.device)
        
        # randomize start object poses
        self.reset_target_pose(env_ids)

        # reset rigid body forces
        self.rb_forces[env_ids, :, :] = 0.0

        # recover obj pcl
        if self.dataset_type=='dexgraspnet' and allow_obj_scale:
            self.obj_pcl_buf[env_ids] = self.obj_pcl_buf[env_ids]/self.cur_obj_scales[env_ids].reshape(self.num_envs, 1, 1).repeat(1,self.obj_pcl_buf.size(1),self.obj_pcl_buf.size(2))

        self.object_new_state = self.object_init_state[env_ids].clone()
        self.hand_new_dof_state = self.prev_targets.clone()

        if self.custom_gen_data and not env_init and not self.wrist_gen:
            self.con_reset_idx(obj_ids=env_ids)
        else:
            self.uncon_reset_idx(env_ids=env_ids, goal_env_ids=goal_env_ids, pose_id=pose_id, is_random=is_random)
        # important reset object velocity and angular velocity to zero
        self.reset_obj_vel(env_ids=env_ids)
        # reset
        self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.rb_forces), None, gymapi.LOCAL_SPACE)

        self.progress_buf[env_ids] = 1
        self.reset_buf[env_ids] = 0
        # self.successes[env_ids] = 0

    def deploy_to_environments(self, env_ids, goal_env_ids=None, step_simulation_step=None):
        self.root_state_tensor[self.hand_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.hand_indices[env_ids], 7:13])
        hand_indices = self.hand_indices[env_ids].to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                    gymtorch.unwrap_tensor(self.root_state_tensor),
                                                    gymtorch.unwrap_tensor(hand_indices), len(hand_indices))

        self.reset_obj_vel(env_ids=env_ids)
        
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.cur_targets),
                                                        gymtorch.unwrap_tensor(hand_indices), len(env_ids))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                            gymtorch.unwrap_tensor(self.dof_state),
                                            gymtorch.unwrap_tensor(hand_indices), len(env_ids))
        
        if step_simulation_step!=None:
            for i in range(step_simulation_step):
                self.gym.simulate(self.sim)
        else:
            self.gym.simulate(self.sim)

        # set_trace()
        if self.has_goal and goal_env_ids!=None:
            object_indices = torch.unique(torch.cat([self.object_indices[env_ids],
                                                    self.goal_object_indices[env_ids],
                                                    self.goal_object_indices[goal_env_ids]]).to(torch.int32))
        else:
            object_indices = torch.unique(self.object_indices[env_ids].to(torch.int32))
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                    gymtorch.unwrap_tensor(self.root_state_tensor),
                                                    gymtorch.unwrap_tensor(object_indices), len(object_indices))

    def uncon_reset_idx(self, env_ids=None, goal_env_ids=None, pose_id=None, is_random=True):
        if self.dataset_type=='dexgraspnet':
            if allow_obj_scale:
                for env_id in env_ids:
                    # take object type
                    tmp_object_type = self.object_types[env_id]
                    tmp_object_id = self.obj_type_id[tmp_object_type]
                    
                    shape_path = os.path.join(self.mesh_dir, tmp_object_type, 'pcd/600.pcd')
                    if pose_id is None:
                        # random choose grasp label
                        tmp_grasp_id = random.randint(0,self.grasp_poses_num[tmp_object_id]-1)
                        if self.custom_gen_data and self.constrained:
                            obj_scale = 0.08
                        else:
                            obj_scale = self.obj_scales[tmp_object_id][tmp_grasp_id].clone()
                    else:
                        tmp_object_grasp_pose_num = self.grasp_poses_num[tmp_object_id]
                        assert pose_id[env_id] <= tmp_object_grasp_pose_num , f"grasp pose id invalid, total grasp poses number is: {tmp_object_grasp_pose_num}"
                        tmp_grasp_id = pose_id[env_id] - 1
                        if self.custom_gen_data and self.constrained:
                            obj_scale = 0.08
                        else:
                            obj_scale = self.obj_scales[tmp_object_id][tmp_grasp_id].clone()

                    if allow_obj_scale:
                        # rescale object
                        self.cur_obj_scales[env_id] = obj_scale
                        self.obj_pcl_buf[env_id] = self.obj_pcl_buf[env_id] * self.cur_obj_scales[env_id]
                        self.gym.set_actor_scale(self.envs[env_id],1,obj_scale)
                self.refresh_env_states()

            for env_id in env_ids:
                # take object type
                tmp_object_type = self.object_types[env_id]
                tmp_object_id = self.obj_type_id[tmp_object_type]
                
                if self.grasp_poses_num[tmp_object_id]-1 < 0:
                    continue

                if pose_id is None:
                    # random choose grasp label
                    if self.grasp_poses_num[tmp_object_id]-1 == 0 or self.random_sample_grasp_label:
                        tmp_grasp_id = 0
                    else:
                        tmp_grasp_id = random.randint(0,self.grasp_poses_num[tmp_object_id]-1)
                    
                    if self.custom_gen_data:
                        grasp_pose = self.grasp_poses[tmp_object_id][tmp_grasp_id][:25].clone()
                        grasp_pose[7:25] = self.dof_norm(grasp_pose[7:25],inv=True)
                    else:
                        grasp_pose = self.grasp_poses[tmp_object_id][tmp_grasp_id].copy()
                else:
                    tmp_object_grasp_pose_num = self.grasp_poses_num[tmp_object_id]
                    assert pose_id[env_id] <= tmp_object_grasp_pose_num , f"grasp pose id invalid, total grasp poses number is: {tmp_object_grasp_pose_num}"
                    tmp_grasp_id = pose_id[env_id] - 1
                    
                    if self.custom_gen_data:
                        grasp_pose = self.grasp_poses[tmp_object_id][tmp_grasp_id][:25] .clone()
                        grasp_pose[7:25] = self.dof_norm(grasp_pose[7:25],inv=True)
                    else:
                        grasp_pose = self.grasp_poses[tmp_object_id][tmp_grasp_id]

                grasp_pose = to_torch(grasp_pose, device=self.device)

                ## reset object pose
                self.root_state_tensor[self.object_indices[env_id]][:3] = 0
                self.root_state_tensor[self.object_indices[env_id]][3:7] = to_torch([0,0,0,1],device=self.device)

                ## reset pcl hand
                new_hand_pos = grasp_pose[:3] 
                new_hand_pos[2] += 0.2
                new_hand_rot = grasp_pose[3:7] 
                self.hand_new_dof_state[env_id][self.actuated_dof_indices] = grasp_pose[7:]

                if self.fake_tendon:
                    # set underactuated dof
                    self.set_underactuated_dof(env_ids=env_id, target='hand_new_dof_state')
                
                if self.has_base:
                    new_hand_euler = self.get_hand_euler_from_quat(new_hand_rot)
                    self.hand_new_dof_state[env_id][self.hand_base_dof_indices] = torch.cat([new_hand_pos.unsqueeze(0), new_hand_euler],-1)
                else:
                    # reset hand root state
                    self.root_state_tensor[self.hand_indices[env_id], :3] = new_hand_pos
                    self.root_state_tensor[self.hand_indices[env_id], 3:7] = new_hand_rot

                # set dof state
                self.shadow_hand_dof_pos[env_id, :] = self.hand_new_dof_state[env_id]
                self.shadow_hand_dof_vel[env_id, :] = self.shadow_hand_dof_default_vel
                self.prev_targets[env_id, :] = self.hand_new_dof_state[env_id]
                self.cur_targets[env_id, :] = self.hand_new_dof_state[env_id]
        
        self.deploy_to_environments(env_ids=env_ids, goal_env_ids=goal_env_ids)
        self.sampled_point_idxs = farthest_point_sample(self.obj_pcl_buf, self.points_per_object, self.device)

    def con_reset_idx(self, obj_ids=None, traj_num_per_object=1, mode='state'):
        valid_states = to_torch([],device=self.device)
        # all_obj_ids = torch.arange(start=0, end=self.num_envs, device=self.device, dtype=torch.long)
        valid_grasp = torch.zeros(self.num_envs, device=self.device)

        if obj_ids is not None:
            obj_ids = torch.tensor(obj_ids,device=self.device).long()
        else:
            obj_ids = torch.arange(start=0, end=self.num_envs, device=self.device, dtype=torch.long)
        
        all_obj_ids = obj_ids.clone()

        if len(all_obj_ids) < self.num_envs:
            print('---')
        self.traj_num = torch.zeros(self.num_envs, device=self.device)
        # for obj_id in obj_ids:
        # st_time = time.time()

        stable_object_pose = torch.zeros((self.num_envs,7), device=self.device)
        grasp_pose = torch.zeros((self.num_envs,7), device=self.device)
        hand_dofs = torch.zeros((self.num_envs,18), device=self.device)
        if self.dataset_type=='dexgraspnet':
            obj_scales = torch.zeros((self.num_envs,1), device=self.device)

        while not (self.traj_num[all_obj_ids]==traj_num_per_object).any():
            if self.table_setting:
                if not self.custom_gen_data:
                    stable_object_pose[obj_ids] = self.gen_stable_object_pose(obj_ids)
            else:
                stable_object_pose = torch.zeros((self.num_envs,7), device=self.device)
                x_unit_tensor = to_torch([1, 0, 0], dtype=torch.float, device=self.device).repeat((len(all_obj_ids), 1))
                y_unit_tensor = to_torch([0, 1, 0], dtype=torch.float, device=self.device).repeat((len(all_obj_ids), 1))
                z_unit_tensor = to_torch([0, 0, 1], dtype=torch.float, device=self.device).repeat((len(all_obj_ids), 1))
                rand_floats = torch_rand_float(-1, 1, (len(all_obj_ids), 4), device=self.device)
                rand_obj_rot = randomize_rotation_xyz(rand_floats[:, 0], rand_floats[:, 1], rand_floats[:, 2], x_unit_tensor, y_unit_tensor, z_unit_tensor)
                stable_object_pose[obj_ids,3:7] = rand_obj_rot

            candidate_grasps_num = self.meta_data['data']['obj_grasp_pose_num']
            if self.reset_sim_every_time:
                candidate_grasps_num = np.array(candidate_grasps_num)[self.object_ids_per_env]

            cur_grasp_pose_idx = np.zeros(self.num_envs, dtype=int)

            if (not self.table_setting) or self.custom_gen_data:
                if self.mode=='eval':
                    cur_grasp_pose_idx = self.test_times.cpu().numpy() - 1
                else:
                    cur_grasp_pose_idx = np.floor((candidate_grasps_num-cur_grasp_pose_idx)*np.random.rand(self.num_envs) + cur_grasp_pose_idx)

            if self.diff_obj_scale:
                cur_grasp_pose_idx = self.cur_grasp_pose_idx

            if self.custom_gen_data:
                if self.reset_sim_every_time:
                    point_cloud_idx = 25 + self.points_per_object * 3
                    stable_object_pose[obj_ids,:] = self.grasp_poses[self.object_ids_per_env, cur_grasp_pose_idx,point_cloud_idx:point_cloud_idx+7].clone()
                    # random_move = torch_rand_float(-0.5, 0.5, (len(all_obj_ids), 2), device=self.device)
                    stable_object_pose[obj_ids,:2] += self.random_move
                    grasp_pose[obj_ids,:] = self.grasp_poses[self.object_ids_per_env,cur_grasp_pose_idx,18:25].clone()
                    grasp_pose[obj_ids,:2] += self.random_move
                    hand_dofs[obj_ids,:] = scale(self.grasp_poses[self.object_ids_per_env,cur_grasp_pose_idx,:18].clone(), self.shadow_hand_dof_lower_limits[self.actuated_dof_indices], self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
                    if self.dataset_type=='dexgraspnet':
                        obj_scales[obj_ids] = self.grasp_poses[self.object_ids_per_env,cur_grasp_pose_idx,point_cloud_idx+7+15+1:point_cloud_idx+7+15+2].clone()
                    self.traj_num[all_obj_ids.cpu()] += 1
                    # set_trace()
                else:
                    point_cloud_idx = 25 + self.points_per_object * 3
                    stable_object_pose[obj_ids,:] = self.grasp_poses[obj_ids,cur_grasp_pose_idx,point_cloud_idx:point_cloud_idx+7].clone()
                    # random_move = torch_rand_float(-0.5, 0.5, (len(all_obj_ids), 2), device=self.device)
                    stable_object_pose[obj_ids,:2] += self.random_move
                    grasp_pose[obj_ids,:] = self.grasp_poses[obj_ids,cur_grasp_pose_idx,18:25].clone()
                    grasp_pose[obj_ids,:2] += self.random_move
                    hand_dofs[obj_ids,:] = scale(self.grasp_poses[obj_ids,cur_grasp_pose_idx,:18].clone(), self.shadow_hand_dof_lower_limits[self.actuated_dof_indices], self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
                    if self.dataset_type=='dexgraspnet':
                        obj_scales[obj_ids] = self.grasp_poses[obj_ids,cur_grasp_pose_idx,point_cloud_idx+7+15+1:point_cloud_idx+7+15+2].clone()
                    self.traj_num[all_obj_ids.cpu()] += 1

        if self.dataset_type == 'dexgraspnet' and allow_obj_scale:
            self.set_object_scale(env_ids=obj_ids, obj_scales=obj_scales[obj_ids])
        # sample same pointcloud for rl in each episode
        self.sampled_point_idxs[all_obj_ids] = farthest_point_sample(self.obj_pcl_buf, self.points_per_object, self.device)[all_obj_ids]

        # reset for get grasp pose fingertip
        self.reset_object_pose(all_obj_ids,stable_object_pose, move_left=True)
        self.reset_hand_pose(all_obj_ids.reshape(-1),(grasp_pose[all_obj_ids,:3], grasp_pose[all_obj_ids,3:7], hand_dofs))
        self.con_final_hand_pose = grasp_pose[all_obj_ids,:7].clone()
        self.con_final_object_pose = stable_object_pose.clone()
        self.target_hand_dof = self.shadow_hand_dof_pos[:,self.actuated_dof_indices].clone()
        # set_trace()
        print(torch.mean(torch.std(self.target_hand_dof,0)))
        if self.constrained:
            # reset trajs -100 for teminate
            self.trajs[all_obj_ids,:,:] = -100
            self.trajs_len[all_obj_ids] *= 0 

        if self.force_render:
            self.render()
        self.reset_obj_vel(env_ids=all_obj_ids)

        self.step_simulation(6)

        # gen traj 
        gen_trajs = self.traj_gen(states=False, hand_pos_only=True, env_ids=all_obj_ids, hand_rotation=self.hand_rotation, start_joint_noise=self.start_joint_noise)

        if self.constrained:
            # set new traj to trajs
            self.trajs_len[all_obj_ids] = gen_trajs.size(1)
            self.trajs[all_obj_ids,:gen_trajs.size(1),:] = gen_trajs

        # reset object and hand init pose
        self.reset_object_pose(all_obj_ids,stable_object_pose)

        # TODO not for all envs
        if self.sample_goal:
            # set_trace()
            self.rl_goal_pose = torch.zeros_like(self.target_hand_dof.clone())
            env_id_idx = 25+self.points_per_object*3+7
            for (e_id, cur_obj_type) in enumerate(self.object_type_per_env):
                obj_id_in_ori = self.filtered_data_oti[cur_obj_type]
                grasp_poses_id = (self.filtered_data[:,env_id_idx]==obj_id_in_ori).nonzero(as_tuple=False).squeeze().reshape(-1)
                grasp_poses_id = grasp_poses_id[np.random.choice(len(grasp_poses_id))]
                self.rl_goal_pose[e_id] = self.dof_norm(self.filtered_data[grasp_poses_id, :18].clone(),inv=True)

        self.reset_hand_pose(all_obj_ids.reshape(-1),(gen_trajs[all_obj_ids,0,:3], gen_trajs[all_obj_ids,0,3:7], gen_trajs[all_obj_ids,0,7:]))
        self.reset_object_pose(all_obj_ids,stable_object_pose, hand_also=False)

        # self.valid_env_ids = all_obj_ids[np.where(self.traj_num==traj_num_per_object)]

    def reset_obj_vel(self, env_ids):
        # important reset object velocity and angular velocity to zero
        object_indices = torch.unique(torch.cat([self.object_indices[env_ids]]).to(torch.int32))
        self.root_state_tensor[self.object_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.object_indices[env_ids], 7:13])
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.root_state_tensor),
                                                        gymtorch.unwrap_tensor(object_indices), len(object_indices))

    #####################################################################
    ###========================utils functions========================###
    #####################################################################
    '''
    for run simulation
    '''
    def render(self, mode="rgb_array", env_ids=None, rgb=False, img_size=256, vis_env_num=0):
        """Draw the frame to the viewer, and check for keyboard events."""
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync

            # fetch results
            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)

            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)

                # Wait for dt to elapse in real time.
                # This synchronizes the physics simulation with the rendering rate.
                self.gym.sync_frame_time(self.sim)

            else:
                self.gym.poll_viewer_events(self.viewer)

            if self.virtual_display and mode == "rgb_array":
                img = self.virtual_display.grab()
                # return np.array(img)
        
        if self.enable_camera_sensors and rgb:
            return self.get_images(env_ids=env_ids, img_size=img_size, vis_env_num=vis_env_num)
        
    def sample_actions(self):
        env_ids = torch.arange(start=0, end=self.num_envs, device=self.device, dtype=torch.long)
        actions = torch_rand_float(-1.0, 1.0, (len(env_ids), self.num_actions), device=self.device)
        return actions
    
    def get_hand_euler_from_quat(self, hand_rot):
        hand_euler = get_euler_xyz(hand_rot.reshape(-1,4))
        hand_euler = torch.cat([hand_euler[2].reshape(-1,1),hand_euler[1].reshape(-1,1),hand_euler[0].reshape(-1,1)],-1)
        return hand_euler

    def set_underactuated_dof(self, env_ids, target):
        if target == 'hand_new_dof_state':
            self.hand_new_dof_state[env_ids, self.unactuated_dof_indices[0]] = self.hand_new_dof_state[env_ids, self.unactuated_dof_indices[0]-1]*0.8
            self.hand_new_dof_state[env_ids, self.unactuated_dof_indices[1]] = self.hand_new_dof_state[env_ids, self.unactuated_dof_indices[1]-1]*0.8
            self.hand_new_dof_state[env_ids, self.unactuated_dof_indices[2]] = self.hand_new_dof_state[env_ids, self.unactuated_dof_indices[2]-1]*0.8
            self.hand_new_dof_state[env_ids, self.unactuated_dof_indices[3]] = self.hand_new_dof_state[env_ids, self.unactuated_dof_indices[3]-1]*0.8
        elif target == 'cur_targets':
            self.cur_targets[env_ids, self.unactuated_dof_indices[0]] = self.cur_targets[env_ids, self.unactuated_dof_indices[0]-1]*0.8
            self.cur_targets[env_ids, self.unactuated_dof_indices[1]] = self.cur_targets[env_ids, self.unactuated_dof_indices[1]-1]*0.8
            self.cur_targets[env_ids, self.unactuated_dof_indices[2]] = self.cur_targets[env_ids, self.unactuated_dof_indices[2]-1]*0.8
            self.cur_targets[env_ids, self.unactuated_dof_indices[3]] = self.cur_targets[env_ids, self.unactuated_dof_indices[3]-1]*0.8

    def pre_physics_step(self, actions, extra=None):
        env_ids_all = torch.arange(start=0, end=self.num_envs, device=self.device, dtype=torch.long)

        if self.method == 'gf'  or self.method == 'gf+rl' or self.method == 'filter' or self.method == 'wristgen':
           
            if self.constrained:
                joint_actions = actions
            else:
                if self.robot_done:
                    stop = actions[:,self.num_actions:]
                    wrist_actions = actions[:,self.num_actions-6:self.num_actions]
                    joint_actions = actions[:,:self.num_actions-6]
                else:
                    wrist_actions = actions[:,self.num_actions-6:]
                    joint_actions = actions[:,:self.num_actions-6]

            self.scale, self.grad = extra

            # get current hand state
            
            cur_hand_dof = self.shadow_hand_dof_pos[:,self.actuated_dof_indices].clone()
            if self.has_base:
                cur_hand_pos = self.rigid_body_states[:, self.hand_mount_handle][:,:3].clone()
                cur_hand_quat = self.rigid_body_states[:, self.hand_mount_handle][:,3:7].clone()
            else:
                cur_hand_pos = self.hand_positions[self.hand_indices, :].clone()
                cur_hand_quat = self.hand_orientations[self.hand_indices, :].clone()

            if self.method == 'gf+rl':
                self.cur_hand_states = cur_hand_dof.clone()
                self.cur_obj_states = self.object_pose.clone()
                self.previous_hand_pos = cur_hand_pos.clone()

            # compute new hand state according to action
            joint_actions = joint_actions * self.action_speed
            # joint_actions = torch.clamp(self.dof_norm((self.dof_norm(cur_hand_dof.clone()) + joint_actions),inv=True) - cur_hand_dof,-self.action_speed,self.action_speed)
            new_hand_dof = torch.clamp(cur_hand_dof.clone() + joint_actions, self.shadow_hand_dof_lower_limits[self.actuated_dof_indices], self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
            
            if self.constrained:
                # fix hand base
                env_ids = torch.arange(start=0, end=self.num_envs, device=self.device, dtype=torch.long)
                if debug:
                    print(F.pairwise_distance(cur_hand_dof, self.previous_dof))
                    print(F.pairwise_distance(self.trajs[env_ids,self.progress_buf-1,:3], self.rigid_body_states[:, self.hand_mount_handle][:,:3]))
                    self.previous_dof = new_hand_dof
                new_hand_pos = self.trajs[env_ids,self.progress_buf,:3]
                new_hand_rot = self.trajs[env_ids,self.progress_buf,3:7]
                stop = torch.ones((actions.size(0),1),device=self.device)*-1
                self.actions = torch.cat([new_hand_pos, new_hand_rot, joint_actions, stop], dim=1)
            else:
                new_hand_pos = cur_hand_pos + wrist_actions[:,:3]
                new_hand_euler = torch.clamp(self.get_hand_euler_from_quat(cur_hand_quat) + wrist_actions[:,3:6], -6, 6)
                # this quat might not correct
                new_hand_rot = quat_from_euler_xyz(new_hand_euler[:,0],new_hand_euler[:,1],new_hand_euler[:,2])
                stop = torch.ones((actions.size(0),1),device=self.device)*-1
                self.actions = torch.cat([new_hand_pos, new_hand_rot, joint_actions, stop], dim=1)

            if self.mode=='eval':
                self.progress_buf[self.eval_finish_env_ids]*=0
            
            # set action to 
            self.cur_targets[:, self.actuated_dof_indices] = new_hand_dof
            if self.fake_tendon:
                self.set_underactuated_dof(env_ids=env_ids_all, target='cur_targets')

            if self.has_base:
                if self.constrained:
                    new_hand_euler = self.get_hand_euler_from_quat(new_hand_rot)
                self.cur_targets[:,self.hand_base_dof_indices] = torch.cat([new_hand_pos, new_hand_euler],-1)
            else:
                self.root_state_tensor[self.hand_indices, :3] = new_hand_pos
                self.root_state_tensor[self.hand_indices, 3:7] = new_hand_rot
                hand_indices = self.hand_indices.to(torch.int32)
                self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                            gymtorch.unwrap_tensor(self.root_state_tensor),
                                                            gymtorch.unwrap_tensor(hand_indices), len(hand_indices))
        else:
            # change action dim
            if not self.constrained:
                action_start_dim = 6
                action_end_dim = self.num_actions
                # TODO not constrained actions stop
            else:
                action_start_dim = 0
                action_end_dim = self.num_actions

            if self.use_relative_control:
                targets = self.prev_targets[:, self.actuated_dof_indices] + self.shadow_hand_dof_speed_scale * self.dt * self.actions[:,action_start_dim:action_end_dim]
                self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(targets,
                                                                            self.shadow_hand_dof_lower_limits[self.actuated_dof_indices], self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
            else:
                self.cur_targets[:, self.actuated_dof_indices] = scale(self.actions[:,action_start_dim:action_end_dim],
                                                                    self.shadow_hand_dof_lower_limits[self.actuated_dof_indices], self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
                self.cur_targets[:, self.actuated_dof_indices] = self.act_moving_average * self.cur_targets[:,
                                                                                                            self.actuated_dof_indices] + (1.0 - self.act_moving_average) * self.prev_targets[:, self.actuated_dof_indices]
                self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(self.cur_targets[:, self.actuated_dof_indices],
                                                                            self.shadow_hand_dof_lower_limits[self.actuated_dof_indices], self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])

        # if not self.constrained:
        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))

        if self.force_scale > 0.0:
            self.rb_forces *= torch.pow(self.force_decay, self.dt / self.force_decay_interval)

            # apply new forces
            force_indices = (torch.rand(self.num_envs, device=self.device) < self.random_force_prob).nonzero()
            self.rb_forces[force_indices, self.object_rb_handles, :] = torch.randn(
                self.rb_forces[force_indices, self.object_rb_handles, :].shape, device=self.device) * self.object_rb_masses * self.force_scale

            self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.rb_forces), None, gymapi.LOCAL_SPACE)

    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1

        if self.constrained:
            stop_env_ids = (self.progress_buf>=self.trajs_len).nonzero(as_tuple=False).squeeze(-1)
            if len(stop_env_ids)>0:
                self.actions[stop_env_ids,-1] = 1
        else:
            stop_env_ids = (self.progress_buf>=self.max_episode_length).nonzero(as_tuple=False).squeeze(-1)
            if len(stop_env_ids)>0:
                self.actions[stop_env_ids,-1] = 1

        env_ids = torch.arange(start=0, end=self.num_envs, device=self.device, dtype=torch.long)
        if not self.table_setting:
            self.reset_obj_vel(env_ids=env_ids)

        if self.method == 'gf'  or self.method == 'gf+rl' or self.method == 'filter' or self.method == 'wristgen':
            self.compute_observations(reset=False)
            self.compute_reward()
        else:
            self.compute_observations()
            self.compute_reward()
        

        if self.viewer and self.debug_viz:
            # draw axes on target object
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            for i in range(self.num_envs):
                if self.has_goal:
                    targetx = (self.goal_pos[i] + quat_apply(self.goal_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                    targety = (self.goal_pos[i] + quat_apply(self.goal_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                    targetz = (self.goal_pos[i] + quat_apply(self.goal_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                    p0 = self.goal_pos[i].cpu().numpy() + self.goal_displacement_tensor.cpu().numpy()
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], targetx[0], targetx[1], targetx[2]], [0.85, 0.1, 0.1])
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], targety[0], targety[1], targety[2]], [0.1, 0.85, 0.1])
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], targetz[0], targetz[1], targetz[2]], [0.1, 0.1, 0.85])

                objectx = (self.object_pos[i] + quat_apply(self.object_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                objecty = (self.object_pos[i] + quat_apply(self.object_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                objectz = (self.object_pos[i] + quat_apply(self.object_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.object_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], objectx[0], objectx[1], objectx[2]], [0.85, 0.1, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], objecty[0], objecty[1], objecty[2]], [0.1, 0.85, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], objectz[0], objectz[1], objectz[2]], [0.1, 0.1, 0.85])

    def draw_line(self, start_point, end_point):
        if self.viewer:
            # draw axes on target object
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            start_point = start_point.cpu().numpy()
            end_point = end_point.cpu().numpy()
            for i in range(self.num_envs):
                self.gym.add_lines(self.viewer, self.envs[i], 1, [start_point[i,0], start_point[i,1], start_point[i,2], end_point[i,0], end_point[i,1], end_point[i,2]], [1, 1, 1])

    def refresh_env_states(self, simulate_gym=True):
        if simulate_gym:
            self.gym.simulate(self.sim)
        
        self.gym.fetch_results(self.sim, True)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        if self.use_contact_sensor:
            self.gym.refresh_net_contact_force_tensor(self.sim)

        if self.obs_type == "full_state" or self.asymmetric_obs or self.method=='gf+rl':
            self.gym.refresh_force_sensor_tensor(self.sim)
            self.gym.refresh_dof_force_tensor(self.sim)
        
        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]

        if self.has_goal:
            self.goal_pose = self.goal_states[:, 0:7]
            self.goal_pos = self.goal_states[:, 0:3]
            self.goal_rot = self.goal_states[:, 3:7]

        self.fingertip_state = self.rigid_body_states[:, self.fingertip_handles][:, :, 0:13]
        self.fingertip_pos = self.rigid_body_states[:, self.fingertip_handles][:, :, 0:3]
        self.fingertip_center_pos = self.rigid_body_states[:, self.fingertip_center_handles][:, :, 0:3]
        self.fingertip_center_back_pos = self.rigid_body_states[:, self.fingertip_center_back_handles][:, :, 0:3]
        self.shadow_hand_rigid_body_poss = self.rigid_body_states[:, self.shadow_hand_rigid_body_handles][:, :, 0:3]
    
    # step simulation with any step_times
    def step_simulation(self, step_times=None):
        # step physics and render each frame
        if step_times is None:
            for i in range(1):
                self.gym.simulate(self.sim)
        else:
            for i in range(step_times):
                self.gym.simulate(self.sim)
            
        if self.force_render:
            self.render()
        # to fix!
        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)
        
        self.refresh_env_states()  
    
    '''
    for reward and filter grasp pose
    '''

    def lift_test(self, env_ids, flist=None, close_dis=0.3, close_dof_indices=None, only_evaluate_height=False):
        # generate stable grasp
        if close_dof_indices is None:
            close_dof_indices = self.close_dof_indices.clone()

        self.close(env_ids, flist, close_dis, close_dof_indices)

        self.reset_obj_vel(env_ids)
        # important reset object velocity and angular velocity to zero
        object_indices = torch.unique(torch.cat([self.object_indices[env_ids]]).to(torch.int32))
        self.root_state_tensor[self.object_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.object_indices[env_ids], 7:13])
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.root_state_tensor),
                                                        gymtorch.unwrap_tensor(object_indices), len(object_indices))
        # reset
        self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.rb_forces), None, gymapi.LOCAL_SPACE)
        
        self.refresh_env_states()
        object_pos_before_lift = self.root_state_tensor[self.object_indices, 0:3]
        if self.has_base:
            hand_pos_before_lift = self.rigid_body_states[:, self.hand_mount_handle][:,:3]
        else:
            hand_pos_before_lift = self.hand_positions[self.hand_indices, :]
        h2o_dist_before = F.pairwise_distance(hand_pos_before_lift,object_pos_before_lift)
        # lift object
        # TODO change lift step and distance, more specific lift
        # max_height = 0.6
        if self.has_base:
            step = 20
            for i in range(step):
                self.cur_targets[env_ids,self.hand_base_dof_indices[2]] = self.cur_targets[env_ids,self.hand_base_dof_indices[2]] + 0.05
                self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))
                self.gym.simulate(self.sim)
                self.refresh_env_states()
                if self.force_render:
                    self.render()
        else:
            step = 200
            apply_forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
            if 'cpu' in self.device:
                apply_forces[env_ids,self.hand_mount_handle,2] = 1
            else:
                apply_forces[env_ids,self.hand_mount_handle,2] = 100
            for i in range(step):
                self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(apply_forces), None, gymapi.ENV_SPACE)
                self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))
                self.gym.simulate(self.sim)
                self.refresh_env_states()
                if self.force_render:
                    self.render()
        object_pos_after_lift = self.root_state_tensor[self.object_indices, 0:3]
        if self.has_base:
            hand_pos_after_lift = self.rigid_body_states[:, self.hand_mount_handle][:,:3]
        else:
            hand_pos_after_lift = self.hand_positions[self.hand_indices, :]
        h2o_dist_after = F.pairwise_distance(hand_pos_after_lift,object_pos_after_lift)
        delta_object_pose = object_pos_after_lift[:,2] - object_pos_before_lift[:,2]
        height_bound = 1- object_pos_before_lift[:,2]
        delta_object_pose = torch.clamp(delta_object_pose,torch.zeros_like(height_bound),height_bound)
        if only_evaluate_height:
            self.lift_successes = torch.where((delta_object_pose>0.05), torch.ones_like(self.lift_successes), self.lift_successes)
        else:
            self.lift_successes = torch.where(((h2o_dist_after-h2o_dist_before)<0.05)&(delta_object_pose>0.1), torch.ones_like(self.lift_successes), self.lift_successes)
        return delta_object_pose

    def close(self, env_ids, flist=None, close_dis=0.3, close_dof_indices=None, check_contact=False):
        # f_close_dof_indices = [12.18]
        # s_close_dof_indices = [1,5,9,14,20]
        # t_close_dof_indices = [2,6,10,15,21]
        
        # for env_id in env_ids:
        #     self.cur_targets[env_id,close_dof_indices] += close_dis

        if check_contact:
            while True:
                close_dof_indices = self.tip_actuated_dof_indices
                self.cur_targets[env_ids.reshape(-1,1).repeat(1,len(close_dof_indices)),close_dof_indices.reshape(1,-1).repeat(len(env_ids),1)] += 0.001
                self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))
                self.gym.simulate(self.sim)
                if self.force_render:
                    self.render()
                self.refresh_env_states()
                contact_info = self.get_contact_info()
        else:
            for i in range(50):
                if i < 30:
                    self.cur_targets[env_ids.reshape(-1,1).repeat(1,len(close_dof_indices)),close_dof_indices.reshape(1,-1).repeat(len(env_ids),1)] += close_dis/30
                    self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))
                self.gym.simulate(self.sim)
                if self.force_render and i%1==0:
                    self.render()
                self.refresh_env_states()

    '''
    for filter grasp pose
    '''
    def grasp_filter(self, obj_ids=None, pose_id=None, close_dis=0.1, close_dof_indices=None, states=None, aug_data=False, test_time=2, move_hand=False, filter_threshold=0.01, hand_step_number=20, mode='state', pregrasp_coff=0.8, reset_coll=False):
        if close_dof_indices is None:
            close_dof_indices = self.close_dof_indices.clone()
        if states is None:
            if obj_ids is None:
                # reset all environments
                obj_ids = torch.arange(start=0, end=self.num_envs, device=self.device, dtype=torch.long)
            else:
                if 'int' in str(type(obj_ids)):
                    obj_ids = torch.tensor([obj_ids], device=self.device, dtype=torch.long)
                else:
                    obj_ids = to_torch(obj_ids, dtype=torch.long, device=self.device)

            if aug_data:
                success_time = torch.zeros(self.num_envs,device=self.device)
                for _ in range(test_time):
                    self.reset_idx(obj_ids, obj_ids, is_random=False, pose_id=pose_id)
                    cur_states = self.get_states()
                    # cur_states = self.get_states(refresh=False)
                    # target_dof = self.shadow_hand_dof_pos[obj_ids][:,self.actuated_dof_indices].clone()
                    # cur_states[obj_ids,:18] = unscale(target_dof, self.shadow_hand_dof_lower_limits[self.actuated_dof_indices], self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
                    cur_states = self.aug_data(cur_states, aug_dis=0)
                    
                    self.reset_obj_vel(env_ids=obj_ids)
                    if move_hand:
                        pregrasp_states = cur_states.clone()
                        pregrasp_states[:,self.tip_actuated_dof_indices_in_states] = unscale(target_dof[:,self.tip_actuated_dof_indices_in_states]-0.2, self.shadow_hand_dof_lower_limits[self.tip_actuated_dof_indices], self.shadow_hand_dof_upper_limits[self.tip_actuated_dof_indices])
                        self.set_states(pregrasp_states)
                        self.close(env_ids=obj_ids, close_dis=0.1, close_dof_indices=self.tip_actuated_dof_indices)
                        # self.move_hand_to_target_dof(obj_ids=obj_ids,target_hand_dof=target_dof)
                    else:
                        self.set_states(cur_states)
                    # TODO aug data not transofrm fingertip
                    cur_states = self.get_states()

                    if self.force_render:
                        self.render()

                    self.lift_successes*=0
                    self.lift_test(obj_ids,close_dis=close_dis, close_dof_indices=close_dof_indices)

                    if self.force_render:
                        self.render()
                    # may because this time object not lift, but hand in object, next turn, as simulation goes. object bump up.
                    success_time[obj_ids] += self.lift_successes[obj_ids]
                return success_time==test_time, cur_states
            else:
                if self.reset_sim_every_time:
                    self._reset_simulator(pose_id=pose_id, env_ids=obj_ids)

                if self.table_setting:
                    success_time = torch.zeros(self.num_envs,device=self.device)
                    valid_states = torch.tensor([],device=self.device)

                    if self.dataset_type=='ddg':
                        stable_object_pose = self.gen_stable_object_pose(obj_ids)
                    elif self.dataset_type=='dexgraspnet':
                        stable_object_pose = torch.zeros(self.num_envs, 7, device=self.device)
                        obj_scales = []

                    for i in range(test_time):
                        if self.dataset_type=='ddg':
                            self.reset_object_pose(env_ids=obj_ids, object_pose=stable_object_pose)
                        tmp_hand_pos = torch.tensor([],device=self.device)
                        tmp_hand_rot = torch.tensor([],device=self.device)
                        tmp_hand_dof = torch.tensor([],device=self.device)
                        for obj_id in obj_ids:
                            if self.dataset_type=='ddg':
                                candidate_grasp_path = random.choice(self.grasp_poses[obj_id])
                                hand_pose, object_pose = config_from_xml(self.pcl_shadow_hands[obj_id].hand, candidate_grasp_path)
                                # get isaac object pose
                                old_object_pos = to_torch(object_pose[:3],device=self.device).reshape(1,-1)
                                old_object_quat = to_torch([*object_pose[4:7], object_pose[3]],device=self.device).reshape(1,-1)
                                # get isaac hand pose
                                self.pcl_shadow_hands[obj_id].reset(hand_pose)
                                old_hand_pos, old_hand_quat, hand_dof = self.get_isaac_hand_state_from_pcl(env_id=obj_id)
                                # hand_dofs[obj_id] = hand_dof
                                # w2o@h2w -> h2o
                                old_w2o_quat, old_w2o_pos = transform_world2target(old_object_quat, old_object_pos) # w2h
                                h2o_pos, h2o_quat = multiply_transform(old_w2o_pos, old_w2o_quat, old_hand_pos.reshape(1,-1), old_hand_quat.reshape(1,-1))

                                # o2w_new@h2o -> h2w_new
                                new_object_pos = stable_object_pose[obj_id][:3].clone().reshape(1,-1)
                                new_object_quat = stable_object_pose[obj_id][3:7].clone().reshape(1,-1)
                                h2w_pos_new, h2w_quat_new = multiply_transform(new_object_pos, new_object_quat, h2o_pos, h2o_quat)
                            elif self.dataset_type=='dexgraspnet':
                                stable_object_pose[obj_id] = to_torch(self.obj_poses[obj_id][pose_id[obj_id]].copy(), device=self.device)
                                tmp_grasp_pose = to_torch(self.grasp_poses[obj_id][pose_id[obj_id]].copy(), device=self.device)
                                h2w_pos_new, h2w_quat_new = multiply_transform(stable_object_pose[obj_id][:3].unsqueeze(0), stable_object_pose[obj_id][3:7].unsqueeze(0), tmp_grasp_pose[:3].unsqueeze(0), tmp_grasp_pose[3:7].unsqueeze(0))
                                h2w_pos_new = h2w_pos_new
                                h2w_quat_new = h2w_quat_new
                                hand_dof = tmp_grasp_pose[7:25]
                                obj_scales.append(self.obj_scales[obj_id][pose_id[obj_id]].copy())
                            tmp_hand_pos = torch.cat([tmp_hand_pos, h2w_pos_new])
                            tmp_hand_rot = torch.cat([tmp_hand_rot, h2w_quat_new])
                            tmp_hand_dof = torch.cat([tmp_hand_dof, hand_dof.unsqueeze(0)])

                        if self.dataset_type == 'dexgraspnet':
                            if allow_obj_scale:
                                self.set_object_scale(env_ids=obj_ids, obj_scales=obj_scales)
                            self.reset_object_pose(env_ids=obj_ids, object_pose=stable_object_pose)
                            self.step_simulation(100)
                        elif self.dataset_type == 'ddg':
                            self.move_hand_to_target_pose(obj_ids=obj_ids,target_hand_pose=torch.cat([tmp_hand_pos,tmp_hand_rot],-1),threshold=hand_step_number, direct=False)
                        cur_states = self.get_states()
                        hand_dof_f = tmp_hand_dof.clone()
                        if mode=='state':
                            cur_states[obj_ids,:18] = unscale(hand_dof_f,self.shadow_hand_dof_lower_limits[self.actuated_dof_indices], self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
                            if self.dataset_type == 'dexgraspnet':
                                cur_states[obj_ids,18:21] = tmp_hand_pos
                                cur_states[obj_ids,21:25] = tmp_hand_rot
                            self.set_states(cur_states)
                            self.step_simulation(1)
                        elif mode=='partial':
                            hand_dof_f[obj_ids,:12]*=0
                            cur_states[obj_ids,:18] = unscale(hand_dof_f,self.shadow_hand_dof_lower_limits[self.actuated_dof_indices], self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
                            self.set_states(cur_states)
                            self.step_simulation(1)
                            self.move_hand_to_target_dof(obj_ids=obj_ids, target_hand_dof=tmp_hand_dof)
                        elif mode=='full':
                            self.move_hand_to_target_dof(obj_ids=obj_ids, target_hand_dof=tmp_hand_dof)
                        elif mode=='pregrasp':
                            hand_dof_f[obj_ids,self.distal_dof_indices]*=pregrasp_coff
                            # hand_dof_f[obj_ids,self.middle_dof_indices]=0
                            cur_states[obj_ids,:18] = unscale(hand_dof_f,self.shadow_hand_dof_lower_limits[self.actuated_dof_indices], self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
                            if self.dataset_type == 'dexgraspnet':
                                cur_states[obj_ids,18:21] = tmp_hand_pos
                                cur_states[obj_ids,21:25] = tmp_hand_rot
                            self.set_states(cur_states)
                            self.step_simulation(1)
                            self.move_hand_to_target_dof(obj_ids=obj_ids, target_hand_dof=tmp_hand_dof)
                        
                        if self.force_render:
                            self.render()

                        if self.dataset_type == 'ddg':
                            self.reset_obj_vel(obj_ids)
                            self.step_simulation(20)

                        gf_states = self.get_states()

                        # palm direction 
                        v_palm_origin_vec = to_torch([0, -1, 0.05, 0],dtype=torch.float, device=self.device).unsqueeze(0).repeat(len(obj_ids),1)
                        v_palm_current_vec = transform_points(tmp_hand_rot,v_palm_origin_vec)

                        # fingertip wrist direction
                        finger_tip_center = torch.mean(self.fingertip_pos,1)
                        wrist_pos = gf_states[:,18:21].clone()
                        vector = finger_tip_center - wrist_pos

                        # hand pose > min obj pcl  & hand pose > 0 & object not flying 
                        pcl_z = self.transform_obj_pcl_2_world(gen_pcl_with_ground=self.gen_pcl_with_ground)[:,:,2]
                        min_pcl_z = torch.min(pcl_z,1)[0]
                        min_handbody_z = torch.min(self.shadow_hand_rigid_body_poss[:,3:,2],1)[0]

                        num_bodies_valid = torch.sum(self.shadow_hand_rigid_body_poss[:,:,2] > 0,-1)
                        # lift_test_obj_ids = (((min_handbody_z - min_pcl_z) > filter_threshold)&(num_bodies_valid>29)&(min_pcl_z<0.005)&((-0.085 < vector[:,2]) & (vector[:,2] < 0))&(v_palm_current_vec[:,2]<-0.2)).nonzero(as_tuple=False).squeeze(-1)
                        lift_test_obj_ids = ((vector[:,2] < 0)).nonzero(as_tuple=False).squeeze(-1)
                        lift_test_obj_ids = to_torch(np.intersect1d(lift_test_obj_ids.cpu().numpy(), obj_ids.cpu().numpy()), dtype=torch.long)

                        self.reset_obj_vel(obj_ids)

                        self.lift_successes *= 0 
                        self.lift_test(lift_test_obj_ids,close_dis=close_dis, close_dof_indices=close_dof_indices)
                        if self.force_render:
                            self.render()
                        
                        valid_grasp_obj_ids = self.lift_successes.nonzero(as_tuple=False).squeeze(-1)
                        final_grasp_obj_ids = to_torch(np.intersect1d(lift_test_obj_ids.cpu().numpy(), valid_grasp_obj_ids.cpu().numpy()), dtype=torch.long)
                        # may because this time object not lift, but hand in object, next turn, as simulation goes. object bump up.
                        
                        gf_states = gf_states[final_grasp_obj_ids]
                        if len(gf_states) > 0:
                            self.set_states(gf_states,step_simulation_step=30)
                            # self.set_states(gf_states,6)
                            # self.step_simulation(5)
                            # self.move_hand_to_target_dof(obj_ids=final_grasp_obj_ids, target_hand_dof=scale(gf_states[:,:18],self.shadow_hand_dof_lower_limits[self.actuated_dof_indices], self.shadow_hand_dof_upper_limits[self.actuated_dof_indices]))
                            self.lift_successes *= 0 
                            self.lift_test(final_grasp_obj_ids,close_dis=close_dis, close_dof_indices=close_dof_indices)
                            double_check_success_ids = self.lift_successes[final_grasp_obj_ids].nonzero(as_tuple=False).squeeze(-1)
                            gf_states = gf_states[double_check_success_ids]
                            success_time[final_grasp_obj_ids] += self.lift_successes[final_grasp_obj_ids]
                            valid_states = torch.cat([valid_states, gf_states])
                    return success_time, valid_states
                else:
                    # return self.lift_successes, cur_states
                    success_time = torch.zeros(self.num_envs,device=self.device)
                    for _ in range(test_time):
                        self.reset_idx(obj_ids, obj_ids, is_random=False, pose_id=pose_id)
                        cur_states = states.clone()
                        cur_states = self.aug_data(cur_states, aug_dis=0)
                        self.set_states(cur_states)
                        # TODO aug data not transofrm fingertip
                        cur_states = self.get_states()

                        if self.force_render:
                            self.render()

                        self.lift_successes*=0
                        self.lift_test(obj_ids,close_dis=close_dis, close_dof_indices=close_dof_indices)

                        if self.force_render:
                            self.render()
                        # may because this time object not lift, but hand in object, next turn, as simulation goes. object bump up.
                        success_time[obj_ids] += self.lift_successes[obj_ids]
                return success_time>0, cur_states
        else:
            if move_hand:
                hand_dof = states[:,:18].to(self.device).float()
                hand_dof = scale(hand_dof,self.shadow_hand_dof_lower_limits[self.actuated_dof_indices], self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])

                if states.size(1) > 3000:
                    point_cloud_idx = 25 + self.points_per_object * 3
                    env_ids = states[:,point_cloud_idx+7:point_cloud_idx+8].to(self.device).long().squeeze(-1)
                else:
                    env_ids = states[:,32:33].to(self.device).long().squeeze(-1)
                
                self.move_hand_to_target_dof(obj_ids=env_ids, target_hand_dof=hand_dof)
            else:
                if mode=='pregrasp':
                    point_cloud_idx = 25 + self.points_per_object * 3
                    obj_ids = states[:,point_cloud_idx+7:point_cloud_idx+8].to(self.device).long().squeeze(-1)
                    hand_dof_f = self.dof_norm(states[:,:18],inv=True)
                    tmp_hand_dof = hand_dof_f.clone()
                    hand_dof_f[obj_ids,:13]*=pregrasp_coff
                    states[obj_ids,:13] = unscale(hand_dof_f,self.shadow_hand_dof_lower_limits[self.actuated_dof_indices], self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])[:,:13]
                    self.set_states(states)
                    self.step_simulation(1)
                    self.move_hand_to_target_dof(obj_ids=obj_ids, target_hand_dof=tmp_hand_dof, open_loop=False)
                else:
                    if self.diff_obj_scale:
                        point_cloud_idx = 25 + self.points_per_object * 3
                        obj_ids = states[:,point_cloud_idx+7:point_cloud_idx+8].to(self.device).long().squeeze(-1)
                        obj_scales = states[:,-1].to(self.device).squeeze(-1)
                        # set_trace()
                        if reset_coll:
                            self.disable_collision = False
                        self._reset_simulator(env_ids=obj_ids, obj_scales=obj_scales)

                    self.set_states(states,step_simulation_step=30)
                    self.step_simulation(1)
                # hand_dof = states[:,:18].to(self.device).float().clone()
                # hand_dof = scale(hand_dof,self.shadow_hand_dof_lower_limits[self.actuated_dof_indices], self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
                # tmp_hand_dof = hand_dof.clone()

                # point_cloud_idx = 25 + self.points_per_object * 3
                # obj_ids = states[:,point_cloud_idx+7:point_cloud_idx+8].to(self.device).long().squeeze(-1)
                # hand_dof[:,:18]*=pregrasp_coff
                # states[:,:18] = unscale(hand_dof,self.shadow_hand_dof_lower_limits[self.actuated_dof_indices], self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
                # self.set_states(states)
                # self.step_simulation(1)
                # self.move_hand_to_target_dof(obj_ids=obj_ids, target_hand_dof=tmp_hand_dof)

            point_cloud_idx = 25 + self.points_per_object * 3
            obj_ids = states[:,point_cloud_idx+7:point_cloud_idx+8].to(self.device).long().squeeze(-1)
            # self.gen_stable_object_pose(obj_ids=obj_ids, reset_object_pose=False)

            if self.force_render:
                self.render()

            if self.method=='wristgen':
                stable_states = self.get_states()

            self.lift_successes*=0
            self.lift_test(obj_ids,close_dis=close_dis, close_dof_indices=close_dof_indices)

            if self.force_render:
                self.render()

            if reset_coll:
                self.disable_collision = True
                lift_success_env_ids = self.lift_successes.nonzero(as_tuple=False).squeeze(-1)
                self.extras['success_num'] = to_torch(len(lift_success_env_ids), dtype=torch.float, device=self.device).unsqueeze(-1)

            if self.method=='wristgen':
                return self.lift_successes[obj_ids], stable_states
            else:
                return self.lift_successes[obj_ids]
    '''
    generate trajectory for constrained env
    '''
    def traj_gen(self, states=False, trajectory_length_range=[0.15, 0.2], angle_range=[0, 20 / 180 * np.pi], random_joint=False, relative=False, points_per_object=1024, hand_pos_only=False, hand_rotation=False, env_ids=None, start_joint_noise=0.0, gf_rot=False, gf_dof=False):
        human_hand_speed = random.uniform(*self.human_hand_speed)
        traj_range_lower = self.detection_freq*trajectory_length_range[0]/human_hand_speed
        traj_range_upper = self.detection_freq*trajectory_length_range[1]/human_hand_speed
        traj_steps_range = [episode_length, episode_length]

        if hand_pos_only:
            if states!=False:
                B = states.size(0)
                hand_dof = self.dof_norm(states[:,:18].clone().to(self.device).float(),inv=True)
                hand_pos = states[:,18:21].clone().to(self.device).float()
                hand_rot = states[:,21:25].clone().to(self.device).float()
                point_cloud_idx = 25 + points_per_object * 3
                fingertip_pos = states[:,point_cloud_idx+8:point_cloud_idx+23].to(self.device).float().reshape(B, -1, 3)
            else:
                if self.has_base:
                    hand_pos = self.rigid_body_states[env_ids, self.hand_mount_handle][:,:3]
                    hand_rot = self.rigid_body_states[env_ids, self.hand_mount_handle][:,3:7]
                else:
                    hand_pos = self.hand_positions[self.hand_indices[env_ids], :]
                    hand_rot = self.hand_orientations[self.hand_indices[env_ids], :]
                hand_dof = self.shadow_hand_dof_pos[env_ids][:,self.actuated_dof_indices].clone()
                fingertip_pos = self.fingertip_pos[env_ids]
                B = hand_pos.size(0)
        else:
            B = states.size(0)
            # self.set_states(states)
            hand_dof = states[:,:18].clone().to(self.device).float()
            hand_pos = states[:,18:21].clone().to(self.device).float()
            hand_rot = states[:,21:25].clone().to(self.device).float()
            point_cloud_idx = 25 + points_per_object * 3
            obj_pcl = states[:,25:point_cloud_idx].clone().to(self.device).float().reshape(B, -1, 3)
            obj_pos = states[:,point_cloud_idx:point_cloud_idx+3].to(self.device).float()
            obj_rot = states[:,point_cloud_idx+3:point_cloud_idx+7].to(self.device).float()
            env_ids = states[:,point_cloud_idx+7:point_cloud_idx+8].to(self.device).long().squeeze(-1)
            fingertip_pos = states[:,point_cloud_idx+8:point_cloud_idx+23].to(self.device).float().reshape(B, -1, 3)
        
        if hand_rotation:
            e_hand_euler = get_euler_xyz(hand_rot)
            e_hand_euler = torch.cat([*e_hand_euler]).reshape(3,B).permute(1,0)
            if self.use_human_rot:
                delta_hand_euler = to_torch(gamma.rvs(*self.rotation_dist_params,B*3), device=self.device).reshape(B,3)
                s_hand_euler = torch.clamp(e_hand_euler+delta_hand_euler, 0.08, 6.2)
            else:
                delta_hand_euler = to_torch(gamma.rvs(*self.rotation_dist_params,B*3), device=self.device).reshape(B,3)
                s_hand_euler = e_hand_euler * (1 - delta_hand_euler/6.28)

        # hand_pose: dict, {"pos":[B, 3], "quat":[B, 4]}
        # finger_tip_location: dict, {"finger_1":[B, 3], "finger_2":[B, 3], "finger_3":[B, 3], "finger_4":[B, 3], "finger_5":[B, 3]}
        # trajectory_length_range: list, [min length, max length]
        # angle_range: list, [min angle, max angle]
        e_hand_pos = hand_pos
        finger_tip_center = torch.mean(fingertip_pos,1)

        trajectory_length = trajectory_length_range[0] + torch.rand(B, 1, device=self.device) * \
                            (trajectory_length_range[1] - trajectory_length_range[0])
        angle_thres = angle_range[0] + torch.rand(B, 1, device=self.device) * (angle_range[1] - angle_range[0])
        self.angle_thres = angle_thres.clone()
            
        vector = e_hand_pos - finger_tip_center
        vector = (vector.permute(1, 0) / torch.norm(vector, dim=1)).permute(1, 0)
        
        rot_axis = torch.tensor([0., 0., 1.], device=self.device).expand(B, 3)
        rot_vec = torch.cross(vector, rot_axis)
        theta = torch.arccos(torch.sum(torch.multiply(vector, rot_axis), dim=1))
        
        rot = rodrigues_to_rotation(rot_vec, theta)
        new_vector = rot.permute(0, 2, 1) @ sample_from_circle(angle_thres)
        new_vector = new_vector.squeeze()
        s_hand_pos = e_hand_pos + new_vector * trajectory_length
        
        # angle = angle_from_vector(new_vector, vector)
        # print("angle: ", '\n', angle.numpy() / 3.1415926 * 180)

        trajs = torch.tensor([],device=self.device)
        traj_len = int(torch_rand_float(traj_steps_range[0], traj_steps_range[1], (1, 1), device=self.device).reshape(-1).repeat(B)[0])

        if start_joint_noise!=0:
            s_hand_dof = torch_random_sample(self.shadow_hand_dof_lower_limits[self.actuated_dof_indices], self.shadow_hand_dof_lower_limits[self.actuated_dof_indices]+start_joint_noise*self.shadow_hand_dof_range[self.actuated_dof_indices], (B, 18), device=self.device)
        else:
            s_hand_dof = torch.zeros((B,18), device=self.device)

        if hand_pos_only:
            e_hand_dof = hand_dof
        else:
            e_hand_dof = scale(hand_dof,self.shadow_hand_dof_lower_limits[self.actuated_dof_indices], self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])

        # gen relative pos gf data
        if relative:
            step = torch.randint(0,traj_len,(B,1), device=self.device)
            
            cur_hand_pos = s_hand_pos + (e_hand_pos - s_hand_pos) * step / (traj_len-1)  
            states[:,18:21] = cur_hand_pos

            if gf_dof:
                cur_hand_dof = s_hand_dof + (e_hand_dof - s_hand_dof) * step / (traj_len-1) 
                states[:,:18] = cur_hand_dof
            if gf_rot:
                cur_hand_euler = s_hand_euler + (e_hand_euler - s_hand_euler) * step / (traj_len-1) 
                cur_hand_quat = quat_from_euler_xyz(cur_hand_euler.permute(1,0)[0],cur_hand_euler.permute(1,0)[1],cur_hand_euler.permute(1,0)[2])
                states[:,21:25] = cur_hand_quat
            # self.set_states(states)
            return states
        else:
            if self.use_human_trajs:
                new_patterns = self.gen_new_pattern(B=B, traj_len=traj_len, delta_pos=abs(e_hand_pos-s_hand_pos), pattern_type='pos')
                sampled_trajs = new_patterns.reshape(B,3,traj_len).permute(0,2,1)
                
                sampled_traj_move = sampled_trajs[:,-1,:] - sampled_trajs[:,0,:]
                traj_scales = (e_hand_pos - s_hand_pos) / sampled_traj_move

                if self.use_human_rot:
                    new_patterns_rot = self.gen_new_pattern(B=B, traj_len=traj_len, delta_pos=abs(e_hand_euler-s_hand_euler), pattern_type='rot')
                    sampled_trajs_rot = new_patterns_rot.reshape(B,3,traj_len).permute(0,2,1)
                    
                    sampled_trajs_rot_move = sampled_trajs_rot[:,-1,:] - sampled_trajs_rot[:,0,:]
                    traj_scales_rot = (e_hand_euler - s_hand_euler) / sampled_trajs_rot_move

                # new_rot_patterns = self.gen_new_pattern(B=B, traj_len=traj_len)
                # for traj_id in range(B):
                #     ori_traj_len_x = len(patterns[3*traj_id])
                #     ori_traj_len_y = len(patterns[3*traj_id+1])
                #     ori_traj_len_z = len(patterns[3*traj_id+2])

                #     sample_interval_x = np.floor(ori_traj_len_x/traj_len)
                #     sample_interval_y = np.floor(ori_traj_len_y/traj_len)
                #     sample_interval_z = np.floor(ori_traj_len_z/traj_len)

                #     sample_traj_steps_x = np.arange(0, ori_traj_len_x, sample_interval_x, dtype=np.int32)[:traj_len]
                #     sample_traj_steps_y = np.arange(0, ori_traj_len_y, sample_interval_y, dtype=np.int32)[:traj_len]
                #     sample_traj_steps_z = np.arange(0, ori_traj_len_z, sample_interval_z, dtype=np.int32)[:traj_len]

                #     sampled_traj_x = patterns[3*traj_id][sample_traj_steps_x]
                #     sampled_traj_y = patterns[3*traj_id+1][sample_traj_steps_y]
                #     sampled_traj_z = patterns[3*traj_id+2][sample_traj_steps_z]

                #     sampled_traj_move_x = sampled_traj_x[-1] - sampled_traj_x[0]
                #     sampled_traj_move_y = sampled_traj_y[-1] - sampled_traj_y[0]
                #     sampled_traj_move_z = sampled_traj_z[-1] - sampled_traj_z[0]

                #     sampled_trajs[traj_id][:,0] = to_torch(sampled_traj_x,device=self.device)
                #     sampled_trajs[traj_id][:,1] = to_torch(sampled_traj_y,device=self.device)
                #     sampled_trajs[traj_id][:,2] = to_torch(sampled_traj_z,device=self.device)

                #     traj_scales[traj_id,0] = (e_hand_pos - s_hand_pos)[traj_id][0]/sampled_traj_move_x
                #     traj_scales[traj_id,1] = (e_hand_pos - s_hand_pos)[traj_id][1]/sampled_traj_move_y
                #     traj_scales[traj_id,2] = (e_hand_pos - s_hand_pos)[traj_id][2]/sampled_traj_move_z


            for step in range(traj_len):
                if self.use_human_trajs:
                    if step == 0:
                        cur_hand_pos = s_hand_pos.clone()
                        if hand_rotation:
                            hand_rot_euler = s_hand_euler.clone()
                    elif step < traj_len:
                        cur_hand_pos = cur_hand_pos + (sampled_trajs[:,step,:] - sampled_trajs[:,step-1,:])*traj_scales + torch_rand_float(-self.traj_noise,self.traj_noise,(B,3), device=self.device)
                        if hand_rotation:
                            if self.use_human_rot:
                                hand_rot_euler = hand_rot_euler + (sampled_trajs_rot[:,step,:] - sampled_trajs_rot[:,step-1,:])*traj_scales_rot + torch_rand_float(-self.traj_noise,self.traj_noise,(B,3), device=self.device)
                            else:
                                hand_rot_euler = s_hand_euler + (e_hand_euler - s_hand_euler) * step / (traj_len-1) 
                else:
                    cur_hand_pos = s_hand_pos + (e_hand_pos - s_hand_pos) * step / (traj_len-1)  
                    if hand_rotation:
                        hand_rot_euler = s_hand_euler + (e_hand_euler - s_hand_euler) * step / (traj_len-1)  
                
                if random_joint:
                    cur_hand_dof = torch_random_sample(self.shadow_hand_dof_lower_limits[self.actuated_dof_indices], self.shadow_hand_dof_lower_limits[self.actuated_dof_indices]+start_joint_noise*self.shadow_hand_dof_range[self.actuated_dof_indices], (B, 18), device=self.device)
                else:
                    cur_hand_dof = s_hand_dof + (e_hand_dof - s_hand_dof) * step / (traj_len-1)  
                
                if not hand_pos_only:
                    states[:,18:21] = cur_hand_pos
                    states[:,:18] = unscale(cur_hand_dof,self.shadow_hand_dof_lower_limits[self.actuated_dof_indices], self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
                
                # self.set_states(states)
                # self.reset_hand_pose(env_ids, (cur_hand_pos, hand_rot, cur_hand_dof))

                if hand_pos_only:
                    # self.reset_hand_poses(env_ids, (cur_hand_pos, hand_rot, cur_hand_dof))
                    if hand_rotation:
                        hand_rot = quat_from_euler_xyz(hand_rot_euler.permute(1,0)[0],hand_rot_euler.permute(1,0)[1],hand_rot_euler.permute(1,0)[2])
                    if step==0:
                        trajs = torch.cat([trajs,torch.cat([cur_hand_pos, hand_rot, cur_hand_dof], dim=1)]).reshape(B,1,-1)
                    elif step>0:
                        trajs = torch.cat([trajs,torch.cat([cur_hand_pos, hand_rot, cur_hand_dof], dim=1).reshape(B,1,-1)],1)
                else:
                    if step==0:
                        trajs = torch.cat([trajs,states]).reshape(B,1,-1)
                    else:
                        trajs = torch.cat([trajs,states.reshape(B,1,-1)],1)
                    
                if self.force_render:
                    self.render()

            return trajs
    
    def sample_human_traj_pattern(self, B, traj_len, delta_pos=None, number_axis=3, pattern_type=None):
        patterns_a = torch.zeros(B*number_axis,self.human_traj_len, device=self.device)
        for path_order in range(B):
            for (i_axis, delta_pos_each) in enumerate(delta_pos[path_order]):
                if pattern_type == 'pos':
                    if delta_pos_each < 0.02:
                        sample_axis = random.choice(['x_pos', 'z_pos'])
                    else:
                        sample_axis = 'y_pos'
                elif pattern_type == 'rot':
                    sample_axis = random.choice(['roll_pos', 'pitch_pos', 'yaw_pos'])
                human_trajs = self.human_pattern[sample_axis]
                patterns_idx = torch_random_sample(0,self.human_traj_pattern_num,(1,),device=self.device).to(torch.long).squeeze()
                patterns_a[path_order*number_axis+i_axis] = human_trajs[patterns_idx,:].clone()
        
        sample_interval = np.floor(self.human_traj_len/traj_len)
        sample_traj_steps = np.arange(0, self.human_traj_len, sample_interval, dtype=np.int32)[:traj_len]
        sample_traj_steps = sample_traj_steps + self.human_traj_len - (sample_traj_steps[-1]+1)
        patterns_a = patterns_a[:,sample_traj_steps]
        return patterns_a

    def gen_new_pattern(self, B, traj_len, delta_pos=None, number_axis=3, pattern_type=None):
        if pattern_type=='pos':
            patterns_a = self.sample_human_traj_pattern(B=B, traj_len=traj_len, delta_pos=delta_pos, number_axis=number_axis, pattern_type=pattern_type)

            if self.env_mode == 'train':
                patterns_b = self.sample_human_traj_pattern(B=B, traj_len=traj_len, delta_pos=delta_pos, number_axis=number_axis, pattern_type=pattern_type)
                coeff = random.uniform(0,1)
                new_patterns = patterns_a * coeff + patterns_b * (1-coeff)
                return new_patterns
            else:
                return patterns_a
        elif pattern_type=='rot':
            patterns_a = self.sample_human_traj_pattern(B=B, traj_len=traj_len, delta_pos=delta_pos, number_axis=number_axis, pattern_type=pattern_type)

            if self.env_mode == 'train':
                patterns_b = self.sample_human_traj_pattern(B=B, traj_len=traj_len, delta_pos=delta_pos, number_axis=number_axis, pattern_type=pattern_type)
                coeff = random.uniform(0,1)
                new_patterns = patterns_a * coeff + patterns_b * (1-coeff)
                return new_patterns
            else:
                return patterns_a

    def gen_stable_object_pose(self, obj_ids=None, reset_object_pose=True, step=500):
        if obj_ids is None:
            obj_ids = torch.arange(start=0, end=self.num_envs, device=self.device, dtype=torch.long)

        if reset_object_pose:
            self.reset_object_pose(env_ids=obj_ids)
        cur_object_pose = self.get_objects_states(obj_ids)

        stable_step = 0
        stable = False
        while not stable and stable_step < step:
            self.gym.simulate(self.sim)
            next_object_pose = self.get_objects_states(obj_ids)
            # print(torch.sum(abs(next_object_pose-cur_object_pose),dim=1))
            # check stable
            if (torch.sum(abs(next_object_pose-cur_object_pose),dim=1) < 5e-5).all():
                stable = True
            cur_object_pose = next_object_pose
            stable_step += 1
            if self.force_render:
                self.render()
        return cur_object_pose
    
    def step_hand(self,hand_pos=None,hand_quat=None,hand_dof=None, env_ids_all=None):
        # print(self.root_state_tensor[self.hand_indices[env_id], :7])
        if env_ids_all is None:
            env_ids_all = torch.arange(start=0, end=self.num_envs, device=self.device, dtype=torch.long)

        self.cur_targets[env_ids_all.reshape(-1,1).repeat(1,len(self.actuated_dof_indices)),self.actuated_dof_indices.reshape(1,-1).repeat(len(env_ids_all),1)] += hand_dof

        if self.fake_tendon:
            self.set_underactuated_dof(env_ids=env_ids_all, target='cur_targets')

        if self.has_base:
            hand_euler = self.get_hand_euler_from_quat(hand_quat)
            self.cur_targets[env_ids_all.reshape(-1,1).repeat(1,len(self.hand_base_dof_indices)),self.hand_base_dof_indices.reshape(1,-1).repeat(len(env_ids_all),1)] = torch.cat([hand_pos, hand_euler],-1)
        else:
            self.root_state_tensor[self.hand_indices, :3] = hand_pos
            self.root_state_tensor[self.hand_indices, 3:7] = hand_quat
            hand_indices = self.hand_indices.to(torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.root_state_tensor),
                                                        gymtorch.unwrap_tensor(hand_indices), len(hand_indices))

        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))

        self.step_simulation(1)

    '''
    for easy access to states and images
    '''
    def get_states(self, is_pcl=True, reset=False, refresh=True, gf_state=False):
        if refresh:
            self.refresh_env_states()

        if self.obs_type == "gf":
            self.compute_gf_state(reset=reset, gf_state=gf_state)
        else:
            print("Unknown observations type!")

        if self.obs_type == "gf":
            point_cloud_idx = 25 + self.points_per_object * 3
            if is_pcl:
                pcl_obs = torch.clamp(self.obs_buf[:,:point_cloud_idx+7], -self.clip_obs, self.clip_obs).to(self.rl_device)
                envid_obs = self.obs_buf[:,point_cloud_idx+7:point_cloud_idx+8].to(self.rl_device)
                fingertip_obs =  torch.clamp(self.obs_buf[:,point_cloud_idx+8:point_cloud_idx+23], -self.clip_obs, self.clip_obs).to(self.rl_device)
                objscale_obs = self.obs_buf[:,point_cloud_idx+23:point_cloud_idx+24].to(self.rl_device)
                self.obs_dict["obs"] = torch.cat([pcl_obs, envid_obs, fingertip_obs, objscale_obs],-1)
                return self.obs_dict["obs"]
            else:
                no_pcl_obs = torch.cat([self.obs_buf[:,:25],self.obs_buf[:,point_cloud_idx:point_cloud_idx+7]], -1)
                no_pcl_obs = torch.clamp(no_pcl_obs, -self.clip_obs, self.clip_obs).to(self.rl_device)
                envid_obs = self.obs_buf[:,point_cloud_idx+7:point_cloud_idx+8].to(self.rl_device)
                fingertip_obs =  torch.clamp(self.obs_buf[:,point_cloud_idx+8:point_cloud_idx+23], -self.clip_obs, self.clip_obs).to(self.rl_device)
                objscale_obs = self.obs_buf[:,point_cloud_idx+23:point_cloud_idx+24].to(self.rl_device)
                self.obs_dict["obs"] = torch.cat([no_pcl_obs, envid_obs, fingertip_obs, objscale_obs],-1)
                return self.obs_dict["obs"]

    def get_images(self, env_ids=None, img_size=256, vis_env_num=0):
        if env_ids is None:
            env_ids = torch.arange(start=0, end=self.num_envs, device=self.device, dtype=torch.long)

        # vis part env
        env_ids = env_ids[:vis_env_num]
        # step the physics simulation
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        # communicate physics to graphics system
        self.gym.step_graphics(self.sim)

        # render the camera sensors
        self.gym.render_all_camera_sensors(self.sim)

        if self.force_render:
            self.render()

        images = []
        # get rgb image
        for env_id in env_ids:
            image = self.gym.get_camera_image(self.sim, self.envs[env_id], self.cameras_handle[env_id], gymapi.IMAGE_COLOR)
            image = np.reshape(image, (np.shape(image)[0],-1,4))[...,:3]
            image = image[:,:,(2,1,0)]
            image = cv2.resize(image,(img_size,img_size))
            images.append(image)
        
        images = np.stack(images, axis=0)
        images = to_torch(images, device=self.device)
        return images

    def get_objects_states(self, obj_ids):
        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)
        self.refresh_env_states()
        return self.root_state_tensor[self.object_indices[obj_ids], 0:7]
    
    def get_handa_states(self, obj_ids):
        h_dof = self.shadow_hand_dof_pos[obj_ids][:,self.actuated_dof_indices].clone()
        if self.has_base:
            hand_pos = self.rigid_body_states[obj_ids, self.hand_mount_handle][:,:3].clone()
            hand_quat = self.rigid_body_states[obj_ids, self.hand_mount_handle][:,3:7].clone()
        else:
            hand_pos = self.hand_positions[self.hand_indices[obj_ids], :].clone()
            hand_quat = self.hand_orientations[self.hand_indices[obj_ids], :].clone()
        return hand_pos, hand_quat, h_dof

    def set_states(self, states, refresh=True, step_simulation_step=12, reset_sim=False, vis_state_input=False):
        if self.diff_obj_scale and reset_sim:
            point_cloud_idx = 25 + self.points_per_object * 3
            obj_ids = states[:,point_cloud_idx+7:point_cloud_idx+8].to(self.device).long().squeeze(-1)
            obj_scales = states[:,-1].to(self.device).squeeze(-1)
            # set_trace()
            if vis_state_input:
                self.vis_states_input = vis_state_input
                self.vis_states = states
            self._reset_simulator(env_ids=obj_ids, obj_scales=obj_scales)
        self.reset_gf_env(states, is_random=False, refresh=refresh, step_simulation_step=step_simulation_step)

        # if step_simulation_step is not None:
        #     self.step_simulation(step_simulation_step)

        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)
        
        if refresh:
            self.refresh_env_states()  

    '''
    aug given dex data
    '''
    def aug_data(self, dex_data, aug_dis=0.5, aug_rot=0, relative=False, points_per_object=1024):
        batch_size = dex_data.size(0)

        # get original hand state
        hand_dof = dex_data[:,:18].clone().to(self.device).float()
        hand_pos = dex_data[:,18:21].clone().to(self.device).float()
        hand_rot = dex_data[:,21:25].clone().to(self.device).float()
        point_cloud_idx = 25 + points_per_object * 3
        obj_pcl = dex_data[:,25:point_cloud_idx].clone().to(self.device).float().reshape(batch_size, -1, 3)
        obj_pos = dex_data[:,point_cloud_idx:point_cloud_idx+3].to(self.device).float()
        obj_rot = dex_data[:,point_cloud_idx+3:point_cloud_idx+7].to(self.device).float()
        obj_ids = dex_data[:,point_cloud_idx+7:point_cloud_idx+8].to(self.device).long()

        if not self.table_setting:
            # randomize new obj pos [-aug_dis, aug_dis]
            rand_obj_pos = torch_rand_float(-aug_dis, aug_dis, (batch_size, 3), device=self.device)
            # randomize new obj ori [-aug_rot, aug_rot]
            x_unit_tensor = to_torch([1, 0, 0], dtype=torch.float, device=self.device).repeat((batch_size, 1))
            y_unit_tensor = to_torch([0, 1, 0], dtype=torch.float, device=self.device).repeat((batch_size, 1))
            z_unit_tensor = to_torch([0, 0, 1], dtype=torch.float, device=self.device).repeat((batch_size, 1))
            rand_floats = torch_rand_float(-aug_rot, aug_rot, (batch_size, 4), device=self.device)
            rand_obj_rot = randomize_rotation_xyz(rand_floats[:, 0], rand_floats[:, 1], rand_floats[:, 2], x_unit_tensor, y_unit_tensor, z_unit_tensor)
        else:
            # rand_object_pose = self.gen_stable_object_pose(obj_ids)
            rand_obj_pos = obj_pos.clone()
            rand_obj_rot = obj_rot.clone()
            random_move = torch_rand_float(-aug_dis, aug_dis, (batch_size, 2), device=self.device)
            rand_obj_pos[:,:2] += random_move
        # w2o_old
        w2o_quat, w2o_pos = transform_world2target(obj_rot, obj_pos)

        # new obj pcl
        # o_new2w @ w2o_old -> o_new2o_old
        o_rand2o_pos, o_rand2o_quat = multiply_transform(rand_obj_pos, rand_obj_rot, w2o_pos, w2o_quat)
        #o_pcl_old2w @ o_new2o_old -> o_pcl_neww   
        B, N, _ = obj_pcl.size()
        padding = torch.zeros([B, N, 1]).to(obj_pcl.device)
        obj_pcl_pad = torch.cat([obj_pcl,padding],-1)
        o_rand2o_quat = o_rand2o_quat.reshape(B,-1,4).expand_as(obj_pcl_pad)
        o_pcl_new_pos = transform_points(o_rand2o_quat, obj_pcl_pad)
        o_rand2o_pos = o_rand2o_pos.reshape(B,-1,3).expand_as(o_pcl_new_pos)
        o_pcl_new_pos += o_rand2o_pos

        # new hand state
        # w2o@h2w -> h2o
        h2o_pos, h2o_quat = multiply_transform(w2o_pos, w2o_quat, hand_pos.reshape(batch_size,-1), hand_rot.reshape(batch_size,-1))
        # o2w_new@h2o -> h2w_new
        h2w_pos_random, h2w_quat_random = multiply_transform(rand_obj_pos, rand_obj_rot, h2o_pos, h2o_quat)

        if relative:
            w2h_quat, w2h_pos = transform_world2target(h2w_quat_random, h2w_pos_random)
            o_pcl_new_pos = self.transform_obj_pcl_2_hand(o_pcl_new_pos, w2h_quat, w2h_pos)

        # TODO transform fingertip
        # assgin new state to ori data
        dex_data[:,18:21] = h2w_pos_random
        dex_data[:,21:25] = h2w_quat_random
        dex_data[:,25:point_cloud_idx] = o_pcl_new_pos.reshape(B,-1)
        dex_data[:,point_cloud_idx:point_cloud_idx+3] = rand_obj_pos
        dex_data[:,point_cloud_idx+3:point_cloud_idx+7] = rand_obj_rot
        
        return dex_data

    def get_obj2hand(self,o_pcl, h_quat, h_pos):
        w2h_quat, w2h_pos = transform_world2target(h_quat, h_pos)
        o_pcl_new_pos = self.transform_obj_pcl_2_hand(o_pcl, w2h_quat, w2h_pos)
        return o_pcl_new_pos
    
    def transform_target2source(self, s_quat, s_pos, t_quat, t_pos):
        w2s_quat, w2s_pos = transform_world2target(s_quat, s_pos)
        t2s_pos, t2s_quat = multiply_transform(w2s_pos, w2s_quat, t_pos, t_quat)
        return t2s_pos, t2s_quat

    def get_h2o_pose(self, o_quat, o_pos, h_quat, h_pos):
        w2o_quat, w2o_pos = transform_world2target(o_quat, o_pos)
        h2o_pos, h2o_quat = multiply_transform(w2o_pos, w2o_quat, h_pos, h_quat)
        return h2o_pos, h2o_quat
    
    def dof_norm(self, hand_dof, inv=False):
        if inv:
            hand_dof = scale(hand_dof,self.shadow_hand_dof_lower_limits[self.actuated_dof_indices], self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
        else:
            hand_dof = unscale(hand_dof,self.shadow_hand_dof_lower_limits[self.actuated_dof_indices], self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
        
        return hand_dof
    '''
    for test
    '''
    def move_hand_to_target(self, obj_ids, target_hand_dof, target_hand_pose, step_number=20):
        target_hand_pos = target_hand_pose[:,:3]
        target_hand_quat = target_hand_pose[:,3:7]
        cur_hand_pos, cur_hand_rot, cur_hand_dof = self.get_handa_states(obj_ids)
        step=0
        while ((torch.sum(abs(target_hand_dof - cur_hand_dof),-1) > 1e-3).any() or (torch.sum(abs(target_hand_pos - cur_hand_pos),-1) > 1e-3).any() or (torch.sum(abs(target_hand_quat - cur_hand_rot),-1) > 1e-3).any()) and step < step_number:
            delta_hand_dof = target_hand_dof - cur_hand_dof
            self.step_hand(hand_pos=target_hand_pos, hand_quat=target_hand_quat, hand_dof=delta_hand_dof, env_ids_all=obj_ids)
            _, _, cur_hand_dof = self.get_handa_states(obj_ids)
            step += 1
        return cur_hand_dof
        
    # test hand dof
    def move_hand_to_target_dof(self, obj_ids, target_hand_dof, step_number=20, open_loop=False, obj_no_move=False):
        cur_hand_pos, cur_hand_rot, cur_hand_dof = self.get_handa_states(obj_ids)
        init_obj_pos = self.object_pos.clone()
        step = 0
        if open_loop:
            delta_hand_dof = target_hand_dof - cur_hand_dof
            for step in range(step_number):
                self.step_hand(hand_pos=cur_hand_pos, hand_quat=cur_hand_rot, hand_dof=delta_hand_dof/step_number, env_ids_all=obj_ids)
                _, _, cur_hand_dof = self.get_handa_states(obj_ids)
        elif obj_no_move:
            while (torch.sum(abs(target_hand_dof - cur_hand_dof),-1) > 1e-3).any() and step < step_number:
                delta_hand_dof = target_hand_dof - cur_hand_dof
                delta_hand_dof = torch.clip(delta_hand_dof,-0.025,0.025)
                move_obj_ids = (torch.mean(abs(self.object_pos - init_obj_pos),-1) < 1e-3).nonzero(as_tuple=False).squeeze(-1)
                self.step_hand(hand_pos=cur_hand_pos[move_obj_ids], hand_quat=cur_hand_rot[move_obj_ids], hand_dof=delta_hand_dof[move_obj_ids], env_ids_all=move_obj_ids)
                _, _, cur_hand_dof = self.get_handa_states(obj_ids)
                step += 1
                print(step)
        else:
            while (torch.sum(abs(target_hand_dof - cur_hand_dof),-1) > 1e-3).any() and step < step_number:
                delta_hand_dof = target_hand_dof - cur_hand_dof
                delta_hand_dof = torch.clip(delta_hand_dof,-0.025,0.025)
                self.step_hand(hand_pos=cur_hand_pos, hand_quat=cur_hand_rot, hand_dof=delta_hand_dof, env_ids_all=obj_ids)
                _, _, cur_hand_dof = self.get_handa_states(obj_ids)
                step += 1
                print(step)
        
        return 1
    
    def move_hand_to_target_pose(self, obj_ids, target_hand_pose, threshold=100, direct=True):
        target_hand_pos = target_hand_pose[:,:3]
        target_hand_quat = target_hand_pose[:,3:7]
        cur_hand_pos, cur_hand_rot, cur_hand_dof = self.get_handa_states(obj_ids)
        step = 0
        if direct:
            while ((torch.sum(abs(target_hand_pos - cur_hand_pos),-1) > 1e-3).any() or (torch.sum(abs(target_hand_quat - cur_hand_rot),-1) > 1e-3).any()) and (step < threshold):
                self.step_hand(hand_pos=target_hand_pos, hand_quat=target_hand_quat, hand_dof=cur_hand_dof, env_ids_all=obj_ids)
                cur_hand_pos, cur_hand_rot, _= self.get_handa_states(obj_ids)
                step += 1
        else:
            init_hand_pos = cur_hand_pos.clone()
            while ((torch.sum(abs(target_hand_pos - cur_hand_pos),-1) > 1e-3).any() or (torch.sum(abs(target_hand_quat - cur_hand_rot),-1) > 1e-3).any()) and (step < threshold):
                self.step_hand(hand_pos=(target_hand_pos-init_hand_pos)*step/(threshold-2)+init_hand_pos, hand_quat=target_hand_quat, hand_dof=cur_hand_dof, env_ids_all=obj_ids)
                cur_hand_pos, cur_hand_rot, _= self.get_handa_states(obj_ids)
                step += 1
        
        self.step_simulation(10)
        return 1

    #####################################################################
    ###=========================gf functions=========================###
    #####################################################################
    def set_object_scale(self, env_ids, obj_scales):
        if allow_obj_scale:
            for (i, env_id) in enumerate(env_ids):
                self.gym.set_actor_scale(self.envs[env_id],1,obj_scales[i])
                # recover obj pcl
                self.obj_pcl_buf[env_id] = self.obj_pcl_buf[env_id]/self.cur_obj_scales[env_id]

                self.cur_obj_scales[env_id] = obj_scales[i]
                self.obj_pcl_buf[env_id] = self.obj_pcl_buf[env_id] * self.cur_obj_scales[env_id]

    def reset_gf_env(self, states=None, is_random=True, refresh=True, step_simulation_step=None):       
        if states is not None:
            hand_dof = states[:,:18].to(self.device).float()
            if is_random:
                hand_dof = np.random.uniform(self.shadow_hand_dof_lower_limits[self.actuated_dof_indices], self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
            else:
                hand_dof = scale(hand_dof,self.shadow_hand_dof_lower_limits[self.actuated_dof_indices], self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
            hand_pos = states[:,18:21].to(self.device).float()
            hand_rot = states[:,21:25].to(self.device).float()
            if states.size(1) > 3000:
                point_cloud_idx = 25 + self.points_per_object * 3
                obj_pos = states[:,point_cloud_idx:point_cloud_idx+3].to(self.device).float()
                obj_rot = states[:,point_cloud_idx+3:point_cloud_idx+7].to(self.device).float()
                env_ids = states[:,point_cloud_idx+7:point_cloud_idx+8].to(self.device).long().squeeze(-1)
                obj_scales = states[:,point_cloud_idx+23:point_cloud_idx+24].to(self.device).squeeze(-1)
            else:
                obj_pos = states[:,25:28].to(self.device).float()
                obj_rot = states[:,28:32].to(self.device).float()
                env_ids = states[:,32:33].to(self.device).long().squeeze(-1)
                obj_scales = states[:,33+15:33+16].to(self.device).squeeze(-1)
        elif states is None and is_random:
            env_ids = torch.arange(start=0, end=self.num_envs, device=self.device, dtype=torch.long)
            ## random dof state
            hand_dof = np.random.uniform(self.shadow_hand_dof_lower_limits[self.actuated_dof_indices], self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
            hand_dof = to_torch(hand_dof, device=self.device)
            hand_pos = torch_rand_float(-1.0, 1.0, (len(env_ids), 3), device=self.device)
            rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), 4), device=self.device)
            hand_rot = randomize_rotation(rand_floats[:, 0], rand_floats[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids])
            obj_pos = torch_rand_float(-1.0, 1.0, (len(env_ids), 3), device=self.device)
            rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), 4), device=self.device)
            obj_rot = randomize_rotation(rand_floats[:, 0], rand_floats[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids])
        
        if self.dataset_type=='dexgraspnet' and allow_obj_scale:
            self.set_object_scale(env_ids=env_ids, obj_scales=obj_scales)
                
        # test grasp label
        for (i, env_id) in enumerate(env_ids):
            # reset object state
            self.root_state_tensor[self.object_indices[env_id]][:3] = to_torch(obj_pos[i],device=self.device)
            self.root_state_tensor[self.object_indices[env_id]][3:7] = to_torch(obj_rot[i],device=self.device)
            
            ## reset shadow hand
            self.hand_new_dof_state = self.cur_targets.clone()
            new_hand_pos, new_hand_rot, self.hand_new_dof_state[env_id][self.actuated_dof_indices] = hand_pos[i], hand_rot[i], hand_dof[i]
            if self.fake_tendon:
                # set underactuated dof
                self.set_underactuated_dof(env_ids=env_id, target='hand_new_dof_state')

            if self.has_base:
                new_hand_euler = self.get_hand_euler_from_quat(new_hand_rot)
                self.hand_new_dof_state[env_id][self.hand_base_dof_indices] = torch.cat([new_hand_pos.unsqueeze(0), new_hand_euler],-1)
            else:
                # reset hand root state
                self.root_state_tensor[self.hand_indices[env_id], :3] = new_hand_pos
                self.root_state_tensor[self.hand_indices[env_id], 3:7] = new_hand_rot

            # set dof state
            self.shadow_hand_dof_pos[env_id, :] = self.hand_new_dof_state[env_id]
            self.shadow_hand_dof_vel[env_id, :] = self.shadow_hand_dof_default_vel
            self.prev_targets[env_id, :] = self.hand_new_dof_state[env_id]
            self.cur_targets[env_id, :] = self.hand_new_dof_state[env_id]

        # self.reset_target_pose(env_ids)

        self.deploy_to_environments(env_ids=env_ids,goal_env_ids=env_ids, step_simulation_step=step_simulation_step)

        self.reset_obj_vel(env_ids=env_ids)

        self.rb_forces[env_ids, :, :] = 0.0
        self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.rb_forces), None, gymapi.LOCAL_SPACE)

    def visualize_gf(self, action, env_id):
        action = to_torch(action, device=self.device)
        env_ids = torch.arange(start=env_id, end=env_id+1, device=self.device, dtype=torch.long)
        # test grasp label
        for env_id in env_ids:
            ## reset shadow hand
            self.hand_new_dof_state[env_id][self.actuated_dof_indices] = scale(action,self.shadow_hand_dof_lower_limits[self.actuated_dof_indices], self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
            if self.fake_tendon:
                # set underactuated dof
                self.set_underactuated_dof(env_ids=env_id, target='hand_new_dof_state')
            # set dof state
            self.shadow_hand_dof_pos[env_id, :] = self.hand_new_dof_state[env_id]
            self.shadow_hand_dof_vel[env_id, :] = self.shadow_hand_dof_default_vel
            self.prev_targets[env_id, :] = self.hand_new_dof_state[env_id]
            self.cur_targets[env_id, :] = self.hand_new_dof_state[env_id]

        hand_indices = self.hand_indices[env_ids].to(torch.int32)
        
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.prev_targets),
                                                        gymtorch.unwrap_tensor(hand_indices), len(env_ids))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(hand_indices), len(env_ids))

        self.refresh_env_states()
        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)

######################################################################################################
###=====================================global functions===========================================###
######################################################################################################

#####################################################################
###=========================jit functions=========================###
#####################################################################
def randomize_hand_pose(rand_floats, rand_rad):
    def_rad = 0.3
    rad = def_rad * (0.95 + rand_rad.squeeze() * 0.1)
    
    rand_pos = rand_floats.clone()

    roll = np.pi * (2*rand_floats[:,0]-1)
    yaw =  np.pi * (2*rand_floats[:,1]-1)
    pitch = np.pi * (2*rand_floats[:,2]-1)

    # roll[:] = 0
    # pitch[:] = -np.pi/2

    # roll[:] = np.pi*3/2
    # pitch[:] = -np.pi/2

    rand_quat = quat_from_euler_xyz(roll,pitch,yaw)

    # rand_euler[:,0] = roll
    # rand_euler[:,1] = yaw
    # rand_euler[:,2] = pitch

    rand_pos[:,0] = rad * (2*rand_floats[:,0]-1)
    rand_pos[:,1] = rad * (2*rand_floats[:,1]-1)
    rand_pos[:,2] = rad * rand_floats[:,2]

    return rand_pos, rand_quat

def torch_random_sample(lower, upper, shape, device):
    return (upper - lower) * torch.rand(*shape, device=device) + lower

@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(quat_from_angle_axis(rand0 * np.pi, x_unit_tensor),
                    quat_from_angle_axis(rand1 * np.pi, y_unit_tensor))

@torch.jit.script
def randomize_rotation_xyz(rand0, rand1, rand2, x_unit_tensor, y_unit_tensor, z_unit_tensor):
    xy = quat_mul(quat_from_angle_axis(rand0 * np.pi, x_unit_tensor),
                    quat_from_angle_axis(rand1 * np.pi, y_unit_tensor))
    xyz = quat_mul(xy,
                    quat_from_angle_axis(rand2 * np.pi, z_unit_tensor))
    return xyz

@torch.jit.script
def randomize_rotation_pen(rand0, rand1, max_angle, x_unit_tensor, y_unit_tensor, z_unit_tensor):
    rot = quat_mul(quat_from_angle_axis(0.5 * np.pi + rand0 * max_angle, x_unit_tensor),
                   quat_from_angle_axis(rand0 * np.pi, z_unit_tensor))
    return rot

@torch.jit.script
def quat_mul_point(quat, pt_input):
    quat_con = quat_conjugate(quat)
    pt_new = quat_mul(quat_mul(quat, pt_input), quat_con)
    return pt_new[:3]

@torch.jit.script
def quat_mul_point_tensor(quat, pt_input):
    quat_con = quat_conjugate(quat)
    pt_new = quat_mul(quat_mul(quat, pt_input), quat_con)
    return pt_new[:,:3]

@torch.jit.script
def transform_points(quat, pt_input):
    quat_con = quat_conjugate(quat)
    pt_new = quat_mul(quat_mul(quat, pt_input), quat_con)
    if len(pt_new.size()) == 3:
        return pt_new[:,:,:3]
    elif len(pt_new.size()) == 2:
        return pt_new[:,:3]

@torch.jit.script
def transform_world2target(quat, pt_input):
    # padding point to 4 dim
    B = pt_input.size(0)
    padding = torch.zeros([B,1]).to(pt_input.device)
    pt_input = torch.cat([pt_input,padding],1)

    quat_inverse = quat_conjugate(quat)
    pt_new = -quat_mul_point_tensor(quat_inverse, pt_input)
    return quat_inverse , pt_new

# @torch.jit.script
def multiply_transform(s_pos, s_quat, t_pos, t_quat):
    t2s_quat = quat_mul(s_quat, t_quat)

    B = t_pos.size()[0]
    padding = torch.zeros([B, 1]).to(t_pos.device)
    t_pos_pad = torch.cat([t_pos,padding],-1)
    s_quat = s_quat.expand_as(t_quat)
    t2s_pos = transform_points(s_quat, t_pos_pad)
    s_pos = s_pos.expand_as(t2s_pos)
    t2s_pos += s_pos
    return t2s_pos, t2s_quat

def quat_axis(q, axis=0):
    basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
    basis_vec[:, axis] = 1
    return quat_rotate(q, basis_vec)