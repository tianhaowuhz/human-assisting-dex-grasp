import numpy as np

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

from Networks.pointnet2.pointnet2_backbone import Pointnet2Backbone
from Networks.pointnet import PointNetEncoder
from ipdb import set_trace

local = False

class ActorCritic(nn.Module):

    def __init__(self, obs_shape, states_shape, actions_shape, initial_std, model_cfg, asymmetric=False, state_base=False, stack_frame_number=3, sub_obs_type=None, num_fingertip=None, pointnet_type='pt2', envs=None, hand_pcl=False, hand_model=None, args=None):
        super(ActorCritic, self).__init__()

        # network parameter
        self.asymmetric = asymmetric
        self.state_base = state_base
        self.stack_frame_number = stack_frame_number
        self.sub_obs_type = sub_obs_type
        self.num_fingertip = num_fingertip
        self.disentangle_hand = model_cfg['distengle']
        self.norm_action = model_cfg['norm_action']
        self.action_scale = model_cfg['action_scale']
        self.pointnet_type = pointnet_type
        self.envs = envs
        self.hand_pcl = hand_pcl
        self.hand_model = hand_model
        
        '''
        init network: current we set self.state_base = False, only set true for pure state input
        '''
        if not self.state_base:
            # get model cfg
            if model_cfg is None:
                self.hand_joint_dim = 18
                self.hand_wrist_dim = 7 * self.stack_frame_number
                actor_hidden_dim = 256
                critic_hidden_dim = 256
                activation = get_activation("selu")
                self.shared_pointnet = True
                self.points_per_object = 1024
            else:
                # get input dim
                self.hand_joint_dim = model_cfg['hand_joint_dim']
                self.hand_wrist_dim = model_cfg['hand_wrist_dim'] * self.stack_frame_number

                # fingertip obs dim
                if "fingertipjoint" in self.sub_obs_type:
                    self.fingertip_dim = self.num_fingertip-1
                else:
                    self.fingertip_dim = 0

                if "disfingertip" in self.sub_obs_type:
                    self.fingertip_dim += self.num_fingertip*1
                elif "absfingertip" in self.sub_obs_type:
                    self.fingertip_dim += self.num_fingertip*3

                # obj pose obs dim
                if "objpose" in self.sub_obs_type:
                    self.objpose_dim = 7
                else:
                    self.objpose_dim = 0
                
                # diso2o obs dim
                if "diso2o" in self.sub_obs_type:
                    self.diso2o_dim = 1
                else:
                    self.diso2o_dim = 0
                
                # goal obs dim
                if "goal" in self.sub_obs_type:
                    self.goal_dim = 18
                else:
                    self.goal_dim = 0
                
                # gf obs dim
                if 'gf' in self.sub_obs_type:
                    self.gf_dim = actions_shape[0]
                else:
                    self.gf_dim = 0

                # network parameter
                actor_hidden_dim = model_cfg['pi_hid_sizes']
                critic_hidden_dim = model_cfg['vf_hid_sizes']
                activation = get_activation(model_cfg['activation'])
                self.shared_pointnet = model_cfg['shared_pointnet']
                self.points_per_object = model_cfg['points_per_object']

            self.action_dim = actions_shape[0]

            '''
            actor layer
            '''
            # state encoder
            if self.disentangle_hand:
                self.actor_hand_joint_global_enc = nn.Sequential(
                        nn.Linear(self.hand_joint_dim + self.fingertip_dim + self.objpose_dim + self.diso2o_dim + self.goal_dim, actor_hidden_dim),
                        activation,
                    )
                
                self.actor_hand_wrist_global_enc = nn.Sequential(
                        nn.Linear(self.hand_wrist_dim, actor_hidden_dim),
                        activation,
                    )
                
                if 'gf' in self.sub_obs_type:
                    self.actor_grad_enc = nn.Sequential(
                        nn.Linear(*actions_shape, actor_hidden_dim),
                        activation,
                    )
            else:
                self.state_dim = self.hand_joint_dim + self.hand_wrist_dim + self.fingertip_dim + self.objpose_dim + self.diso2o_dim + self.goal_dim + self.gf_dim
                self.actor_hand_global_enc = nn.Sequential(
                        nn.Linear(self.state_dim, actor_hidden_dim),
                        activation,
                        nn.Linear(actor_hidden_dim, actor_hidden_dim),
                        activation,
                        nn.Linear(actor_hidden_dim, actor_hidden_dim),
                        activation,
                    )

            # pointcloud feature encoder
            self.actor_obj_global_enc = nn.Sequential(
                nn.Linear(self.points_per_object, actor_hidden_dim),
                activation,
            )

            # mlp output
            if self.disentangle_hand:
                if 'gf' in self.sub_obs_type:
                    total_feat_num = 2 + 1 + 1
                else:
                    total_feat_num = 2 + 1
            else:
                total_feat_num = 1 + 1

            if self.disentangle_hand:
                self.actor_mlp1 = nn.Sequential(
                    nn.Linear(actor_hidden_dim*total_feat_num, actor_hidden_dim),
                    activation,
                )
            else:
                self.actor_mlp1 = nn.Sequential(
                    nn.Linear(actor_hidden_dim*total_feat_num, actor_hidden_dim),
                    activation,
                    nn.Linear(actor_hidden_dim, actor_hidden_dim),
                    activation,
                )

            # norm output action
            if self.norm_action:
                self.actor_mlp2 = nn.Sequential(
                    nn.Linear(actor_hidden_dim, *actions_shape),
                    get_activation("tanh"),
                )
            else:
                self.actor_mlp2 = nn.Sequential(
                    nn.Linear(actor_hidden_dim, *actions_shape),
                )

            '''
            critic layer
            '''
            # state encoder
            if self.disentangle_hand:
                self.critic_hand_joint_global_enc = nn.Sequential(
                        nn.Linear(self.hand_joint_dim + self.fingertip_dim + self.objpose_dim + self.diso2o_dim + self.goal_dim, critic_hidden_dim),
                        activation,
                    )
                
                self.critic_hand_wrist_global_enc = nn.Sequential(
                        nn.Linear(self.hand_wrist_dim, critic_hidden_dim),
                        activation,
                    )
                
                if 'gf' in self.sub_obs_type:
                    self.critic_grad_enc = nn.Sequential(
                        nn.Linear(*actions_shape, critic_hidden_dim),
                        activation,
                    )
            else:
                self.state_dim = self.hand_joint_dim + self.hand_wrist_dim + self.fingertip_dim + self.objpose_dim + self.diso2o_dim + self.goal_dim + self.gf_dim
                self.critic_hand_global_enc = nn.Sequential(
                        nn.Linear(self.state_dim, critic_hidden_dim),
                        activation,
                        nn.Linear(critic_hidden_dim, critic_hidden_dim),
                        activation,
                        nn.Linear(critic_hidden_dim, critic_hidden_dim),
                        activation,
                    )

            # pointcloud feature encoder
            self.critic_obj_global_enc = nn.Sequential(
                nn.Linear(self.points_per_object, critic_hidden_dim),
                activation,
            )

            # mlp output
            if self.disentangle_hand:
                self.critic_mlp1 = nn.Sequential(
                    nn.Linear(critic_hidden_dim*total_feat_num, critic_hidden_dim),
                    activation,
                )

                if args.exp_name == 'ilad':
                    self.additional_critic_mlp1 = nn.Sequential(
                        nn.Linear(critic_hidden_dim*total_feat_num + self.action_dim, critic_hidden_dim),
                        activation,
                    )
            else:
                self.critic_mlp1 = nn.Sequential(
                    nn.Linear(critic_hidden_dim*total_feat_num, critic_hidden_dim),
                    activation,
                    nn.Linear(critic_hidden_dim, critic_hidden_dim),
                    activation,
                )

                if args.exp_name == 'ilad':
                    self.additional_critic_mlp1 = nn.Sequential(
                        nn.Linear(critic_hidden_dim*total_feat_num + self.action_dim, critic_hidden_dim),
                        activation,
                        nn.Linear(critic_hidden_dim, 1),
                    )
            self.critic_mlp2 = nn.Sequential(
                nn.Linear(critic_hidden_dim, 1),
            )

            '''
            shared layer
            '''
            if self.shared_pointnet:
                if self.pointnet_type == 'pt':
                    self.pointnet_enc = PointNetEncoder()
                elif self.pointnet_type == 'pt2':
                    self.pointnet_enc = Pointnet2Backbone() # for pointnet2
            else:
                if self.pointnet_type == 'pt':
                    self.actor_pointnet_enc = PointNetEncoder()
                    self.critic_pointnet_enc = PointNetEncoder()
                elif self.pointnet_type == 'pt2':
                    self.actor_pointnet_enc = Pointnet2Backbone() # for pointnet2
                    self.critic_pointnet_enc = Pointnet2Backbone() # for pointnet2

            # Action noise
            self.log_std = nn.Parameter(np.log(initial_std) * torch.ones(*actions_shape))
        else:
            # get model config
            if model_cfg is None:
                actor_hidden_dim = [256, 256, 256]
                critic_hidden_dim = [256, 256, 256]
                activation = get_activation("selu")
            else:
                if local:
                    actor_hidden_dim = [256, 256, 256]
                    critic_hidden_dim = [256, 256, 256]
                    activation = get_activation("selu")
                else:
                    actor_hidden_dim = model_cfg['pi_hid_sizes']
                    critic_hidden_dim = model_cfg['vf_hid_sizes']
                    activation = get_activation(model_cfg['activation'])

            # Policy
            actor_layers = []
            actor_layers.append(nn.Linear(*obs_shape, actor_hidden_dim[0]))
            actor_layers.append(activation)
            for l in range(len(actor_hidden_dim)):
                if l == len(actor_hidden_dim) - 1:
                    actor_layers.append(nn.Linear(actor_hidden_dim[l], *actions_shape))
                else:
                    actor_layers.append(nn.Linear(actor_hidden_dim[l], actor_hidden_dim[l + 1]))
                    actor_layers.append(activation)
            self.actor = nn.Sequential(*actor_layers)

            # Value function
            critic_layers = []
            if self.asymmetric:
                critic_layers.append(nn.Linear(*states_shape, critic_hidden_dim[0]))
            else:
                critic_layers.append(nn.Linear(*obs_shape, critic_hidden_dim[0]))
            critic_layers.append(activation)
            for l in range(len(critic_hidden_dim)):
                if l == len(critic_hidden_dim) - 1:
                    critic_layers.append(nn.Linear(critic_hidden_dim[l], 1))
                else:
                    critic_layers.append(nn.Linear(critic_hidden_dim[l], critic_hidden_dim[l + 1]))
                    critic_layers.append(activation)
            self.critic = nn.Sequential(*critic_layers)

            print(self.actor)
            print(self.critic)

            # Action noise
            self.log_std = nn.Parameter(np.log(initial_std) * torch.ones(*actions_shape))

            # Initialize the weights like in stable baselines
            actor_weights = [np.sqrt(2)] * len(actor_hidden_dim)
            actor_weights.append(0.01)
            critic_weights = [np.sqrt(2)] * len(critic_hidden_dim)
            critic_weights.append(1.0)
            self.init_weights(self.actor, actor_weights)
            self.init_weights(self.critic, critic_weights)
    
    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def forward(self):
        raise NotImplementedError
    
    def forward_actor(self, observations):
        '''
        process observation
        '''
        batch_size = observations.size(0)
        
        if self.disentangle_hand:
            hand_joint_batch, hand_wrist_batch, grad_batch, obj_batch = self.process_observations(observations=observations)
        else:
            state_batch, obj_batch = self.process_observations(observations=observations)
  
        '''
        forward
        '''
        # pointcloud encoder
        if self.shared_pointnet:
            if self.pointnet_type == 'pt':
                obj_feat, _, _ = self.pointnet_enc(obj_batch.reshape(batch_size,3,-1))
            elif self.pointnet_type == 'pt2':
                obj_feat, _ = self.pointnet_enc(obj_batch.reshape(batch_size,-1,3))
        else:
            if self.pointnet_type == 'pt':
                obj_feat, _, _ = self.actor_pointnet_enc(obj_batch.reshape(batch_size,3,-1))
            elif self.pointnet_type == 'pt2':
                obj_feat, _ = self.actor_pointnet_enc(obj_batch.reshape(batch_size,-1,3))
        obj_feat = self.actor_obj_global_enc(obj_feat.reshape(batch_size,-1)) # B x 512

        # state encoder
        if self.disentangle_hand:
            hand_joint_global_feat = self.actor_hand_joint_global_enc(hand_joint_batch) # B x 512
            hand_wrist_global_feat = self.actor_hand_wrist_global_enc(hand_wrist_batch) # B x 512
            hand_global_feat = torch.cat([hand_wrist_global_feat, hand_joint_global_feat],-1)                

            if 'gf' in self.sub_obs_type:
                grad_feat = self.actor_grad_enc(grad_batch) # B x 512
                total_feat = torch.cat([hand_global_feat, obj_feat, grad_feat],-1)
            else:
                total_feat = torch.cat([hand_global_feat, obj_feat],-1)
        else:
            hand_global_feat = self.actor_hand_global_enc(state_batch)
            total_feat = torch.cat([hand_global_feat, obj_feat],-1)

        # mlp
        x = self.actor_mlp1(total_feat)
        x = self.actor_mlp2(x)*self.action_scale
        return x
    
    def forward_critic(self, observations):
        """
        process observation
        """
        batch_size = observations.size(0)

        if self.disentangle_hand:
            hand_joint_batch, hand_wrist_batch, grad_batch, obj_batch = self.process_observations(observations=observations)
        else:
            state_batch, obj_batch = self.process_observations(observations=observations)

        '''
        forward
        '''
        # point cloud encoder
        if self.shared_pointnet:
            if self.pointnet_type == 'pt':
                obj_feat, _, _ = self.pointnet_enc(obj_batch.reshape(batch_size,3,-1))
            elif self.pointnet_type == 'pt2':
                obj_feat, _ = self.pointnet_enc(obj_batch.reshape(batch_size,-1,3))
        else:
            if self.pointnet_type == 'pt':
                obj_feat, _, _ = self.critic_pointnet_enc(obj_batch.reshape(batch_size,3,-1))
            elif self.pointnet_type == 'pt2':
                obj_feat, _ = self.critic_pointnet_enc(obj_batch.reshape(batch_size,-1,3))
        obj_feat = self.critic_obj_global_enc(obj_feat.reshape(batch_size,-1)) # B x 512
        
        # state encoder
        if self.disentangle_hand:
            hand_joint_global_feat = self.critic_hand_joint_global_enc(hand_joint_batch) # B x 512
            hand_wrist_global_feat = self.critic_hand_wrist_global_enc(hand_wrist_batch) # B x 512
            hand_global_feat = torch.cat([hand_wrist_global_feat, hand_joint_global_feat],-1)

            if 'gf' in self.sub_obs_type:
                grad_feat = self.critic_grad_enc(grad_batch) # B x 512
                total_feat = torch.cat([hand_global_feat, obj_feat, grad_feat],-1)
            else:
                total_feat = torch.cat([hand_global_feat, obj_feat],-1)
        else:
            hand_global_feat = self.critic_hand_global_enc(state_batch)
            
            total_feat = torch.cat([hand_global_feat, obj_feat],-1)
        
        # mlp
        x = self.critic_mlp1(total_feat)
        x = self.critic_mlp2(x)
        return x

    def forward_additional_critic(self, observations, actions):
        """
        process observation
        """
        batch_size = observations.size(0)

        if self.disentangle_hand:
            hand_joint_batch, hand_wrist_batch, grad_batch, obj_batch = self.process_observations(observations=observations)
        else:
            state_batch, obj_batch = self.process_observations(observations=observations)

        '''
        forward
        '''
        # point cloud encoder
        if self.shared_pointnet:
            if self.pointnet_type == 'pt':
                obj_feat, _, _ = self.pointnet_enc(obj_batch.reshape(batch_size,3,-1))
            elif self.pointnet_type == 'pt2':
                obj_feat, _ = self.pointnet_enc(obj_batch.reshape(batch_size,-1,3))
        else:
            if self.pointnet_type == 'pt':
                obj_feat, _, _ = self.critic_pointnet_enc(obj_batch.reshape(batch_size,3,-1))
            elif self.pointnet_type == 'pt2':
                obj_feat, _ = self.critic_pointnet_enc(obj_batch.reshape(batch_size,-1,3))
        obj_feat = self.critic_obj_global_enc(obj_feat.reshape(batch_size,-1)) # B x 512
        
        # state encoder
        if self.disentangle_hand:
            hand_joint_global_feat = self.critic_hand_joint_global_enc(hand_joint_batch) # B x 512
            hand_wrist_global_feat = self.critic_hand_wrist_global_enc(hand_wrist_batch) # B x 512
            hand_global_feat = torch.cat([hand_wrist_global_feat, hand_joint_global_feat],-1)

            if 'gf' in self.sub_obs_type:
                grad_feat = self.critic_grad_enc(grad_batch) # B x 512
                total_feat = torch.cat([hand_global_feat, obj_feat, grad_feat],-1)
            else:
                total_feat = torch.cat([hand_global_feat, obj_feat],-1)
        else:
            hand_global_feat = self.critic_hand_global_enc(state_batch)
            
            total_feat = torch.cat([hand_global_feat, obj_feat],-1)
        
        # mlp
        total_feat = torch.concat([total_feat, actions], -1)
        x = self.additional_critic_mlp1(total_feat)
        return x

    def process_observations(self,observations):
        '''
        get all obs batch
        '''
        hand_joint_batch = observations[:,:self.hand_joint_dim] # B x 18
        hand_wrist_batch = observations[:,self.hand_joint_dim:self.hand_joint_dim+self.hand_wrist_dim] # B x 7 * sfn
        fingertip_idx = self.hand_joint_dim+self.hand_wrist_dim+self.points_per_object*3
        obj_batch = observations[:,self.hand_joint_dim+self.hand_wrist_dim:fingertip_idx] # B x 1024*3

        if self.hand_pcl:
            hand_pos_2_w = hand_wrist_batch[:,:3].clone()
            hand_quat_2_w = hand_wrist_batch[:,3:7].clone()
            hand_pos_2_h, hand_quat_2_h = self.envs.transform_target2source(hand_quat_2_w, hand_pos_2_w, hand_quat_2_w, hand_pos_2_w)

            ori_hand_dof = self.envs.dof_norm(hand_joint_batch.clone(),inv=True)
            hand_pcl_2h = self.hand_model.get_hand_pcl(hand_pos=hand_pos_2_h, hand_quat=hand_quat_2_h, hand_dof=ori_hand_dof)
            obj_batch = torch.cat([obj_batch, hand_pcl_2h.reshape(hand_pcl_2h.size(0),-1)],1)

        if "fingertipjoint" in self.sub_obs_type:
            fingertipjoint_batch = observations[:,fingertip_idx:fingertip_idx+self.num_fingertip-1]
            fingertip_idx = fingertip_idx+self.num_fingertip-1

            if "disfingertip" in self.sub_obs_type:
                fingertip_batch = observations[:,fingertip_idx:fingertip_idx+self.num_fingertip]
                objpose_idx = fingertip_idx+self.num_fingertip
                fingertip_batch = torch.cat([fingertipjoint_batch,fingertip_batch],-1)
            elif "absfingertip" in self.sub_obs_type:
                fingertip_batch = observations[:,fingertip_idx:fingertip_idx+self.num_fingertip*3]
                objpose_idx = fingertip_idx+self.num_fingertip*3
                fingertip_batch = torch.cat([fingertipjoint_batch,fingertip_batch],-1)
            else:
                objpose_idx = fingertip_idx          
                fingertip_batch = fingertipjoint_batch
        else:
            objpose_idx = fingertip_idx          

        if "objpose" in self.sub_obs_type:
            objpose_batch = observations[:,objpose_idx:objpose_idx+7]
            diso2o_idx = objpose_idx+7
        else:
            diso2o_idx = objpose_idx

        if "diso2o" in self.sub_obs_type:
            diso2o_batch = observations[:,diso2o_idx:diso2o_idx+1]
            goal_idx = diso2o_idx + 1
        else:
            goal_idx = diso2o_idx

        if "goal" in self.sub_obs_type:
            goal_batch = observations[:,goal_idx:goal_idx+18]

        if 'gf' in self.sub_obs_type:
            grad_batch = observations[:,-self.action_dim:] # B x 18
        else:
            grad_batch = None

        if self.disentangle_hand:
            if "fingertip" in self.sub_obs_type:
                hand_joint_batch = torch.cat([hand_joint_batch, fingertip_batch], -1)
            if "objpose" in self.sub_obs_type:
                hand_joint_batch = torch.cat([hand_joint_batch, objpose_batch], -1)
            if "diso2o" in self.sub_obs_type:
                hand_joint_batch = torch.cat([hand_joint_batch, diso2o_batch], -1)
            if "goal" in self.sub_obs_type:
                hand_joint_batch = torch.cat([hand_joint_batch, goal_batch], -1)
            
            return hand_joint_batch, hand_wrist_batch, grad_batch, obj_batch
        else:
            state_batch = torch.cat([hand_wrist_batch,hand_joint_batch],-1)
            if "fingertip" in self.sub_obs_type:
                state_batch = torch.cat([state_batch, fingertip_batch],-1)
            
            if "objpose" in self.sub_obs_type:
                state_batch = torch.cat([state_batch, objpose_batch], -1)
            
            if "diso2o" in self.sub_obs_type:
                state_batch = torch.cat([state_batch, diso2o_batch], -1)
            
            if "goal" in self.sub_obs_type:
                state_batch = torch.cat([state_batch, goal_batch],-1)

            if 'gf' in self.sub_obs_type:
                state_batch = torch.cat([state_batch, grad_batch], -1)
            
            return state_batch, obj_batch
        
    def act(self, observations, states):
        if self.state_base:
            actions_mean = self.actor(observations)
        else:
            actions_mean = self.forward_actor(observations)

        # print(self.log_std)
        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions = distribution.sample()
        actions_log_prob = distribution.log_prob(actions)

        if self.asymmetric:
            value = self.critic(states)
        else:
            if self.state_base:
                value = self.critic(observations)
            else:
                value = self.forward_critic(observations)

        return actions.detach(), actions_log_prob.detach(), value.detach(), actions_mean.detach(), self.log_std.repeat(actions_mean.shape[0], 1).detach()

    def cal_actions_log_prob(self,observations, actions):
        if self.state_base:
            actions_mean = self.actor(observations)
        else:
            actions_mean = self.forward_actor(observations)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions_log_prob = distribution.log_prob(actions)
        return actions.detach(), actions_log_prob.detach(), actions_mean.detach()
        
    def act_inference(self, observations):
        if self.state_base:
            actions_mean = self.actor(observations)
        else:
            actions_mean = self.forward_actor(observations)
        return actions_mean

    def evaluate(self, observations, states, actions):
        if self.state_base:
            actions_mean = self.actor(observations)
        else:
            actions_mean = self.forward_actor(observations)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions_log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()

        if self.asymmetric:
            value = self.critic(states)
        else:
            if self.state_base:
                value = self.critic(observations)
            else:
                value = self.forward_critic(observations)

        return actions_log_prob, entropy, value, actions_mean, self.log_std.repeat(actions_mean.shape[0], 1)

def get_activation(act_name):
    print(act_name)
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        set_trace()
        return None
