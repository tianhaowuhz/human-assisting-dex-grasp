from datetime import datetime
import os
import time
import functools

from gym.spaces import Space

import numpy as np
import statistics
from collections import deque
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from Algorithms.ppo import RolloutStorage
from Algorithms.ppo import ActorCritic

# gf part
from Algorithms.SDE_update import loss_fn_cond, cond_ode_sampler, init_sde
from Networks.SDENets_update import CondScoreModel

import copy
from tqdm import tqdm
from ipdb import set_trace
import time
import pickle
import cv2
import matplotlib.pyplot as plt
import io
import _pickle as CPickle

save_video = False
img_size = 256
save_state = False

def images_to_video(path, images, fps=10, size=(256,256), suffix='mp4'):
    path = path+f'.{suffix}'
    out = cv2.VideoWriter(filename=path, fourcc=cv2.VideoWriter_fourcc(*'mp4v'), fps=fps, frameSize=size, isColor=True)
    for item in images:
        out.write(item.astype(np.uint8))
    out.release()

def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

class GFPPO:
    def __init__(self,
                 vec_env,
                 cfg_train,
                 device='cpu',
                 sampler='sequential',
                 log_dir='run',
                 is_testing=False,
                 print_log=True,
                 apply_reset=False,
                 asymmetric=False,
                 args=None,
                 ):
        self.args = args
        ''' PPO '''
        # PPO parameters
        if not isinstance(vec_env.observation_space, Space):
            raise TypeError("vec_env.observation_space must be a gym Space")
        if not isinstance(vec_env.state_space, Space):
            raise TypeError("vec_env.state_space must be a gym Space")
        if not isinstance(vec_env.action_space, Space):
            raise TypeError("vec_env.action_space must be a gym Space")
        self.observation_space = vec_env.observation_space
        self.action_space = vec_env.action_space
        self.state_space = vec_env.state_space
        self.cfg_train = copy.deepcopy(cfg_train)
        learn_cfg = self.cfg_train["learn"]
        self.device = device
        self.asymmetric = asymmetric
        self.desired_kl = learn_cfg.get("desired_kl", None)
        self.schedule = learn_cfg.get("schedule", "fixed")
        self.step_size = learn_cfg["optim_stepsize"]
        self.init_noise_std = learn_cfg.get("init_noise_std", 0.3)
        self.model_cfg = self.cfg_train["policy"]
        self.num_transitions_per_env=learn_cfg["nsteps"]
        self.learning_rate=learn_cfg["optim_stepsize"]

        self.clip_param = learn_cfg["cliprange"]
        self.num_learning_epochs = learn_cfg["noptepochs"]
        self.num_mini_batches = learn_cfg["nminibatches"]
        self.value_loss_coef = learn_cfg.get("value_loss_coef", 2.0)
        self.entropy_coef = learn_cfg["ent_coef"]
        self.gamma = learn_cfg["gamma"]
        self.lam = learn_cfg["lam"]
        self.max_grad_norm = learn_cfg.get("max_grad_norm", 2.0)
        self.use_clipped_value_loss = learn_cfg.get("use_clipped_value_loss", False)

        # policy type 
        self.action_type = self.cfg_train["setting"]["action_type"]
        self.sub_action_type = self.cfg_train["setting"]["sub_action_type"]
        self.action_clip = self.cfg_train["setting"]["action_clip"]
        self.grad_process = self.cfg_train["setting"]["grad_process"]
        self.grad_scale = self.cfg_train["setting"]["grad_scale"]

        if self.action_type=='joint' and self.sub_action_type=='add+jointscale':
            action_space_shape = (18+18,)
        else:
            action_space_shape = self.action_space.shape
        print(f'action_space_shape:{action_space_shape}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        self.vec_env = vec_env
        self.vec_env.grad_scale = self.grad_scale
        
        pointnet_version = self.cfg_train["policy"]["pointnet_version"]

        hand_pcl = self.cfg_train["policy"]["hand_pcl"]
        hand_model = None

        # PPO components
        self.stack_frame_numer = self.vec_env.stack_frame_numbers
        self.actor_critic = ActorCritic(self.observation_space.shape, self.state_space.shape, action_space_shape,
                                        self.init_noise_std, self.model_cfg, asymmetric=asymmetric, stack_frame_number=self.stack_frame_numer, 
                                        sub_obs_type=self.vec_env.sub_obs_type, num_fingertip=self.vec_env.num_fingertips, pointnet_type=pointnet_version, 
                                        envs=self.vec_env, hand_pcl=hand_pcl, hand_model=hand_model, args=args)

        # pointnet backbone
        
        self.pointnet_finetune = self.model_cfg['finetune_pointnet']
        self.finetune_pointnet_bz = 128
        if self.model_cfg['pretrain_pointnet']:
            if pointnet_version == 'pt2':
                pointnet_model_dict = torch.load(os.path.join(args.score_model_path,'pt2.pt'), map_location=self.device)
            elif pointnet_version == 'pt':
                pointnet_model_dict = torch.load(os.path.join(args.score_model_path,'pt.pt'), map_location=self.device)
            if self.model_cfg['shared_pointnet']:
                self.actor_critic.pointnet_enc.load_state_dict(pointnet_model_dict)
                if not self.model_cfg['finetune_pointnet']:
                    # freeze pointnet
                    for name,param in self.actor_critic.pointnet_enc.named_parameters():
                        param.requires_grad = False
            else:
                self.actor_critic.actor_pointnet_enc.load_state_dict(pointnet_model_dict)
                self.actor_critic.critic_pointnet_enc.load_state_dict(pointnet_model_dict)

                if not self.model_cfg['finetune_pointnet']:
                    # freeze pointnet
                    for name,param in self.actor_critic.actor_pointnet_enc.named_parameters():
                        param.requires_grad = False
                    for name,param in self.actor_critic.critic_pointnet_enc.named_parameters():
                        param.requires_grad = False

        self.actor_critic.to(self.device)
        self.storage = RolloutStorage(self.vec_env.num_envs, self.num_transitions_per_env, self.observation_space.shape,
                                      self.state_space.shape, action_space_shape, self.device, sampler)
        
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.actor_critic.parameters()), lr=self.learning_rate)

        ''' SDE '''
        if 'gf' in self.vec_env.sub_obs_type:
            # init SDE config
            self.prior_fn, self.marginal_prob_fn, self.sde_fn = init_sde("vp")
            self.score = CondScoreModel(
                self.marginal_prob_fn,
                hidden_dim=args.hidden_dim,
                embed_dim=args.embed_dim,
                mode=args.score_mode,
                relative=args.relative,
                space=args.space,
                pointnet_version='pt2',
            )
            model_dict = torch.load(os.path.join(args.score_model_path,'score.pt'))
            self.score.load_state_dict(model_dict)
            self.score.to(device)
            self.score.eval()
            self.points_per_object = args.points_per_object
        self.t0 = args.t0
        self.ori_grad = None

        ''' Log '''
        # self.log_dir = log_dir
        if self.args.model_dir != "" and self.vec_env.mode=='train':
            time_now = self.args.model_dir.split('/')[8].split('_')[0] 
        else:
            time_now = time.strftime('%m-%d-%H-%M',time.localtime(time.time()))

        self.log_dir = os.path.join(f"./logs/{args.exp_name}/{time_now}_handrot:{self.vec_env.hand_rotation}_t0:{self.t0}_sfn:{self.vec_env.stack_frame_numbers}_{self.vec_env.num_envs}ne_{len(self.vec_env.shapes_all)}obj_gpt:{self.grad_process}_gs:{self.grad_scale}_at:{self.action_type}_subat:{self.sub_action_type}_rt:{self.vec_env.reward_type}_rn:{self.vec_env.reward_normalize}_simfreq:{self.vec_env.similarity_reward_freq}_cd:{self.vec_env.close_dis}_pts:{pointnet_version}_seed{args.seed}")
        self.print_log = print_log
        self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        self.tot_timesteps = 0
        self.tot_time = 0
        self.is_testing = is_testing
        self.current_learning_iteration = 0

        if save_video:
            self.video_log_dir = os.path.join(self.log_dir,'video')
            os.makedirs(self.video_log_dir,exist_ok=True)
            self.vis_env_num = self.args.vis_env_num

        self.apply_reset = apply_reset

        ''' Evaluation '''
        if  'gf_check' in self.action_type:
            self.eval_round = 20
        else:
            self.eval_round = 5

        if self.vec_env.mode == 'eval':
            self.eval_round = self.args.eval_times

        if save_state:
            self.eval_metrics = {
            'obj_shapes':[],
            'time_step':[],
            'success_rate':[],
            'gt_dist':[],
            'stability':[],
            'lift_nums':np.zeros(self.vec_env.num_envs),
            'gf_state_init':[],
            'gf_state_final':[],
            'gf_state_gt':[],
            }
        else:
            self.eval_metrics = {
            'obj_shapes':[],
            'time_step':[],
            'success_rate':[],
            'gt_dist':[],
            'stability':[],
            'lift_nums':np.zeros(self.vec_env.num_envs),
            'obj_translation':[],
            'obj_cosine_similarity':[],
            }
        self.eval_metrics['obj_shapes'] = self.vec_env.object_types

    def test(self, path):
        self.actor_critic.load_state_dict(torch.load(path, map_location=self.device))
        self.actor_critic.eval()

    def load(self, path):
        self.actor_critic.load_state_dict(torch.load(path, map_location=self.device))
        self.current_learning_iteration = int(path.split("_")[-1].split(".")[0])
        self.actor_critic.train()

        model_dir = path[:-len(path.split('/')[-1])] + f"metric_{self.args.exp_name}_{self.args.seed}.pkl"
        self.eval_metrics = CPickle.load(open(model_dir, 'rb'))

    def save(self, path):
        torch.save(self.actor_critic.state_dict(), path)
    
    def eval(self, it):
        # eval initilization
        self.vec_env.eval(vis=save_video)
        test_times = 0
        success_times = 0 # total_success_times / total_trials
        success_rates = [] # s_rate for each round
        reward_all = []
        if 'gf_check' in self.action_type:
            total_diff_direction_num = 0
            total_dof_error = 0
            diff_joint_num = torch.zeros(18,device=self.device)
        
        if self.vec_env.mode == 'train':
            save_time = 0 # means save all videos
        else:
            save_time = self.eval_round - 1

        # start evaluation
        with tqdm(total=self.eval_round) as pbar:
            pbar.set_description('Validating:')
            with torch.no_grad():
                for r in range(self.eval_round) :
                    if save_video and r<=save_time:
                        all_images = torch.tensor([],device=self.device)
                    # reset env
                    current_obs = self.vec_env.reset()['obs']
                    current_states = self.vec_env.get_state()
                    eval_done_envs = torch.zeros(self.vec_env.num_envs, dtype=torch.long, device=self.device)

                    if save_state:
                        self.eval_metrics['gf_state_init'].append(self.vec_env.get_states(gf_state=True))
                        self.eval_metrics['gf_state_gt'].append(self.vec_env.target_hand_dof)

                    # step
                    while True :
                        # Compute the action
                        actions, grad = self.compute_action(current_obs=current_obs,mode='eval')
                        # print(grad)
                        step_actions = self.process_actions(actions=actions.clone(), grad=grad)
                        # primitive_actions.append(torch.mean(grad).item())
                        # all_actions.append(torch.mean(step_actions).item())
                        if self.vec_env.progress_buf[0] == 49 and save_state:
                            self.eval_metrics['gf_state_final'].append(self.vec_env.get_states(gf_state=True))

                        # Step the vec_environment
                        next_obs, rews, dones, infos = self.vec_env.step(step_actions, (actions,grad))

                        if save_video and r<=save_time:
                            image = self.vec_env.render(rgb=True, img_size=img_size, vis_env_num=self.vis_env_num).reshape(self.vis_env_num, 1, img_size, img_size, 3)
                            all_images = torch.cat([all_images, image],1)
                        current_obs.copy_(next_obs['obs'])

                        # done
                        new_done_env_ids = (dones&(1-eval_done_envs)).nonzero(as_tuple=False).squeeze(-1)
                        if len(new_done_env_ids) > 0:
                            if self.vec_env.disable_collision:
                                print('-----------------------------------')
                                print('no coll succ:', infos['success_num'])
                                self.vec_env.grasp_filter(states=self.eval_metrics['gf_state_final'][r], test_time=1, reset_coll=True)
                            
                            self.eval_metrics['time_step'].append(it)
                            self.eval_metrics['success_rate'].append(float(infos['success_rate'].cpu().numpy()))
                            # self.eval_metrics['obj_translation'].append(float(infos['obj_translation'].cpu().numpy()))
                            # self.eval_metrics['obj_cosine_similarity'].append(float(infos['obj_cosine_similarity'].cpu().numpy()))
                            self.eval_metrics['gt_dist'].append(float(infos['gt_dist'].cpu().numpy()))
                            self.eval_metrics['lift_nums']+=infos['lift_nums'].cpu().numpy()
                            if self.vec_env.mode == 'eval':
                                with open(f'logs/{self.args.exp_name}/metrics_{self.args.eval_name}_eval_{self.args.seed}.pkl', 'wb') as f: 
                                    pickle.dump(self.eval_metrics, f)
                            else:
                                with open(os.path.join(self.log_dir, f'metric_{self.args.exp_name}_{self.args.seed}.pkl'), 'wb') as f: 
                                    pickle.dump(self.eval_metrics, f)

                            if 'gf_check' in self.action_type:
                                final_hand_dof = self.vec_env.final_hand_dof
                                target_hand_dof = self.vec_env.target_hand_dof
                                diff_direction_ids = ((self.vec_env.final_hand_dof * self.vec_env.target_hand_dof)<0).nonzero()      
                                same_direction_ids = ((self.vec_env.final_hand_dof * self.vec_env.target_hand_dof)>0).nonzero()   
                                for mm in range(18):
                                    diff_joint_num[mm] += torch.sum(diff_direction_ids[:,1]==mm)   
                                print(len(diff_direction_ids)/self.vec_env.num_envs)
                                print(diff_joint_num)
                                dof_error = torch.mean(abs(target_hand_dof[same_direction_ids[:,0],same_direction_ids[:,1]] - final_hand_dof[same_direction_ids[:,0],same_direction_ids[:,1]]))
                                print(dof_error)
                                total_diff_direction_num+=(len(diff_direction_ids)/self.vec_env.num_envs)
                                total_dof_error+=(dof_error)

                            if r > save_time:
                                self.vec_env.graphics_device_id = -1
                                self.vec_env.enable_camera_sensors = False

                            if save_video and r<=save_time:
                                for (i,images) in enumerate(all_images):
                                    obj_type = self.vec_env.object_type_per_env[i]
                                    save_path = os.path.join(self.video_log_dir,f'{obj_type}_epoach:{it}_round:{r}')
                                    images_to_video(path=save_path, images=images.cpu().numpy(), size=(img_size,img_size))

                            test_times += len(new_done_env_ids)
                            success_times += infos['success_num']
                            reward_all.extend(rews[new_done_env_ids].cpu().numpy())
                            eval_done_envs[new_done_env_ids] = 1
                            print(f'eval_success_rate: {success_times/test_times}')
                            success_rates.append(infos['success_num'] / len(new_done_env_ids))

                        if test_times==(r+1)*self.vec_env.num_envs:
                            break
                    pbar.update(1)
        if 'gf_check' in self.action_type:
            print(f'total_diff_direction_num:{total_diff_direction_num/self.eval_round}')
            print(f'total_dof_error:{total_dof_error/self.eval_round}')

        assert test_times==self.eval_round*self.vec_env.num_envs
        success_rates = torch.tensor(success_rates)
        sr_mu, sr_std = success_rates.mean().cpu().numpy().item(), success_rates.std().cpu().numpy().item()
        print(f'====== t0: {self.t0} || num_envs: {self.vec_env.num_envs} || eval_times: {self.eval_round}')
        print(f'eval_success_rate: {sr_mu:.2f} +- {sr_std:.2f}')
        eval_rews = np.mean(reward_all)
        print(f'eval_rewards: {eval_rews}')
        self.writer.add_scalar('Eval/success_rate', sr_mu, it)
        self.writer.add_scalar('Eval/eval_rews', eval_rews, it)

    def run(self, num_learning_iterations, log_interval=1):
        if self.is_testing:
            self.eval(0)
        else:
            # train initilization
            self.actor_critic.train()
            self.vec_env.train()
            rewbuffer = deque(maxlen=100)
            lenbuffer = deque(maxlen=100)
            cur_reward_sum = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)
            cur_episode_length = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)
            reward_sum = []
            episode_length = []

            # reset env
            current_obs = self.vec_env.reset()['obs']
            current_states = self.vec_env.get_state()
            for it in range(self.current_learning_iteration, num_learning_iterations):
                start = time.time()
                ep_infos = []
                if 'ori_similarity' in self.vec_env.reward_type:
                    ori_sim_all = []
                # Rollout
                for _ in range(self.num_transitions_per_env):
                    if self.apply_reset:
                        current_obs = self.vec_env.reset()['obs']
                        current_states = self.vec_env.get_state()

                    # Compute the action
                    actions, actions_log_prob, values, mu, sigma, grad = self.compute_action(current_obs=current_obs, current_states=current_states)
                    step_actions = self.process_actions(actions=actions.clone(), grad=grad)

                    # Step the vec_environment
                    next_obs, rews, dones, infos = self.vec_env.step(step_actions, (actions,grad))

                    next_states = self.vec_env.get_state()

                    # Record the transition
                    self.storage.add_transitions(current_obs, current_states, actions, rews, dones, values, actions_log_prob, mu, sigma)
                    current_obs.copy_(next_obs['obs'])
                    current_states.copy_(next_states)

                    # Book keeping
                    ep_infos.append(infos.copy())
                    # set_trace()
                    if 'ori_similarity' in self.vec_env.reward_type:
                        ori_sim_all.append(torch.mean(infos['ori_similarity']))
                    # self.writer.add_scalar('Episode/ori_sim_all', torch.mean(infos['ori_similarity']), _)

                    if self.print_log:
                        cur_reward_sum[:] += rews
                        cur_episode_length[:] += 1

                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        reward_sum.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        episode_length.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                    
                    # done
                    if torch.sum(dones) > 0:
                        current_obs = self.vec_env.reset(dones)['obs']
                        current_states = self.vec_env.get_state()
                        print(infos['success_rate'])
                        if 'ori_similarity' in self.vec_env.reward_type:
                            fig = plt.figure()
                            plt.plot(torch.tensor(ori_sim_all).cpu().numpy())
                            ori_sim_all_img = get_img_from_fig(fig, dpi=100)
                            # ori_sim_all_img = cv2.resize(ori_sim_all_img,(256,256))
                            self.writer.add_image("ori_sim", ori_sim_all_img, it, dataformats='HWC')

                if self.print_log:
                    # reward_sum = [x[0] for x in reward_sum]
                    # episode_length = [x[0] for x in episode_length]
                    rewbuffer.extend(reward_sum)
                    lenbuffer.extend(episode_length)

                _, _, last_values, _, _, _ = self.compute_action(current_obs=current_obs, current_states=current_states, mode='train')
                stop = time.time()
                collection_time = stop - start
                mean_trajectory_length, mean_reward = self.storage.get_statistics()

                # Learning step
                start = stop
                self.storage.compute_returns(last_values, self.gamma, self.lam)
                mean_value_loss, mean_surrogate_loss = self.update()
                self.storage.clear()
                stop = time.time()
                learn_time = stop - start
                if self.print_log:
                    self.log(locals())
                if it % log_interval == 0:
                    self.actor_critic.eval()
                    self.eval(it)
                    self.actor_critic.train()
                    self.vec_env.train()
                    self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))

                    current_obs = self.vec_env.reset()['obs']
                    current_states = self.vec_env.get_state()
                    cur_episode_length[:] = 0
                    # TODO clean extras
                ep_infos.clear()
            self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(num_learning_iterations)))

    def log(self, locs, width=70, pad=35):
        self.tot_timesteps += self.num_transitions_per_env * self.vec_env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                if key=='success_num':
                    value = torch.sum(infotensor)
                    self.writer.add_scalar('Episode/' + 'total_success_num', value, locs['it'])
                    ep_string += f"""{f'Total episode {key}:':>{pad}} {value:.4f}\n"""
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.actor_critic.log_std.exp().mean()

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        self.writer.add_scalar('Train2/mean_reward/step', locs['mean_reward'], locs['it'])
        self.writer.add_scalar('Train2/mean_episode_length/episode', locs['mean_trajectory_length'], locs['it'])

        fps = int(self.num_transitions_per_env * self.vec_env.num_envs / (locs['collection_time'] + locs['learn_time']))

        str = f" \033[1m Learning iteration {locs['it']}/{locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
                          f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                          f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                          f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0

        batch = self.storage.mini_batch_generator(self.num_mini_batches)

        for epoch in range(self.num_learning_epochs):
            # for obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch \
            #        in self.storage.mini_batch_generator(self.num_mini_batches):

            for indices in batch:
                # print(len(indices))

                obs_batch = self.storage.observations.view(-1, *self.storage.observations.size()[2:])[indices]
                if self.asymmetric:
                    states_batch = self.storage.states.view(-1, *self.storage.states.size()[2:])[indices]
                else:
                    states_batch = None
                actions_batch = self.storage.actions.view(-1, self.storage.actions.size(-1))[indices]
                target_values_batch = self.storage.values.view(-1, 1)[indices]
                returns_batch = self.storage.returns.view(-1, 1)[indices]
                old_actions_log_prob_batch = self.storage.actions_log_prob.view(-1, 1)[indices]
                advantages_batch = self.storage.advantages.view(-1, 1)[indices]
                old_mu_batch = self.storage.mu.view(-1, self.storage.actions.size(-1))[indices]
                old_sigma_batch = self.storage.sigma.view(-1, self.storage.actions.size(-1))[indices]

                actions_log_prob_batch, entropy_batch, value_batch, mu_batch, sigma_batch = self.actor_critic.evaluate(obs_batch,
                                                                                                                       states_batch,
                                                                                                                       actions_batch)

                # KL
                if self.desired_kl != None and self.schedule == 'adaptive':

                    kl = torch.sum(
                        sigma_batch - old_sigma_batch + (torch.square(old_sigma_batch.exp()) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch.exp())) - 0.5, axis=-1)
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.step_size = max(1e-5, self.step_size / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.step_size = min(1e-2, self.step_size * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.step_size

                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                   1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates

        return mean_value_loss, mean_surrogate_loss

    '''
    utils
    '''
    def grad_norm(self,grad):
        scale_grad = (torch.max((abs(grad)),dim=1)[0]).reshape(-1,1).expand_as(grad)
        grad = grad/scale_grad
        return grad
    
    
    def action2grad(self, x, inv=False, relative=True, cur_x=None):
        if not inv:
            batch_size = x.size(0)
            state_dim = x.size(1)
            x = torch.cat([torch.sin(x).reshape(batch_size,state_dim,1), torch.cos(x).reshape(batch_size,state_dim,1)],2).reshape(batch_size,-1)
            return x
        else:
            batch_size = x.size(0)
            state_dim = x.size(1)
            x = x.reshape(batch_size,int(state_dim/2),2)
            cur_x = cur_x.reshape(batch_size,int(state_dim/2),2)

            cur_x = torch.cat([-cur_x[:,:,0:1], cur_x[:,:,1:2]],dim=-1)
            ori_grad = torch.sum(torch.cat([x[:,:,1:2], x[:,:,0:1]], dim=-1) * cur_x, dim=-1, keepdim=True).reshape(batch_size,int(state_dim/2))
            return ori_grad
    
    def get_obs_with_grad(self, current_obs, reset=False, t=None):
        # compute score
        B = current_obs.size(0)
        cur_hand_dof = current_obs[:,:18].clone() #【-1，1】
        pcl_index = self.stack_frame_numer*7 + 18
        cur_obj_pcl = current_obs[:,pcl_index:self.points_per_object*3+pcl_index].clone().reshape(-1, 3, self.points_per_object)

        if reset:
            with torch.no_grad():  
                in_process_sample, res = cond_ode_sampler(
                        self.score,
                        self.prior_fn,
                        self.sde_fn,
                        (cur_hand_dof, cur_obj_pcl),
                        t0=0.5,
                        device=self.device,
                        num_steps=51,
                        batch_size=B,
                        space=self.args.space,
                    )
                goal_pose = in_process_sample[-1,:,:]
            return goal_pose
        else:
            if self.args.space == 'riemann':
                if 'direct' in self.args.score_model_path:
                    cur_hand_dof = self.vec_env.dof_norm(cur_hand_dof,inv=True)
                cur_hand_dof = self.action2grad(cur_hand_dof)

            if t is None:
                batch_time_step = torch.ones(B, device=self.device).unsqueeze(1) * self.t0
            else:
                t_max = 0.5
                t_min = 1e-5
                t = torch.tanh(t) * (t_max - t_min) / 2 + (t_max + t_min)/2
                batch_time_step = torch.clamp(t.reshape(B,-1), 1e-5, 0.5)
                self.vec_env.extras['t_value'] = torch.mean(abs(batch_time_step),-1)

            if self.args.space == 'riemann':
                grad = torch.zeros(B,36,device=self.device)
            elif self.args.space == 'euler':
                grad = torch.zeros(B,18,device=self.device)

            bz = 256
            iter_num = int(np.ceil(B/bz))

            for order in range(iter_num):
                with torch.no_grad():  
                    if self.args.space == 'riemann':
                        grad[order*bz:(order+1)*bz,:36] = self.score((cur_hand_dof[order*bz:(order+1)*bz,:], cur_obj_pcl[order*bz:(order+1)*bz,:]), batch_time_step[order*bz:(order+1)*bz,:]).detach()
                    elif self.args.space == 'euler':  
                        grad[order*bz:(order+1)*bz,:18] = self.score((cur_hand_dof[order*bz:(order+1)*bz,:], cur_obj_pcl[order*bz:(order+1)*bz,:]), batch_time_step[order*bz:(order+1)*bz,:]).detach()

            if self.args.space == 'riemann':
                grad = self.action2grad(grad, inv=True, cur_x=cur_hand_dof)

            if 'pure_ori_similarity' in self.vec_env.reward_type:
                self.ori_grad = grad.clone()

            if 'direct' not in self.args.score_model_path:
                #denormalize to dof original range
                grad = grad * self.vec_env.shadow_hand_dof_range[self.vec_env.actuated_dof_indices] / 2

            if self.grad_process is not None:
                if 'norm' in self.grad_process:
                    grad = self.grad_norm(grad)
                if 'clip' in self.grad_process:
                    grad = torch.clamp(grad,-self.grad_scale,self.grad_scale)
                if 'scale' in self.grad_process:
                    grad = grad * self.grad_scale

            if 'pure_ori_similarity' not in self.vec_env.reward_type:
                self.ori_grad = grad.clone()

            if self.action_type != 'controlt':
                current_obs[:,-18:] = grad

            # print(grad[0])
            return current_obs, grad
    
    def process_actions(self, actions, grad):
        if self.action_type=='joint':
            if self.sub_action_type=='add+jointscale':
                self.vec_env.extras['grad_ss_mean'] = torch.mean(abs(actions[:,:18]),-1)
                self.vec_env.extras['grad_ss_std'] = torch.std(abs(actions[:,:18]),-1)
                self.vec_env.extras['residual_mean'] = torch.mean(abs(actions[:,18:]),-1)
                self.vec_env.extras['residual_std'] = torch.std(abs(actions[:,18:]),-1)
                step_actions = grad*actions[:,:18] + actions[:,18:]
            else:
                step_actions = actions*grad
        elif self.action_type=='direct':
            step_actions = actions
        elif 'gf' in self.action_type:
            step_actions = grad
        return step_actions

    def compute_action(self, current_obs, current_states=None, mode='train'):
        # compute gf
        if 'gf' in self.vec_env.sub_obs_type:
            current_obs, grad = self.get_obs_with_grad(current_obs)
        else:
            grad = torch.zeros((current_obs.size(0),18), device=self.device)

        if self.pointnet_finetune:
            batch_num = current_obs.size(0)//self.finetune_pointnet_bz + 1
            for _ in range(batch_num):
                current_obs_batch = current_obs[self.finetune_pointnet_bz*_:self.finetune_pointnet_bz*(_+1),:]
                # current_states_batch = current_states[:,self.finetune_pointnet_bz*batch_num+self.finetune_pointnet_bz*(batch_num+1)]
                if mode=='train':
                    actions_batch, actions_log_prob_batch, values_batch, mu_batch, sigma_batch = self.actor_critic.act(current_obs_batch, current_states)
                else:
                    actions_batch = self.actor_critic.act_inference(current_obs_batch)
                if _ == 0:
                    if mode=='train':
                        actions, actions_log_prob, values, mu, sigma = actions_batch, actions_log_prob_batch, values_batch, mu_batch, sigma_batch
                    else:
                        actions = actions_batch
                else:
                    if mode=='train':
                        actions = torch.cat([actions, actions_batch])
                        actions_log_prob = torch.cat([actions_log_prob,actions_log_prob_batch])
                        values = torch.cat([values,values_batch])
                        mu = torch.cat([mu, mu_batch])
                        sigma = torch.cat([sigma, sigma_batch])
                    else:
                        actions = torch.cat([actions, actions_batch])
        else:
            if mode=='train':
                actions, actions_log_prob, values, mu, sigma = self.actor_critic.act(current_obs, current_states)
            else:
                actions = self.actor_critic.act_inference(current_obs)

        if mode=='train':
            return actions, actions_log_prob, values, mu, sigma, grad
        else:
            return actions, grad