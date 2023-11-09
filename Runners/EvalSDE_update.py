#!/usr/bin/env python
import isaacgym
import condexenvs
import argparse
import functools
import sys
import os
import cv2
import numpy as np
import tqdm
import pickle
import random
from ipdb import set_trace
from tensorboardX import SummaryWriter

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision.utils import make_grid

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from Algorithms.SDE_update import loss_fn_cond, cond_ode_sampler, init_sde, ExponentialMovingAverage
from Networks.SDENets_update import CondScoreModel
from utils.utils import exists_or_mkdir, save_video, get_dict_key
from utils.hand_model import HandModel

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
points_per_object = 1024

def visualize_states(states, logger, suffix, epoch, nrow=4, save_path=None):
    imgs = []
    for state in states:
        env_id = int(state[25+points_per_object*3+7:25+points_per_object*3+8].long().cpu().numpy())
        envs.set_states(state.unsqueeze(0))
        img = envs.render(rgb=True,img_size=256)[env_id]
        if save_path is not None:
            cv2.imwrite(f'{save_path}_{env_id}_{suffix}_{epoch}.jpg', img.cpu().numpy())
        img = torch.flip(img, [-1])
        imgs.append(img) # rendered_img: [256, 256, 3]
    
    ts_imgs = torch.stack(imgs).permute(0, 3, 1, 2)
    grid = make_grid(ts_imgs.float(), padding=2, nrow=nrow, normalize=True)
    logger.add_image(f'Images/{suffix}', grid, epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ''' configurator '''
    # score matching parameter
    parser.add_argument('--test_decay', type=str, default='False')
    parser.add_argument('--score_mode', type=str, default='target')
    parser.add_argument('--sde_mode', type=str, default='vp') # ['ve', 'vp', 'subvp']
    parser.add_argument('--sde_min', type=float, default=0.1)
    parser.add_argument('--sde_max', type=float, default=10.0)
    parser.add_argument('--n_epoches', type=int, default=10000)
    parser.add_argument('--eval_freq', type=int, default=10)
    parser.add_argument('--eval_times', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--t0', type=float, default=1e-3)
    parser.add_argument('--ema_rate', type=float, default=0.999)
    parser.add_argument('--repeat_num', type=int, default=1)
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--grad_clip', type=float, default=1.)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--base_noise_scale', type=float, default=0.01)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--relative', action='store_true', help="relative obj pcl state")
    parser.add_argument('--hand_pcl', action='store_true', help="encode hand pcl")
    parser.add_argument('--gf_rot', action='store_true', help="rot in traj gen")

    # env
    parser.add_argument('--num_envs', type=int, default=4, help='total env nums')
    parser.add_argument('--num_run_envs', type=int, default=1, help='running env nums')
    parser.add_argument('--max_episode_steps', type=int, default=200, help='step numbers for each episode')
    parser.add_argument("--method", type=str, default='filter')
    parser.add_argument('--gui', action='store_false', help="enable gui")
    parser.add_argument("--mode", type=str, default='eval')
    parser.add_argument("--dataset_type", type=str, default='unseencategory')

    # mode 
    parser.add_argument('--model_name', type=str, default='fine', metavar='NAME', help="the name of the model (default: fine")
    parser.add_argument('--quick', action='store_true', help="test on small cases")
    parser.add_argument('--train_model', action='store_true', help="train model")
    parser.add_argument('--con', action='store_true', help="continue train the given model")
    parser.add_argument('--demo_gen', action='store_true', help="demo gen mode")
    parser.add_argument('--demo_nums', type=int, default=8, help='total demo nums')
    parser.add_argument('--demo_name', type=str, default='small_test', help='demo names')
    parser.add_argument('--space', type=str, default='riemann', help='angle space')
    parser.add_argument('--constrained', action='store_true', help="whether constrain base")
    parser.add_argument('--gt', action='store_true', help="gt mode")
    parser.add_argument('--pointnet_network_type', type=str, default='ori', help='demo names')
    parser.add_argument('--device_id', type=int, default=0, help='device_id')

    # log
    parser.add_argument("--video_freq", type=int, default=1, help="save video freq")
    parser.add_argument("--log_dir", type=str, default='gf_overfit')

    args = parser.parse_args()
    
    device_id = args.device_id
    device = f'cuda:{device_id}'

    hand_model = HandModel(
            mjcf_path='mjcf/shadow_hand_vis.xml',
            mesh_path='mjcf/meshes',
            device=device
        )

    ''' make env '''
    num_envs = args.num_envs # 53
    envs = condexenvs.make(
        seed=args.seed, 
        task="ShadowHandCon", 
        num_envs=num_envs, 
        sim_device=device,
        rl_device=device,
        graphics_device_id = device_id,
        # virtual_screen_capture=True,
        headless=args.gui,
        force_render = False,
        mode = args.mode, 
        num_run_envs = args.num_run_envs,
        method = args.method,
        dataset_type = args.dataset_type,
    )
    envs.reset(env_init=True)

    ''' seed '''
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    random.seed(args.seed)

    ''' logging '''
    exists_or_mkdir('./logs')
    debug_path = f'./logs/{args.log_dir}/debug/'
    exists_or_mkdir(debug_path)
    tb_path = f'./logs/{args.log_dir}/tb'
    exists_or_mkdir(tb_path)
    writer = SummaryWriter(tb_path)
        
    ''' create dataset and dataloader '''
    dataset_path = f'./ExpertDatasets/grasp_data/ground/{args.demo_name}_rc.pth'
    assert os.path.exists(dataset_path), 'Dataset not found!'
    with open(dataset_path, 'rb') as f:
        data_samples = pickle.load(f)

    dataset_ot_path = f'./ExpertDatasets/grasp_data/ground/{args.demo_name}_rc_ot.pth'
    assert os.path.exists(dataset_ot_path), 'Dataset oti not found!'
    with open(dataset_ot_path, 'rb') as f:
        data_ot = pickle.load(f)
    
    # change data oti
    dataset_oti_path = f'./ExpertDatasets/grasp_data/ground/{args.demo_name}_oti.pth'
    with open(dataset_oti_path, 'rb') as f:
        data_oti = pickle.load(f)

    for (i, data) in enumerate(data_samples):
        env_id_in_full = int(data[25+points_per_object*3+7:25+points_per_object*3+8])
        object_type = get_dict_key(data_oti, env_id_in_full)
        env_id_in_current = envs.obj_type_id[object_type]
        # if env_id_in_full != env_id_in_current:
        #     set_trace()
        data_samples[i,3104] = env_id_in_current

    # data_samples = data_samples[data_samples[:,-1]==0]

    # data_samples = data_samples[6:8,:]
    total_data_number = len(data_samples)
    print(args.demo_name, total_data_number)
    # augment demos
    if total_data_number < args.demo_nums:
        # new_data_samples = data_samples[0]
        # for i in range(args.demo_nums): 
        #     new_data_samples = np.vstack((new_data_samples,data_samples[(i+1)%total_data_number]))
        # dataset = torch.tensor(new_data_samples, device=device)
        new_data_samples = data_samples
        for i in range(args.demo_nums - total_data_number): 
            new_data_samples = np.vstack((new_data_samples,data_samples[i%total_data_number]))
        dataset = torch.tensor(new_data_samples, device=device)
    else:
        dataset = torch.tensor(data_samples, device=device)
    
    dataset = dataset[: args.demo_nums]
    dataset = dataset.reshape(-1, dataset.shape[-1])

    ''' SDE '''
    # init SDE config
    prior_fn, marginal_prob_fn, sde_fn = init_sde(args.sde_mode, min=args.sde_min, max=args.sde_max)
    score = CondScoreModel(
        marginal_prob_fn,
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim,
        mode=args.score_mode,
        relative=args.relative,
        pointnet_network_type=args.pointnet_network_type,
        space=args.space,
    )
    model_dict = torch.load(args.model_name)
    score.load_state_dict(model_dict)
    score.to(device)
    
    ''' evaluation loop '''
    print("Starting evaluation Loop...")
    score.eval()

    total_success_numer = 0

    # set_trace()
    args.num_envs = len(data_ot)
    num_envs = len(data_ot)
    success_each_object = np.zeros(args.num_envs)
    
    test_per_object = int(len(dataset)/num_envs)
    i = 0

    # set_trace()
    with tqdm.tqdm(total=args.demo_nums) as pbar:
        for i in range(test_per_object):
            data_idx = np.linspace(i,test_per_object*(num_envs-1)+i,num_envs, dtype=int)
            dex_data = dataset[data_idx,:]
            # dex_data = envs.aug_data(dex_data, aug_dis=0)

            if args.relative:
                hand_dof = dex_data[:,:18].clone().to(device).float()
                hand_pos = dex_data[:,18:21].clone().to(device).float()
                hand_quat = dex_data[:,21:25].clone().to(device).float()
                obj_pcl = dex_data[:,25:3097].clone().to(device).float().reshape(-1, 1024, 3)
                obj_pcl_2h = envs.get_obj2hand(obj_pcl, hand_quat, hand_pos).reshape(-1, 3, 1024)
            else:
                hand_dof = dex_data[:,:25].clone().to(device).float()
                obj_pcl_2h = dex_data[:,25:3097].clone().to(device).float().reshape(-1, 3, 1024)

            if args.space=='riemann' and ('direct' in args.model_name):
                hand_dof = envs.dof_norm(hand_dof,inv=True)
            
            with torch.no_grad():  
                # visualize_states(dex_data.clone(), writer, 'GT', i)

                if not args.gt:
                    # eval_data_len = len(dex_data)
                    bz = 128
                    iter_num = int(np.ceil(num_envs/bz))
                    in_process_sample_last = torch.zeros_like(dex_data)[:,:18]
                    # in_process_sample = torch.zeros()
                    # set_trace()
                    for order in range(iter_num):
                        in_process_sample, res = cond_ode_sampler(
                            score,
                            prior_fn,
                            sde_fn,
                            (hand_dof[order*bz:(order+1)*bz,:], obj_pcl_2h[order*bz:(order+1)*bz,:]),
                            t0=args.t0,
                            device=device,
                            num_steps=20,
                            batch_size=len(hand_dof[order*bz:(order+1)*bz,:]),
                            hand_pcl=args.hand_pcl, 
                            full_state=dex_data[order*bz:(order+1)*bz,:].clone(), 
                            envs=envs, 
                            hand_model=hand_model,
                            space=args.space,
                            relative=args.relative,
                        )
                        # set_trace()
                        if args.space == 'riemann' and ('direct' in args.model_name):
                            # set_trace()
                            # in_process_sample_last[order*bz:(order+1)*bz,:18] = torch.clip(in_process_sample[-1,:,:18].clone(),-1,1).reshape(-1,18)
                            in_process_sample_last[order*bz:(order+1)*bz,:18] = torch.clip(envs.dof_norm(in_process_sample[-1,:,:18].clone()),-1,1).reshape(-1,18)
                        else:
                            in_process_sample_last[order*bz:(order+1)*bz,:18] = torch.clip(in_process_sample[-1,:,:18].clone(),-1,1).reshape(-1,18)

                if (i+1)%args.video_freq==0:
                    # quat_norm = torch.norm(res_ode[:,21:25],dim=1).reshape(test_num,-1).expand_as(res_ode[:,21:25])
                    # res_ode[:,21:25] = res_ode[:,21:25]/quat_norm
                    states = torch.tensor([],device=device)
                    video_state = dex_data.clone()

                    env_ids = [0]
                    # video_state = envs.aug_data(video_state, aug_dis=0)
                    # for env_id in env_ids:
                    #     visualize_states(video_state[env_id,:].unsqueeze(0), writer, 'GT', i+1, save_path=debug_path)

                    if args.constrained:
                        traj = envs.traj_gen(video_state, random_joint=True)
                        # get init state
                        cur_hand_state = traj[:,:1,:].clone().to(device).float()
                        states = torch.cat([states, cur_hand_state])

                        # get init_batch
                        if args.relative:
                            cur_hand_dof = traj[:, 0,:18].clone().to(device).float()
                            hand_pos = traj[:, 0,18:21].clone().to(device).float().reshape(-1,3)
                            hand_quat = traj[:, 0,21:25].clone().to(device).float().reshape(-1,4)
                            obj_pcl = traj[:, 0,25:3097].clone().to(device).float().reshape(-1, 1024, 3)
                            obj_pcl = envs.get_obj2hand(obj_pcl, hand_quat, hand_pos).reshape(-1, 3, 1024)
                        else:
                            cur_hand_dof = traj[:, 0,:25].clone().to(device).float()
                            obj_pcl = traj[:, 0,25:3097].clone().to(device).float().reshape(-1, 3, 1024)


                        curr_time = np.linspace(0.5, 0.01, traj.size(1))
                        eps=1

                        for step in range((traj.size(1)-1)):
                            # cal new hand dof 
                            batch_time_step = torch.ones(traj.size(0), device=device) * 0.1
                            grad = score((cur_hand_dof, obj_pcl), batch_time_step)
                            scale_grad = (torch.max((abs(grad)),dim=1)[0]).reshape(-1,1).expand_as(grad)
                            actions = grad/scale_grad
                            # set_trace()
                            # print(actions[0])
                            cur_hand_dof += actions * 0.025
                            # cur_hand_dof += eps * grad
                            cur_hand_dof = torch.clamp(cur_hand_dof, -1.0, 1.0)
                            # set hand dof to next state
                            next_state = traj[:,step+1,:]
                            next_state[:,:18] = cur_hand_dof[:,:18]
                            # concat next state
                            states = torch.cat([states, next_state.reshape(traj.size(0),1,-1)],1)

                            # update batch
                            if args.relative:
                                cur_hand_dof = next_state[:, :18].clone().to(device).float()
                                hand_pos = next_state[:, 18:21].clone().to(device).float().reshape(-1,3)
                                hand_quat = next_state[:, 21:25].clone().to(device).float().reshape(-1,4)
                                obj_pcl = next_state[:, 25:3097].clone().to(device).float().reshape(-1, 1024, 3)
                                obj_pcl = envs.get_obj2hand(obj_pcl, hand_quat, hand_pos).reshape(-1, 3, 1024)
                            else:
                                cur_hand_dof[:, :25] = next_state[:, :25]
                                obj_pcl = next_state[:, 25:3097].clone().to(device).float().reshape(-1, 3, 1024)

                        success = envs.grasp_filter(states=states[:,-1,:],close_dis=0.1)
                        success_each_object += success.cpu().numpy()
                        total_success_numer += torch.sum(success).cpu().numpy()
                        print(success_each_object)
                        print(f'success rate: {total_success_numer/((i+1)*num_envs)}, total_test_time: {((i+1)*num_envs)}')
                        # for video
                        for env_id in env_ids:
                            # print(states[env_id,:,:][:,:18])
                            save_video(envs, states[env_id,:,:], save_path=debug_path + f'{env_id}_Generated_{i+1}', suffix='mp4', points_per_object=points_per_object)
                    else:
                        if not args.gt:
                            # in_process_sample_last = torch.clip(in_process_sample[-1,:,:].clone(),-1,1)
                            video_state[:,:18] = in_process_sample_last
                        success = envs.grasp_filter(states=video_state,close_dis=0.1)
                        success_each_object += success.cpu().numpy()
                        total_success_numer += torch.sum(success).cpu().numpy()
                        print(success_each_object)
                        print(f'success rate: {total_success_numer/((i+1)*num_envs)}, total_test_time: {((i+1)*num_envs)}')

                        
                        # if not args.gt:
                        #     video_state = video_state[env_id:env_id+1,:]
                        #     for ips_ode in in_process_sample[:, env_id:env_id+1, :]:
                        #         # quat_norm = torch.norm(ips_ode[:,21:25],dim=1).reshape(test_num,-1).expand_as(ips_ode[:,21:25])
                        #         # ips_ode[:,21:25] = ips_ode[:,21:25]/quat_norm
                        #         video_state[:,:18] = ips_ode[:,:18]
                        #         states = torch.cat([states, video_state])
                        # save_video(envs, states, save_path=debug_path + f'{env_id}_Generated_{i+1}', suffix='mp4')
            
            pbar.update(num_envs)
