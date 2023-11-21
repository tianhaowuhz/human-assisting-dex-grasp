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
import time
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
from utils.utils import exists_or_mkdir, save_video, get_dict_key, DexDataset

points_per_object = 1024
vis_image = False
max_bz = 256
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ''' configurator '''
    # score matching parameter
    parser.add_argument('--test_decay', type=str, default='False')
    parser.add_argument('--score_mode', type=str, default='target')
    parser.add_argument('--sde_mode', type=str, default='vp') # ['ve', 'vp', 'subvp']
    parser.add_argument('--sde_min', type=float, default=0.1)
    parser.add_argument('--sde_max', type=float, default=10.0)
    parser.add_argument('--n_epoches', type=int, default=50000)
    parser.add_argument('--eval_freq', type=int, default=1000)
    parser.add_argument('--eval_times', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--t0', type=float, default=0.5)
    parser.add_argument('--ema_rate', type=float, default=0.999)
    parser.add_argument('--repeat_num', type=int, default=1)
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--grad_clip', type=float, default=1.)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--base_noise_scale', type=float, default=0.01)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--hidden_dim', type=int, default=1024)
    parser.add_argument('--embed_dim', type=int, default=512)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--relative', action='store_true', help="relative obj pcl state")

    # env
    parser.add_argument('--num_envs', type=int, default=4, help='total env nums')
    parser.add_argument('--num_run_envs', type=int, default=1, help='running env nums')
    parser.add_argument('--max_episode_steps', type=int, default=200, help='step numbers for each episode')
    parser.add_argument("--method", type=str, default='filter')
    parser.add_argument('--gui', action='store_false', help="enable gui")
    parser.add_argument("--mode", type=str, default='eval')
    parser.add_argument("--dataset_type", type=str, default='train')

    # mode 
    parser.add_argument('--model_name', type=str, default='fine', metavar='NAME', help="the name of the model (default: fine")
    parser.add_argument('--quick', action='store_true', help="test on small cases")
    parser.add_argument('--train_model', action='store_true', help="train model")
    parser.add_argument('--con', action='store_true', help="continue train the given model")
    parser.add_argument('--demo_gen', action='store_true', help="demo gen mode")
    parser.add_argument('--demo_nums', type=int, default=8, help='total demo nums')
    parser.add_argument('--demo_name', type=str, default='small_test', help='demo names')
    parser.add_argument('--space', type=str, default='riemann', help='angle space')
    parser.add_argument('--eval_demo_name', type=str, default='small_test', help='demo names')
    parser.add_argument('--constrained', action='store_false', help="whether constrain base")
    parser.add_argument('--gt', action='store_true', help="gt mode")
    parser.add_argument('--device_id', type=int, default=0, help='device_id')
    # tensorboard
    parser.add_argument("--log_dir", type=str, default='gf_overfit')
    parser.add_argument("--pt_version", type=str, default='pt2')

    args = parser.parse_args()

    device = f'cuda:{args.device_id}'
    
    ''' make env '''
    num_envs = args.num_envs # 53
    envs = condexenvs.make(
        seed=args.seed, 
        task="ShadowHandCon", 
        num_envs=num_envs, 
        sim_device=device,
        rl_device=device,
        graphics_device_id = args.device_id,
        virtual_screen_capture=False,
        headless=args.gui,
        force_render = False,
        mode = args.mode, 
        num_run_envs = args.num_run_envs,
        method = args.method,
        dataset_type = args.dataset_type,
    )
    envs.reset(env_init=True)

    print(args)
    # set_trace()

    ''' seed '''
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    random.seed(args.seed)

    ''' logging '''
    exists_or_mkdir('./logs')
    ckpt_path = f'./logs/{args.log_dir}/'
    exists_or_mkdir(ckpt_path)
    tb_path = f'./logs/{args.log_dir}/tb'
    exists_or_mkdir(tb_path)
    writer = SummaryWriter(tb_path)
        
    ''' create train dataset and dataloader '''
    dataset_path = f'./ExpertDatasets/grasp_data/ground/{args.demo_name}.pth'
    assert os.path.exists(dataset_path), 'Dataset not found!'
    with open(dataset_path, 'rb') as f:
        data_samples = pickle.load(f)
    print(len(data_samples))

    '''
    eval 
    '''
    eval_dataset_path = f'./ExpertDatasets/grasp_data/ground/{args.eval_demo_name}_rc.pth'
    assert os.path.exists(eval_dataset_path), 'Eval Dataset not found!'
    with open(eval_dataset_path, 'rb') as f:
        eval_data_samples = pickle.load(f)

    eval_dataset_ot_path = f'./ExpertDatasets/grasp_data/ground/{args.eval_demo_name}_rc_ot.pth'
    assert os.path.exists(eval_dataset_ot_path), 'Eval Dataset oti not found!'
    with open(eval_dataset_ot_path, 'rb') as f:
        eval_data_ot = pickle.load(f)
    
    # change data object type id
    eval_dataset_oti_path = f'./ExpertDatasets/grasp_data/ground/{args.eval_demo_name}_oti.pth'
    with open(eval_dataset_oti_path, 'rb') as f:
        eval_data_oti = pickle.load(f)

    for (i, data) in enumerate(eval_data_samples):
        env_id_in_full = int(data[25+points_per_object*3+7:25+points_per_object*3+8])
        object_type = get_dict_key(eval_data_oti, env_id_in_full)
        env_id_in_current = envs.obj_type_id[object_type]
        eval_data_samples[i,3104] = env_id_in_current
    
    eval_dataset = torch.tensor(eval_data_samples, device=device)
    eval_dataset = eval_dataset.reshape(-1, eval_dataset.shape[-1])
    args.num_envs = len(eval_data_ot)
    num_envs = len(eval_data_ot)
    test_per_object = int(len(eval_dataset)/num_envs)
    eval_demo_number = len(eval_data_samples)

    total_data_number = len(data_samples)
    # augment demos
    if total_data_number < args.demo_nums:
        new_data_samples = data_samples
        for i in range(args.demo_nums - total_data_number): 
            new_data_samples = np.vstack((new_data_samples,data_samples[i%total_data_number]))
        dataset = new_data_samples
    else:
        dataset = data_samples
    
    dataset = dataset[: args.demo_nums]
    dataset = dataset.reshape(-1, dataset.shape[-1])
    # set_trace()
    if 'all' in dataset_path:
        print('balance data')
        dataset = DexDataset(dataset)
        print(len(dataset))
    if args.relative:
        dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
    else:
        dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)

    ''' SDE '''
    # init SDE config
    prior_fn, marginal_prob_fn, sde_fn = init_sde(args.sde_mode, min=args.sde_min, max=args.sde_max)

    score = CondScoreModel(
        marginal_prob_fn,
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim,
        mode=args.score_mode,
        relative=args.relative,
        space=args.space,
        pointnet_version=args.pt_version
    )
    score.to(device)
    optimizer = optim.Adam(score.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    ema = ExponentialMovingAverage(score.parameters(), decay=args.ema_rate)

    # set_trace()
    print(total_data_number, dataset_path)
    ''' training loop '''
    print("Starting Training Loop...")
    for epoch in tqdm.trange(args.n_epoches):
        # For each batch in the dataloader
        # set_trace()
        for i, dex_data in enumerate(dataloader):
            cur_step = i + epoch * len(dataloader)
            # dex_data item: 0~17: joint parameters, 
            # 18~24: hand base pose, pos(3)+oriq(4) 
            # 25~3096 obj point cloud, 
            # 3097~3103 obj pose pos(3)+oriq(4)
            # 3104~3105 env_id 
            dex_data = torch.tensor(dex_data, device=device)
            state_size = dex_data.size(1)

            if args.relative:
                hand_dof = dex_data[:,:18].clone().to(device).float()
                hand_pos = dex_data[:,18:21].clone().to(device).float()
                hand_quat = dex_data[:,21:25].clone().to(device).float()
                obj_pcl = dex_data[:,25:points_per_object*3+25].clone().to(device).float().reshape(-1, points_per_object, 3)
                obj_pcl_2h = envs.get_obj2hand(obj_pcl, hand_quat, hand_pos).reshape(-1, 3, points_per_object)
            else:
                hand_dof = dex_data[:,:25].clone().to(device).float()
                obj_pcl_2h = dex_data[:,25:points_per_object*3+25].clone().to(device).float().reshape(-1, 3, points_per_object)
            
            if args.space=='riemann' and ('direct' in args.log_dir):
                hand_dof = envs.dof_norm(hand_dof,inv=True)

            # calc score-matching loss
            loss = 0
            total_loss = 0
            optimizer.zero_grad()
            for _ in range(args.repeat_num):
                if args.batch_size > max_bz:
                    iter_num = int(np.ceil(args.batch_size/max_bz))
                    for iter in range(iter_num):
                        loss = loss_fn_cond(score, (hand_dof[iter*max_bz:(iter+1)*max_bz,:], obj_pcl_2h[iter*max_bz:(iter+1)*max_bz,:]), marginal_prob_fn, sde_fn, is_likelihood_weighting=False, device=device, full_state=dex_data[iter*max_bz:(iter+1)*max_bz,:].clone(), envs=envs, space=args.space, relative=args.relative)
                        loss /= iter_num
                        loss /= args.repeat_num
                        total_loss += loss.item()
                        loss.backward()
                else:
                    loss = loss_fn_cond(score, (hand_dof, obj_pcl_2h), marginal_prob_fn, sde_fn, is_likelihood_weighting=False, device=device, full_state=dex_data, envs=envs, space=args.space, relative=args.relative)
                    loss /= args.repeat_num
                    total_loss += loss.item()
                    loss.backward()
            
            ''' warmup '''
            if args.warmup > 0:
                for g in optimizer.param_groups:
                    g['lr'] = args.lr * np.minimum(cur_step / args.warmup, 1.0)
            ''' grad clip '''
            if args.grad_clip >= 0:
                torch.nn.utils.clip_grad_norm_(score.parameters(), max_norm=args.grad_clip)

            optimizer.step()
            # add writer
            writer.add_scalar('train_loss/current', total_loss, cur_step)
            
            ''' ema '''
            ema.update(score.parameters())
            ''' get ema training loss '''
            if args.ema_rate > 0 and cur_step % 5 == 0:
                ema.store(score.parameters())
                ema.copy_to(score.parameters())
                with torch.no_grad():
                    loss = 0
                    total_loss = 0
                    for _ in range(args.repeat_num):
                        # calc score-matching loss
                        if args.batch_size > max_bz:
                            iter_num = int(np.ceil(args.batch_size/max_bz))
                            for iter in range(iter_num):
                                loss = loss_fn_cond(score, (hand_dof[iter*max_bz:(iter+1)*max_bz,:], obj_pcl_2h[iter*max_bz:(iter+1)*max_bz,:]), marginal_prob_fn, sde_fn, is_likelihood_weighting=False, device=device, full_state=dex_data[iter*max_bz:(iter+1)*max_bz,:], envs=envs, space=args.space, relative=args.relative)
                                loss /= iter_num
                                loss /= args.repeat_num
                                total_loss += loss.item()
                        else:
                            loss = loss_fn_cond(score, (hand_dof, obj_pcl_2h), marginal_prob_fn, sde_fn, is_likelihood_weighting=False, device=device, full_state=dex_data, envs=envs, space=args.space, relative=args.relative)
                            loss /= args.repeat_num
                            total_loss += loss.item()
                    writer.add_scalar('train_loss/ema', loss, cur_step)
                ema.restore(score.parameters())
        
        ''' evaluation '''
        score.eval()
        if (epoch+1) % args.eval_freq == 0:
            with torch.no_grad():  
                # checkpoint current parameters
                ema.store(score.parameters())
                ema.copy_to(score.parameters())

                total_success_numer = 0

                with tqdm.tqdm(total=eval_demo_number) as pbar:
                    for eval_order in range(1):
                        eval_data_idx = np.linspace(eval_order,test_per_object*(num_envs-1)+eval_order,num_envs, dtype=int)
                        eval_dex_data = eval_dataset[eval_data_idx,:]

                        if args.relative:
                            hand_dof = eval_dex_data[:,:18].clone().to(device).float()
                            hand_pos = eval_dex_data[:,18:21].clone().to(device).float()
                            hand_quat = eval_dex_data[:,21:25].clone().to(device).float()
                            obj_pcl = eval_dex_data[:,25:3097].clone().to(device).float().reshape(-1, 1024, 3)
                            obj_pcl_2h = envs.get_obj2hand(obj_pcl, hand_quat, hand_pos).reshape(-1, 3, 1024)
                        else:
                            hand_dof = eval_dex_data[:,:25].clone().to(device).float()
                            obj_pcl_2h = eval_dex_data[:,25:3097].clone().to(device).float().reshape(-1, 3, 1024)
                        
                        if args.space=='riemann' and ('direct' in args.log_dir):
                            hand_dof = envs.dof_norm(hand_dof,inv=True)

                        with torch.no_grad():  
                            iter_num = int(np.ceil(num_envs/max_bz))
                            in_process_sample_last = torch.zeros_like(eval_dex_data)[:,:18]

                            for order in range(iter_num):
                                in_process_sample, res = cond_ode_sampler(
                                    score,
                                    prior_fn,
                                    sde_fn,
                                    (hand_dof[order*max_bz:(order+1)*max_bz,:], obj_pcl_2h[order*max_bz:(order+1)*max_bz,:]),
                                    t0=args.t0,
                                    device=device,
                                    num_steps=500,
                                    batch_size=len(hand_dof[order*max_bz:(order+1)*max_bz,:]),
                                    full_state=eval_dex_data[order*max_bz:(order+1)*max_bz,:].clone(), 
                                    envs=envs, 
                                    space=args.space,
                                    relative=args.relative,
                                )
                                if args.space == 'riemann' and ('direct' in args.log_dir):
                                    in_process_sample_last[order*max_bz:(order+1)*max_bz,:18] = torch.clip(envs.dof_norm(in_process_sample[-1,:,:18].clone()),-1,1).reshape(-1,18)
                                else:
                                    in_process_sample_last[order*max_bz:(order+1)*max_bz,:18] = torch.clip(in_process_sample[-1,:,:18].clone(),-1,1).reshape(-1,18)

                            states = torch.tensor([],device=device)
                            video_state = eval_dex_data.clone()
                            video_state[:,:18] = in_process_sample_last

                            success = envs.grasp_filter(states=video_state,close_dis=0.1)
                            total_success_numer += torch.sum(success).cpu().numpy()
                            print(f'success rate: {total_success_numer/((eval_order+1)*num_envs)}, total_test_time: {((eval_order+1)*num_envs)}')
                        pbar.update(num_envs)

                writer.add_scalar('eval/success_rate', total_success_numer/((eval_order+1)*num_envs), epoch)
                torch.save(score.cpu().state_dict(), ckpt_path + f'score_{epoch}.pt')
                torch.save(score.obj_enc.cpu().state_dict(), ckpt_path + f'{args.pt_version}_{epoch}.pt')
                score.to(device)

                # restore checkpointed parameters, and continue training
                ema.restore(score.parameters())
        score.train()



