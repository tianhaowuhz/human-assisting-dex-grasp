import isaacgym
import condexenvs
import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from Algorithms.ppo import GFPPO
from utils.config import load_cfg, get_args, set_np_formatting

if __name__ == '__main__':
    set_np_formatting()

    args = get_args()
    cfg_train, logdir = load_cfg(args)

    '''
    change for different method
    '''
    cfg_train['setting']['grad_scale'] = 1.0
    reward_normalize = False
    cfg_train['policy']['pointnet_version'] = 'pt'

    if args.exp_name == 'ours':
        reward_type = "ori_similarity+height+sr_faketendon+1osr+175h_normriemann+26999pt"
        reward_normalize = False
        sub_obs_type = "joint+fingertipjoint+wrist+objpcl+gf"
        cfg_train['setting']['action_type'] = "joint"
        cfg_train['setting']['sub_action_type'] = "add+jointscale"
        cfg_train['policy']['pretrain_pointnet'] = True

    '''
    policy
    '''  
    
    cfg_train['policy']['hand_pcl'] = False

    '''
    training setting
    '''
    cfg_train['learn']['nsteps'] = 50
    cfg_train['learn']['noptepochs'] = 2
    cfg_train['learn']['nminibatches'] = int(args.num_envs*cfg_train['learn']['nsteps']/64)
    if cfg_train['policy']['pointnet_version'] == 'pt':
        cfg_train['learn']['optim_stepsize'] = 0.0003
        # cfg_train['policy']['norm_action'] = True
    else:
        cfg_train['learn']['optim_stepsize'] = 0.0003
    cfg_train['learn']['desired_kl'] = 0.016
    cfg_train['learn']['gamma'] = 0.99
        
    envs = condexenvs.make(
        seed=args.seed, 
        task="ShadowHandCon", 
        num_envs=args.num_envs, 
        sim_device=f"cuda:{args.run_device_id}",
        rl_device=f"cuda:{args.run_device_id}",
        graphics_device_id = -1,
        headless=args.headless,
        mode = args.mode, 
        eval_times=args.eval_times,
        method = args.method,
        constrained = args.constrained,
        reward_type = reward_type,
        reward_normalize = reward_normalize, 
        sub_obs_type = sub_obs_type,
        dataset_type = args.dataset_type,
    )
    envs.reset(env_init=True)

    learn_cfg = cfg_train["learn"]
    is_testing = learn_cfg["test"]
    # Override resume and testing flags if they are passed as parameters.
    if args.model_dir != "":
        chkpt_path = args.model_dir
    logdir = logdir + "_seed{}".format(args.seed)

    runner = GFPPO(vec_env=envs,
                cfg_train = cfg_train,
                device=envs.rl_device,
                sampler=learn_cfg.get("sampler", 'sequential'),
                log_dir=logdir,
                is_testing=is_testing,
                print_log=learn_cfg["print_log"],
                apply_reset=False,
                asymmetric=False,
                args=args,
            )

    if args.model_dir != "":
        if is_testing:
            runner.test(chkpt_path) 
        else:
            runner.load(chkpt_path) 

    iterations = cfg_train["learn"]["max_iterations"]
    if args.max_iterations > 0:
        iterations = args.max_iterations

    runner.run(num_learning_iterations=iterations, log_interval=cfg_train["learn"]["save_interval"])