import hydra
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from condexenvs.utils.reformat import omegaconf_to_dict


OmegaConf.register_new_resolver('eq', lambda x, y: x.lower()==y.lower())
OmegaConf.register_new_resolver('contains', lambda x, y: x.lower() in y.lower())
OmegaConf.register_new_resolver('if', lambda pred, a, b: a if pred else b)
OmegaConf.register_new_resolver('resolve_default', lambda default, arg: default if arg=='' else arg)


def make(
    seed: int, 
    task: str, 
    num_envs: int, 
    sim_device: str,
    rl_device: str,
    graphics_device_id: int = -1,
    headless: bool = False,
    multi_gpu: bool = False,
    virtual_screen_capture: bool = False,
    force_render: bool = True,
    cfg: DictConfig = None,
    mode: str = 'train',
    num_run_envs: int = 3,
    eval_times: int = 3,
    method: str = 'gf',
    constrained: bool = False,
    reward_type = 'sr',
    reward_normalize = False,
    sub_obs_type = 'joint+fingertipjoint+wrist+objpcl',
    object_shapes: list = None,
    dataset_type: str = 'train',
    cem_traj_gen: bool = False,
): 
    from condexenvs.utils.rlgames_utils import get_rlgames_env_creator
    # create hydra config if no config passed in
    if cfg is None:
        # reset current hydra config if already parsed (but not passed in here)
        if HydraConfig.initialized():
            task = HydraConfig.get().runtime.choices['task']
            hydra.core.global_hydra.GlobalHydra.instance().clear()

        with initialize(config_path="./cfg"):
            cfg = compose(config_name="config", overrides=[f"task={task}"])
            cfg_dict = omegaconf_to_dict(cfg.task)
            cfg_dict['env']['numEnvs'] = num_envs
            cfg_dict['env']['runEnvs'] = num_run_envs
            cfg_dict['env']['method'] = method
            cfg_dict['env']['constrained'] = constrained
            cfg_dict['env']['rewardType'] = reward_type
            cfg_dict['env']['rewardNormalize'] = reward_normalize
            cfg_dict['env']['subObservationType'] = sub_obs_type
            cfg_dict['env']['object_shapes'] = object_shapes
            cfg_dict['env']['asset']['assetSubType'] = dataset_type
            cfg_dict['env']['cemTrajGen']= cem_traj_gen
            cfg_dict['task']['mode'] = mode
            cfg_dict['task']['eval_times'] = eval_times
    # reuse existing config
    else:
        cfg_dict = omegaconf_to_dict(cfg.task)

    create_rlgpu_env = get_rlgames_env_creator(
        seed=seed,
        task_config=cfg_dict,
        task_name=cfg_dict["name"],
        sim_device=sim_device,
        rl_device=rl_device,
        graphics_device_id=graphics_device_id,
        headless=headless,
        multi_gpu=multi_gpu,
        virtual_screen_capture=virtual_screen_capture,
        force_render=force_render,
    )
    return create_rlgpu_env()
