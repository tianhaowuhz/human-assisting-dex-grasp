#####################################################################################################################################################################
#####################################################-------- RL Train-------#########################################################################################
#####################################################################################################################################################################
python ./Runners/TrainGFPPO.py \
--seed=0 \
--headless \
--num_envs=100 \
--dataset_type='train' \
--score_model_path='/home/jiyao/tianhaowu/projects/human-assisting-dex-grasp/Ckpt/gf' \
--t0=0.005 \
--exp_name="ours" \
--run_device_id=0 \
--constrained \
