#####################################################################################################################################################################
#####################################################-------- Data Gen-------#########################################################################################
#####################################################################################################################################################################

# python ./Runners/GenWristPose.py \
# --t0=0.5 \
# --test_ratio=0.1 \
# --hidden_dim=1024 \
# --base_noise_scale=0.01 \
# --embed_dim=512 \
# --num_envs=421 \
# --relative \
# --demo_name=train_cdis0.3_t1_all_ground_nobase \
# # --move_hand \
# # --gui \
#####################################################################################################################################################################
#####################################################-------- RL Train-------#########################################################################################
#####################################################################################################################################################################

#-----------------------------------------------------------------------------------------------------------------------
#------------------------------------------- Training  -------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

python ./Runners/TrainGFPPO.py \
--seed=456 \
--headless \
--num_envs=100 \
--dataset_type='train' \
--score_model_path='/home/jiyao/tianhaowu/projects/ConDex/Models/train_new3_rel_01-10range_rt1_5farrel_riemann_1ks_noaug_vp' \
--t0=0.005 \
--exp_name="ours" \
--run_device_id=0 \
--constrained \
# --model_dir="/home/jiyao/tianhaowu/projects/ConDex/logs/ours/05-10-08-13_handrot:True_t0:0.005_sfn:5_100ne_3127obj_gpt:norm+scale_gs:1.0_at:direct_subat:add+jointscale_rt:ori_similarity+height+sr_faketendon+1osr+175h_normriemann+26999pt_rn:False_simfreq:5_cd:0.1_pts:pt_seed12/model_2750.pt" \
# --experiment=humantraj5s_256_std_similarity+t2fcontact_250ne_5obj \

#-----------------------------------------------------------------------------------------------------------------------
#------------------------------------------- Evaluation  -------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

# python ./Runners/EvalPPO.py \
# --constrained \
# --headless \
# --num_envs=106 \
# --score_model_path="/home/jiyao/tianhaowu/projects/ConDex/Models/train_all_350*40_vp" \
# --t0=0.1 \
# --run_device_id=4 \
# --mode='eval' \
# --eval_times=40 \
# --seed=0 \
# --exp_name="ours" \
# --eval_name="ours" \
# --model_dir="/home/jiyao/tianhaowu/projects/ConDex/logs/ours/handrot:True_t0:0.1_sfn:5_350ne_350obj_gpt:norm+scale_gs:0.4_at:joint_subat:add_rt:ori_similarity+sr_simfreq:5_seed0/model_200.pt" \

# --exp_name="ours" \
# --eval_name="wosim" \
# --model_dir="/home/jiyao/tianhaowu/projects/ConDex/logs/ours/handrot:True_t0:0.1_sfn:5_350ne_350obj_gpt:norm+scale_gs:0.5_at:joint_subat:add_rt:sr_simfreq:5_seed456/model_200.pt" \

# --exp_name="gf" \
# --eval_name="gf" \

# --exp_name="goalrl" \
# --eval_name="goalrl" \
# --model_dir="/home/jiyao/tianhaowu/projects/ConDex/logs/goalrl/handrot:True_t0:0.1_sfn:5_350ne_350obj_gpt:norm+scale_gs:0.5_at:direct_subat:none_rt:fingertip_dis+goaldist+sr_simfreq:5_seed456/model_200.pt" \

# --exp_name="rl" \
# --eval_name="rl" \
# --model_dir="/home/jiyao/tianhaowu/projects/ConDex/logs/rl/handrot:True_t0:0.1_sfn:5_350ne_350obj_gpt:norm+scale_gs:0.5_at:direct_subat:none_rt:fingertip_dis+sr_simfreq:5_seed456/model_200.pt" \

# --exp_name="ours" \
# --eval_name="ours" \
# --model_dir="/home/jiyao/tianhaowu/projects/ConDex/logs/ours/handrot:True_t0:0.1_sfn:5_350ne_350obj_gpt:norm+scale_gs:0.5_at:joint_subat:add_rt:ori_similarity+sr_simfreq:5_seed456/model_200.pt" \

# ours
#--seed=0 \
#--model_dir="/home/jiyao/tianhaowu/projects/ConDex/logs/ours/handrot:True_t0:0.1_sfn:5_350ne_350obj_gpt:norm+scale_gs:0.5_at:joint_subat:add_rt:ori_similarity+sr_simfreq:5_seed0/model_100.pt" \

# --model_dir="/home/jiyao/tianhaowu/projects/ConDex/logs/gfppo/t0:0.1_sfn:5_350ne_350obj_gpt:norm+scale_gs:1.0_at:joint_rt:std_similarity+tsifsim+tsanysim+collision+height_sr10_sensor_coll0.1_0.1as_tgradgoal_2sigma_simfreq:2_seed0/model_800.pt" \
# --experiment=humantraj5s_256_std_similarity+t2fcontact_250ne_5obj \