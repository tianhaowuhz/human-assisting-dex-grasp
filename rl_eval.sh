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

# python ./Runners/TrainGFPPO.py \
# --seed=0 \
# --constrained \
# --headless \
# --num_envs=350 \
# --score_model_path="/home/jiyao/tianhaowu/projects/ConDex/Models/train_all_350*40_vp" \
# --t0=0.1 \
# --exp_name="ours" \
# --run_device_id=4 \
# --model_dir="/home/jiyao/tianhaowu/projects/ConDex/logs/gfppo/t0:0.1_sfn:5_350ne_350obj_gpt:norm+scale_gs:1.0_at:joint_rt:std_similarity+tsifsim+tsanysim+collision+height_sr10_sensor_coll0.1_0.1as_tgradgoal_2sigma_simfreq:2_seed0/model_800.pt" \
# --experiment=humantraj5s_256_std_similarity+t2fcontact_250ne_5obj \

#-----------------------------------------------------------------------------------------------------------------------
#------------------------------------------- Evaluation  -------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
# unseen 1298  seen 519

# python ./Runners/EvalPPO.py \
# --constrained \
# --headless \
# --num_envs=519 \
# --dataset_type='seencategory' \
# --score_model_path="/home/jiyao/tianhaowu/projects/ConDex/Models/train_new3_rel_01-10range_rt1_5farrel_riemann_1ks_noaug_vp" \
# --t0=0.005 \
# --run_device_id=6 \
# --mode='eval' \
# --eval_times=5 \
# --seed=101112 \
# --exp_name="ilad" \
# --eval_name="ilad_unseen" \
# --model_dir="/home/jiyao/tianhaowu/projects/ConDex/Models/evaluation/ilad/handrot:True_t0:0.005_sfn:5_100ne_3127obj_gpt:norm+scale_gs:1.0_at:direct_subat:none_rt:abscomplexfingertip_dis+height+sr_rn:False_simfreq:5_cd:0.1_pts:pt_seed101112/model_2000.pt" \

# ours
python ./Runners/EvalPPO.py \
--constrained \
--headless \
--num_envs=1298 \
--dataset_type='unseencategory' \
--score_model_path="Ckpt/gf" \
--t0=0.005 \
--run_device_id=0 \
--mode='eval' \
--eval_times=5 \
--seed=0 \
--exp_name="ours" \
--eval_name="ours_unseen" \
--model_dir="Ckpt/gfppo.pt" \

# rl
# python ./Runners/EvalPPO.py \
# --constrained \
# --headless \
# --num_envs=519 \
# --dataset_type='seencategory' \
# --score_model_path="/home/jiyao/tianhaowu/projects/ConDex/Models/train_new3_rel_01-10range_rt1_5farrel_riemann_1ks_noaug_vp" \
# --t0=0.005 \
# --run_device_id=2 \
# --mode='eval' \
# --eval_times=5 \
# --seed=456 \
# --exp_name="rl" \
# --eval_name="rl_seen" \
# --model_dir="/home/jiyao/tianhaowu/projects/ConDex/Models/evaluation/rl/handrot:True_t0:0.005_sfn:5_100ne_3127obj_gpt:norm+scale_gs:1.0_at:direct_subat:none_rt:abscomplexfingertip_dis+height+sr_faketendon+175hp_rn:False_simfreq:5_cd:0.1_pts:pt2_seed456/model_2000.pt" \

#goalrl
# python ./Runners/EvalPPO.py \
# --constrained \
# --headless \
# --num_envs=519 \
# --dataset_type='seencategory' \
# --score_model_path="/home/jiyao/tianhaowu/projects/ConDex/Models/train_new3_rel_01-10range_rt1_5farrel_riemann_1ks_noaug_vp" \
# --t0=0.005 \
# --run_device_id=2 \
# --mode='eval' \
# --eval_times=5 \
# --seed=456 \
# --exp_name="goalrl" \
# --eval_name="goalrl_seen" \
# --model_dir="/home/jiyao/tianhaowu/projects/ConDex/Models/evaluation/goalrl/handrot:True_t0:0.005_sfn:5_100ne_3127obj_gpt:norm+scale_gs:1.0_at:direct_subat:none_rt:abscomplexfingertip_dis+goaldist+height+sr_tg_faketendon_rn:False_simfreq:5_cd:0.1_pts:pt2_seed456/model_2000.pt"