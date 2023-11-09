#####################################################################################################################################################################
#####################################################-------- GF Evaluation-------#########################################################################################
#####################################################################################################################################################################
# python ./Runners/EvalSDE_update.py \
# --log_dir test \
# --sde_mode vp \
# --mode train \
# --n_epoches 1 \
# --eval_freq 1 \
# --batch_size 16 \
# --lr 2e-4 \
# --t0 0.5 \
# --test_ratio 0.1 \
# --hidden_dim 1024 \
# --embed_dim 512 \
# --base_noise_scale 0.01 \
# --demo_nums 15135 \
# --num_envs=3027 \
# --demo_name=train_semcore_0.1_dc_100ss_5faronly_rel \
# --pointnet_network_type=new_3 \
# --device_id=5 \
# --sde_min=0.1 \
# --sde_max=10.0 \
# --model_name='/home/jiyao/tianhaowu/projects/ConDex/logs/train_new3_rel_01-10range_rt1_5farrel_riemann_direct_1ks_noaug_train_vp/score_6999.pt' \
# --relative \
# --space riemann \
# # --gt \

#eval
python ./Runners/EvalSDE_update.py \
--log_dir test \
--sde_mode vp \
--mode eval \
--dataset_type unseencategory \
--n_epoches 1 \
--eval_freq 1 \
--batch_size 16 \
--lr 2e-4 \
--t0 0.5 \
--test_ratio 0.1 \
--hidden_dim 1024 \
--embed_dim 512 \
--base_noise_scale 0.01 \
--demo_nums 7375 \
--num_envs=1475 \
--demo_name=unseencategory_ddgmujoco_1497 \
--pointnet_network_type=new_3 \
--device_id=0 \
--space riemann \
--model_name='/home/jiyao/tianhaowu/projects/ConDex/logs/train_new3_rel_01-10range_rt1_5farrel_riemann_1ks_noaug_train_pt_vp/score_14999.pt' \
--relative \
# --gt \

# python ./Runners/EvalSDE_update.py \
# --log_dir unseen_new3_vp \
# --sde_mode vp \
# --mode train \
# --n_epoches 1 \
# --eval_freq 1 \
# --batch_size 16 \
# --lr 2e-4 \
# --t0 0.5 \
# --test_ratio 0.1 \
# --hidden_dim 1024 \
# --embed_dim 512 \
# --base_noise_scale 0.01 \
# --demo_nums 3127 \
# --num_envs=3127 \
# --demo_name=train_semcore_0.1_dc_100ss_1random \
# --relative \
# --pointnet_network_type=new_3 \
# --model_name='/home/jiyao/tianhaowu/projects/ConDex/Models/train_new3_5far_rel_01-20range_vp/score.pt' \
# --device_id=1 \
# --gt \
# # --hand_pcl \

# CUDA_VISIBLE_DEVICES=0 python ./Runners/EvalSDE.py \
# --n_epoches 1 \
# --eval_freq 1 \
# --batch_size 4 \
# --lr 2e-4 \
# --t0 0.5 \
# --test_ratio 0.1 \
# --hidden_dim 1024 \
# --embed_dim 512 \
# --base_noise_scale 0.01 \
# --demo_nums 260 \
# --num_envs=52 \
# --demo_name=eval_260_demo \
# --log_dir=1000train_260eval \
