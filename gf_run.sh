#####################################################################################################################################################################
#####################################################-------- Filter Grasp-------#########################################################################################
#####################################################################################################################################################################
#train
# python ./Runners/FilterGrasp.py \
# --num_envs=3251 \
# --demo_name=train_semcore_0.1_dc \
# --mode=train \
# --close_dis=0.1 \
# --filter_time=1 \
# --num_required_object=5000 \
# --num_required_grasp_pose=300 \
# --filter_thr=0.01 \
# --hand_step_number=40 \
# --filter_mode='state' \
# --pregrasp_coff=0.9

# python ./Runners/FilterGrasp.py \
# --num_envs=1363 \
# --demo_name=unseencategory_ddgmujoco \
# --mode=train \
# --close_dis=0.01 \
# --filter_time=1 \
# --num_required_object=5000 \
# --num_required_grasp_pose=300 \
# --filter_thr=0.01 \
# --hand_step_number=40 \
# --filter_mode='state' \
# --pregrasp_coff=0.9

# eval
# python ./Runners/FilterGrasp.py \
# --num_envs=130 \
# --demo_name=eval_ycbbg_dgn_gpu_0.08scale_0.01_state_4791 \
# --mode=eval \
# --close_dis=0.01 \
# --filter_time=1 \
# --num_required_object=130 \
# --num_required_grasp_pose=40 \
# --filter_thr=0.01 \
# --hand_step_number=40 \
# --filter_mode='state' \
# --pregrasp_coff=0.9
# # --gui \

#####################################################################################################################################################################
#####################################################-------- GF Train-------#########################################################################################
#####################################################################################################################################################################


#-----------------------------------------------------------------------------------------------------------------------
#------------------------------------------- Dataset  -------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

# CUDA_VISIBLE_DEVICES=0 python ./Runners/DataSDE.py \
# --num_envs=4 \
# --demo_nums 8 \
# --demo_name=8_demo \
# --mode=train \ 



#-----------------------------------------------------------------------------------------------------------------------
#------------------------------------------- Training  -------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
python ./Runners/TrainSDE_update.py \
--log_dir gf \
--sde_mode vp \
--batch_size 3077 \
--lr 2e-4 \
--t0 0.5 \
--train_model \
--demo_nums 15387 \
--num_envs=3027 \
--demo_name=train_semcore_0.1_dc_100ss_5far_rel_rc \
--eval_demo_name=train_semcore_0.1_dc_100ss_5faronly_rel \
--device_id=0 \
--mode train \
--dataset_type train \
--relative \
--space riemann \

# python ./Runners/TrainSDE_update.py \
# --log_dir train_new3_rel_01-10range_rt1_5farrel_riemann_1ks_noaug_eval_vp \
# --sde_mode vp \
# --n_epoches 50000 \
# --eval_freq 1000 \
# --batch_size 3027 \
# --lr 2e-4 \
# --t0 0.5 \
# --sde_min 0.1 \
# --sde_max 10.0 \
# --test_ratio 0.1 \
# --hidden_dim 1024 \
# --embed_dim 512 \
# --base_noise_scale 0.01 \
# --train_model \
# --demo_nums 15387 \
# --num_envs=1475 \
# --demo_name=train_semcore_0.1_dc_100ss_5far_rel_rc \
# --eval_demo_name=unseencategory_ddgmujoco_1497 \
# --pointnet_network_type=new_3 \
# --device_id=2 \
# --repeat_num=1 \
# --mode eval \
# --relative \
# --space riemann \
# # --gf_rot \

# python ./Runners/TrainSDE_update.py \
# --log_dir test \
# --sde_mode vp \
# --n_epoches 50000 \
# --eval_freq 500 \
# --batch_size 301 \
# --lr 2e-4 \
# --t0 0.5 \
# --sde_min 0.1 \
# --sde_max 25.0 \
# --test_ratio 0.1 \
# --hidden_dim 1024 \
# --embed_dim 512 \
# --base_noise_scale 0.01 \
# --train_model \
# --demo_nums 301 \
# --num_envs=301 \
# --demo_name=train_semcore_0.1_dc_100ss_1far_301_rc \
# --eval_demo_name=train_semcore_0.1_dc_100ss_1far_301 \
# --pointnet_network_type=new_3 \
# --device_id=2 \
# --mode train \
# --space euler \
# --relative \
#####################################################################################################################################################################
#####################################################-------- GF Evaluation-------#########################################################################################
#####################################################################################################################################################################
# python ./Runners/EvalSDE_update.py \
# --log_dir new3_vp \
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
# --demo_name=train_semcore_0.1_dc_100ss_5faronly \
# --pointnet_network_type=new_3 \
# --device_id=3 \
# --sde_min=0.01 \
# --sde_max=10.0 \
# --model_name='/home/jiyao/tianhaowu/projects/ConDex/logs/train_new3_rel_001-10range_rt1_5faronly_riemann_vp/score_1599.pt' \
# --relative \
# # --gt \

#eval
# python ./Runners/EvalSDE_update.py \
# --log_dir unseen_new3_vp \
# --sde_mode vp \
# --mode eval \
# --n_epoches 1 \
# --eval_freq 1 \
# --batch_size 16 \
# --lr 2e-4 \
# --t0 0.5 \
# --test_ratio 0.1 \
# --hidden_dim 1024 \
# --embed_dim 512 \
# --base_noise_scale 0.01 \
# --demo_nums 6765 \
# --num_envs=1353 \
# --demo_name=unseencategory_ddgmujoco_new \
# --relative \
# --pointnet_network_type=new_3 \
# --device_id=1 \
# --model_name='/home/jiyao/tianhaowu/projects/ConDex/logs/train_new3_rel_01-10range_rt1_5faronly_riemann_vp/score_5999.pt' \
# # --gt \

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
