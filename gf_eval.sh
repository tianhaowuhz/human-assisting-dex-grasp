#####################################################################################################################################################################
#####################################################-------- GF Evaluation-------#########################################################################################
#####################################################################################################################################################################
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
