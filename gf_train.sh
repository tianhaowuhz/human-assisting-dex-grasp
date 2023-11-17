#####################################################################################################################################################################
#####################################################-------- GF Train-------#########################################################################################
#####################################################################################################################################################################
python ./Runners/TrainSDE_update.py \
--log_dir gf \
--sde_mode vp \
--batch_size 3077 \
--lr 2e-4 \
--t0 0.5 \
--train_model \
--demo_nums 15387 \
--num_envs=3027 \
--demo_name=train_gf \
--eval_demo_name=train_eval \
--device_id=0 \
--mode train \
--dataset_type train \
--relative \
--space riemann \
