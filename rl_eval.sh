#-----------------------------------------------------------------------------------------------------------------------
#------------------------------------------- Evaluation  -------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
# unseen 1298  seen 519
# ours
python ./Runners/EvalGFPPO.py \
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
