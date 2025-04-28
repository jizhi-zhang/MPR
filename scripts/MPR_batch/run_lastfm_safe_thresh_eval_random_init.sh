top_k=5
gpu_id=0
task_type="Lastfm-360K"
epoch=200
for seed in 1
do 
for learning_rate in 1e-3
do
for weight_decay in 1e-7
do
for fair_reg in 1e-1
do
for partial_ratio_male in  0.5
do
for partial_ratio_female in 0.5 0.4 0.3 0.2 0.1
do
for gender_train_epoch in 1000
do
for beta in 0.005
do 

main_folder=./MPR_${task_type}_EXP_seed${seed}_rmse_thresh_eval_batch_random_init/change_ratio_and_epoch/partial_ratio_male${partial_ratio_male}/partial_ratio_female_${partial_ratio_female}/gender_train_epoch_${gender_train_epoch}/
mkdir -p ${main_folder}learning_rate_${learning_rate}_weight_decay_${weight_decay}_fair_reg_${fair_reg}_beta_${beta}
nohup python3 -u mpr.py --gpu_id ${gpu_id} --learning_rate $learning_rate --partial_ratio_male $partial_ratio_male --partial_ratio_female $partial_ratio_female \
--gender_train_epoch $gender_train_epoch --weight_decay $weight_decay --fair_reg $fair_reg --beta $beta --task_type ${task_type} \
--saving_path ${main_folder}learning_rate_${learning_rate}_weight_decay_${weight_decay}_fair_reg_${fair_reg}_beta_${beta}/ \
--orig_unfair_model /NAS/zhangjz/MPR_diff_seed_construct_batch/pretrained_model/${task_type}/MF_orig_model \
--num_epochs $epoch \
--result_csv ${main_folder}result.csv --seed ${seed} --data_path ./datasets/${task_type}/ \
--predict_sst_path ./predict_sst_diff_seed_batch/ > ${main_folder}learning_rate_${learning_rate}_weight_decay_${weight_decay}_fair_reg_${fair_reg}_beta_${beta}/train.log 2>&1
done 
done 
done
done
done
done
done
done
