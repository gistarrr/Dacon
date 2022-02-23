python train.py \
--data_name train_data.csv \
--aeda True \
--run_name Explainable_2ep_Kfold \
--save_path ./checkpoints/Explainable_2ep_Kfold \
--model_name_or_path klue/roberta-large \
--do_train \
--do_eval \
--output_dir /content/results \
--overwrite_output_dir True \
--save_total_limit 5 \
--save_strategy steps \
--num_train_epochs 2 \
--learning_rate 3e-5 \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 32 \
--gradient_accumulation_steps 1 \
--evaluation_strategy steps \
--logging_steps 100 \
--eval_steps 250 \
--save_steps 250 \
--load_best_model_at_end True \
--metric_for_best_model accuracy \
--use_SIC True \
--no_cuda True

# python train.py \
# --data_name full_train_data.csv \
# --run_name Explainable_test2 \
# --model_name_or_path klue/roberta-large \
# --do_train \
# --do_eval \
# --save_path ./checkpoints/Explainable_test \
# --output_dir ./checkpoints/test \
# --overwrite_output_dir True \
# --save_total_limit 5 \
# --save_strategy steps \
# --num_train_epochs 2 \
# --learning_rate 3e-5 \
# --per_device_train_batch_size 16 \
# --per_device_eval_batch_size 32 \
# --gradient_accumulation_steps 1 \
# --evaluation_strategy steps \
# --logging_steps 1 \
# --eval_steps 250 \
# --load_best_model_at_end True \
# --metric_for_best_model accuracy \
# --use_SIC \
# --no_cuda True



# --warmup_steps 10000 \
# --weight_decay 1e-2 \
# --adam_beta1  0.9 \
# --adam_beta2  0.999 \
# --adam_epsilon 1e-06 \
#--label_smoothing_factor 0.1

# python inference.py \
# --save_path ./checkpoints/krl_3ep \
# --output_name krl_3ep.csv \
# --do_predict \
# --output_dir ./results \
# --per_device_eval_batch_size 32 \
# --no_cuda True

# python inference.py \
# --save_path ./checkpoints/krl_2ep \
# --output_name krl_2ep.csv \
# --do_predict \
# --output_dir ./results \
# --per_device_eval_batch_size 32 \
# --no_cuda True

# python inference.py \
# --save_path ./checkpoints/Explainable_5ep \
# --use_SIC \
# --output_name Explainable_5ep.csv \
# --do_predict \
# --output_dir ./results \
# --per_device_eval_batch_size 32 \
# --no_cuda True