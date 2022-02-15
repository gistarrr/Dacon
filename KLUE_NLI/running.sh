python train.py \
--model_name_or_path klue/roberta-large \
--do_train \
--do_eval \
--output_dir ./results \
--save_total_limit 3 \
--save_strategy epoch \
--num_train_epochs 5 \
--learning_rate 3e-5 \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 32 \
--gradient_accumulation_steps 1 \
--evaluation_strategy epoch \
--load_best_model_at_end True \
--metric_for_best_model accuracy \
--no_cuda True \
--use_SIC True


# --warmup_steps 10000 \
# --weight_decay 1e-2 \
# --adam_beta1  0.9 \
# --adam_beta2  0.999 \
# --adam_epsilon 1e-06 \
#--label_smoothing_factor 0.1