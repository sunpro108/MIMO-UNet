CUDA_VISIBLE_DEVICES=1 python main.py \
--model_name dwt \
--mode train \
--subset Hday2night \
--valid_freq 20 \
--learning_rate 1e-3 \