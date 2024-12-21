export CUDA_VISIBLE_DEVICES=5

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=104
model_name=CPAT

root_path_name=../dataset/illness
data_path_name=national_illness.csv
model_id_name=national_illness

python -u ../run.py \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_24' \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --pred_len 24 \
    --enc_in 7 \
    --n_heads 4 \
    --d_model 16 \
    --d_ff 16 \
    --dropout 0.3 \
    --fc_dropout 0.3 \
    --head_dropout 0 \
    --patch_len 8 \
    --stride 2\
    --win_size 4 \
    --des 'Exp' \
    --train_epochs 100 \
    --patience 20\
    --lradj 'constant' \
    --itr 1 --batch_size 16 --learning_rate 0.0025 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_24'.log 

python -u ../run.py \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_36' \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --pred_len 36 \
    --enc_in 7 \
    --n_heads 4 \
    --d_model 16 \
    --d_ff 32 \
    --dropout 0.3 \
    --fc_dropout 0.3 \
    --head_dropout 0 \
    --patch_len 4 \
    --stride 2 \
    --win_size 4 \
    --des 'Exp' \
    --train_epochs 100 \
    --patience 20 \
    --lradj 'constant' \
    --itr 1 --batch_size 16 --learning_rate 0.0025 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_36'.log 

python -u ../run.py \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_48' \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --pred_len 48 \
    --enc_in 7 \
    --n_heads 4 \
    --d_model 16 \
    --d_ff 128 \
    --dropout 0.3 \
    --fc_dropout 0.3 \
    --head_dropout 0 \
    --patch_len 8 \
    --stride 2 \
    --win_size 8 \
    --des 'Exp' \
    --train_epochs 100 \
    --patience 20 \
    --lradj 'constant' \
    --itr 1 --batch_size 16 --learning_rate 0.0025 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_48'.log 

python -u ../run.py \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_60' \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --pred_len 60 \
    --enc_in 7 \
    --n_heads 4 \
    --d_model 16 \
    --d_ff 128 \
    --dropout 0.3 \
    --fc_dropout 0.3 \
    --head_dropout 0 \
    --patch_len 24\
    --stride 2 \
    --win_size 4 \
    --des 'Exp' \
    --train_epochs 100 \
    --patience 20 \
    --lradj 'constant' \
    --itr 1 --batch_size 16 --learning_rate 0.0025 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_60'.log 