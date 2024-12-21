export CUDA_VISIBLE_DEVICES=5

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=336
model_name=CPAT

root_path_name=../dataset/traffic
data_path_name=traffic.csv
model_id_name=traffic
data_name=custom

python -u ../run.py \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_96' \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --pred_len 96 \
    --enc_in 862 \
    --n_heads 4 \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.2 \
    --fc_dropout 0.2 \
    --head_dropout 0 \
    --patch_len 16 \
    --stride 2 \
    --win_size 2 \
    --des 'Exp' \
    --train_epochs 100 \
    --patience 10 \
    --lradj 'TST' \
    --pct_start 0.2 \
    --itr 1 --batch_size 4 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_96'.log 

python -u ../run.py \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_192' \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --pred_len 192 \
    --enc_in 862 \
    --n_heads 4 \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.2 \
    --fc_dropout 0.2 \
    --head_dropout 0 \
    --patch_len 16 \
    --stride 2 \
    --win_size 2 \
    --des 'Exp' \
    --train_epochs 100 \
    --patience 10 \
    --lradj 'TST' \
    --pct_start 0.2 \
    --itr 1 --batch_size 4 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_192'.log 

python -u ../run.py \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_336' \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --pred_len 336 \
    --enc_in 862 \
    --n_heads 4 \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.2 \
    --fc_dropout 0.2 \
    --head_dropout 0 \
    --patch_len 16 \
    --stride 2 \
    --win_size 2 \
    --des 'Exp' \
    --train_epochs 100 \
    --patience 10 \
    --lradj 'TST' \
    --pct_start 0.2 \
    --itr 1 --batch_size 4 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_336'.log 

python -u ../run.py \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_720' \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --pred_len 720 \
    --enc_in 862 \
    --n_heads 4 \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.2 \
    --fc_dropout 0.2 \
    --head_dropout 0 \
    --patch_len 8 \
    --stride 4 \
    --win_size 4 \
    --des 'Exp' \
    --train_epochs 100 \
    --patience 10 \
    --lradj 'TST' \
    --pct_start 0.2 \
    --itr 1 --batch_size 4 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_720'.log 