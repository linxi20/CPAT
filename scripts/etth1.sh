export CUDA_VISIBLE_DEVICES=5

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=336
model_name=CPAT

root_path_name=../dataset/ETT-small
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1

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
    --enc_in 7 \
    --n_heads 4 \
    --d_model 16 \
    --d_ff 32 \
    --dropout 0.3 \
    --fc_dropout 0.3 \
    --head_dropout 0 \
    --patch_len 4 \
    --stride 2 \
    --win_size 2 \
    --des 'Exp' \
    --train_epochs 100 \
    --patience 10 \
    --itr 1 --batch_size 128 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_96'.log 

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
    --enc_in 7 \
    --n_heads 4 \
    --d_model 16 \
    --d_ff 16 \
    --dropout 0.3 \
    --fc_dropout 0.3 \
    --head_dropout 0 \
    --patch_len 4 \
    --stride 2 \
    --win_size 2 \
    --des 'Exp' \
    --train_epochs 100 \
    --patience 10 \
    --itr 1 --batch_size 128 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_192'.log 

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
    --enc_in 7 \
    --n_heads 4 \
    --d_model 16 \
    --d_ff 32 \
    --dropout 0.3 \
    --fc_dropout 0.3 \
    --head_dropout 0 \
    --patch_len 4 \
    --stride 2 \
    --win_size 2 \
    --des 'Exp' \
    --train_epochs 100 \
    --patience 10 \
    --itr 1 --batch_size 128 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_336'.log 

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
    --enc_in 7 \
    --n_heads 4 \
    --d_model 16 \
    --d_ff 32 \
    --dropout 0.3 \
    --fc_dropout 0.3 \
    --head_dropout 0 \
    --patch_len 4 \
    --stride 2 \
    --win_size 2 \
    --des 'Exp' \
    --train_epochs 100 \
    --patience 10 \
    --itr 1 --batch_size 128 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_720'.log 