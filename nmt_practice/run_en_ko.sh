#!/bin/sh

CUDA_VISIBLE_DEVICES=3 python -m nmt.nmt \
    --src=en --tgt=ko \
    --attention=bahdanau \
    --attention_architecture=gnmt \
    --encoder_type=gnmt \
    --vocab_prefix=/home/yhboo/project/nprc_parctice/nmt/tmp/nmt_data_KAIST/vocab  \
    --train_prefix=/home/yhboo/project/nprc_parctice/nmt/tmp/nmt_data_KAIST/train \
    --dev_prefix=/home/yhboo/project/nprc_parctice/nmt/tmp/nmt_data_KAIST/valid  \
    --test_prefix=/home/yhboo/project/nprc_parctice/nmt/tmp/nmt_data_KAIST/test \
    --out_dir=/home/yhboo/project/nprc_parctice/nmt/results/gnmt_model_ko_same_valid \
    --num_train_steps=20000 \
    --steps_per_stats=1000 \
    --num_layers=3 \
    --num_units=256 \
    --dropout=0.2 \
    --metrics=bleu
