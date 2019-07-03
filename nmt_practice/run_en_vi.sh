#!/bin/sh

CUDA_VISIBLE_DEVICES=3 python -m nmt.nmt \
    --src=en --tgt=vi \
    --attention=bahdanau \
    --attention_architecture=gnmt \
    --encoder_type=gnmt \
    --vocab_prefix=/home/yhboo/project/nprc_parctice/nmt/tmp/nmt_data_en_vi/vocab  \
    --train_prefix=/home/yhboo/project/nprc_parctice/nmt/tmp/nmt_data_en_vi/train \
    --dev_prefix=/home/yhboo/project/nprc_parctice/nmt/tmp/nmt_data_en_vi/tst2012  \
    --test_prefix=/home/yhboo/project/nprc_parctice/nmt/tmp/nmt_data_en_vi/tst2013 \
    --out_dir=/home/yhboo/project/nprc_parctice/nmt/results/gnmt_model_vi \
    --num_train_steps=12000 \
    --steps_per_stats=100 \
    --num_layers=3 \
    --num_units=256 \
    --dropout=0.2 \
    --metrics=bleu
