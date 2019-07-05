#!/bin/sh
    
#--infer_mode=beam_search \

CUDA_VISIBLE_DEVICES=3 python -m nmt.nmt \
    --out_dir=/home/yhboo/project/nprc_parctice/nmt/results/gnmt_model_ko_same_valid \
    --inference_input_file=./examples/eval_ex.en \
    --inference_output_file=./examples/eval_ex_beam10_all.ko \
    --infer_mode=beam_search \
    --beam_width=10 \
    --length_penalty_weight=0.0 \
    --coverage_penalty_weight=0.0 \
    --num_translations_per_input=10 \
    --sampling_temperature=0.5
