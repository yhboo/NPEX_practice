# NPEX_practice

original source codes in:
https://github.com/tensorflow/nmt

# Inference with model

python -m nmt.nmt \
    --out_dir=MODEL_PATH \
    --inference_input_file=./examples/eval_ex.en \
    --inference_output_file=./examples/eval_ex_beam10_all.ko \
    --infer_mode=beam_search \
    --beam_width=10 \
    --num_translations_per_input=10 \
    --sampling_temperature=0.5



# EN_VI translation


//Download en-vi translation dataset

nmt/scripts/download_iwslt15.sh PATH/nmt_data


//train GNMT

mkdir PATH/nmt_model

python -m nmt.nmt \
    --attention_architecture=gnmt \
    --attention=bahdanau \
    --encoder_type=gnmt \
    --src=en --tgt=vi \
    --vocab_prefix=PATH/nmt_data/vocab  \
    --train_prefix=PATH/nmt_data/train \
    --dev_prefix=PATH/nmt_data/tst2012  \
    --test_prefix=PATH/nmt_data/tst2013 \
    --out_dir=PATH/nmt_model \
    --num_train_steps=12000 \
    --steps_per_stats=100 \
    --num_layers=3 \
    --num_units=256 \
    --dropout=0.2 \
    --metrics=bleu

# EN_KO translation
//KAIST en-ko translation dataset will be used
http://semanticweb.kaist.ac.kr/home/index.php/KAIST_Corpus


//train GNMT

mkdir PATH/nmt_model_ko

python -m nmt.nmt \
    --attention_architecture=gnmt \
    --attention=bahdanau \
    --encoder_type=gnmt \
    --src=en --tgt=ko \
    --vocab_prefix=PATH/nmt_data_KAIST/vocab  \
    --train_prefix=PATH/nmt_data_KAIST/train \
    --dev_prefix=PATH/nmt_data_KAIST/valid  \
    --test_prefix=PATH/nmt_data_KAIST/test \
    --out_dir=PATH/nmt_model_ko \
    --num_train_steps=12000 \
    --steps_per_stats=100 \
    --num_layers=3 \
    --num_units=256 \
    --dropout=0.2 \
    --metrics=bleu
