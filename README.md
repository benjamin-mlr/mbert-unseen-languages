# When Being Unseen from mBERT is just the Beginning: Handling New Languages With Multilingual Language Models 

This repository includes pointers and scripts to reproduce experiments presented in [When Being Unseen from mBERT is just the Beginning: Handling New Languages With Multilingual Language Models](https://arxiv.org/abs/2010.12858)

##  Transliteration 

## Linguistically motivated Transliteration 
 

### Uyghur to the Latin script  

Install pearl at https://www.perl.org/get.html and run:

`cat ug.txt | perl ./alTranscribe.pl -f ug -t tr  > ug_latin_script.txt`

### Sorani to the Latin script
  
`cat ckb.txt | perl ./alTranscribe.pl -f ckb -t ku > ckb_latin_script.txt`

### 
  
## Fine-tuning mBERT

We use the script  [run_language_modeling.py](https://github.com/huggingface/transformers/blob/master/examples/legacy/run_language_modeling.py)

### MLM 

 
```
export TRAIN_FILE=./train.txt
export TEST_FILE=./test.txt

python ./run_language_modeling.py \
    --output_dir=output \
    --model_type="bert" \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE \
    --per_gpu_train_batch_size 8 \
    --per_gpu_eval_batch_size 4 \
    --num_train_epochs 20 \
    --output_dir ./ \
    --evaluate_during_training 1 \
    --save_steps 1000 \
    --save_total_limit 2 \
    --mlm \
    --overwrite_output_dir \
    --block_size 128 \
    --line_by_line
```


### mBERT Unsupervised Fine Tuning


We use the script  [run_language_modeling.py](https://github.com/huggingface/transformers/blob/master/examples/legacy/run_language_modeling.py)
 
```
export TRAIN_FILE=./train.txt
export TEST_FILE=./test.txt

python ./run_language_modeling.py \
    --output_dir=output \
    --model_type="bert" \
    --model_name_or_path="bert-base-multilingual-cased" \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE \
    --per_gpu_train_batch_size 8 \
    --per_gpu_eval_batch_size 4 \
    --num_train_epochs 20 \
    --output_dir ./ \
    --evaluate_during_training 1 \
    --save_steps 1000 \
    --save_total_limit 2 \
    --mlm \
    --overwrite_output_dir \
    --block_size 128 \
    --line_by_line
```
