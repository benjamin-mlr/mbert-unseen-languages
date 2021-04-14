# When Being Unseen from mBERT is just the Beginning: Handling New Languages With Multilingual Language Models 

This repository includes pointers and scripts to reproduce experiments presented in [When Being Unseen from mBERT is just the Beginning: Handling New Languages With Multilingual Language Models](https://arxiv.org/abs/2010.12858)   (accepted to [NAACL-HLT 2021](https://2021.naacl.org/))

  
## Fine-tuning mBERT

### Data

#### Raw data for MLM training/unsupervised fine-tuning

Download [OSCAR](https://oscar-corpus.com/) deduplicated datasets. 


### MLM training 

We use the script [run_language_modeling.py](https://github.com/huggingface/transformers/blob/master/examples/legacy/run_language_modeling.py) from [Hugging-Face](https://huggingface.co/transformers/)

 
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


Similarly [run_language_modeling.py](https://github.com/huggingface/transformers/blob/master/examples/legacy/run_language_modeling.py) from [Hugging-Face](https://huggingface.co/transformers/)
 
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



##  Transliteration 

## Linguistically motivated Transliteration 
 

### Uyghur to the Latin script  

Install pearl at https://www.perl.org/get.html and run:


`cat ug.txt | perl ./alTranscribe.pl -f ug -t tr  > ug_latin_script.txt`

### Sorani to the Latin script
  
`cat ckb.txt | perl ./alTranscribe.pl -f ckb -t ku > ckb_latin_script.txt`


# How to cite 

If you extend or use this work, please cite:

```
@misc{muller2020unseen,
      title={When Being Unseen from mBERT is just the Beginning: Handling New Languages With Multilingual Language Models}, 
      author={Benjamin Muller and Antonis Anastasopoulos and Benoît Sagot and Djamé Seddah},
      year={2020},
      eprint={2010.12858},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```