

#  Fine Tuning BERT 

This project includes all the code needed to train and use trained models for Dependancy Parsing, Part-of-Speech Tagging built in fine-tuning BERT.
 
## Project Setting up 

### env 

contains all the environment variables, directories, flags that should be adapte for a given environment 

`sh ./install.sh`


`conda activate transfer` 


## Download pretrained model

For checkpoint compatibility reasons, we do not use the standart `from_pretrained` method function from Hugging-Face. Therefore, we download manually the checkpoints.


In `./transfer/downstream/pretrained` download checkpoints and unzip: 

- for Multilingual BERT: https://drive.google.com/file/d/1QVazXC8p3JttjepLkpYyRxU2Gz-rPjWN/view?usp=sharing      
- for CamemBERT: https://drive.google.com/file/d/1v69-oDZiIEodKoLMjxOR9Lqf1meIEATr/view?usp=sharing    


## Fine-tuning 


```
ython ./transfer/downstream/finetune/train.py  

 python ./transfer/downstream/finetune/train.py  
 --tasks parsing 
 --train_path ./train.conllu 
 --dev_path ./dev.conllu 
 --test_paths ./test.conllu
 --hidden_dropout_prob 0.1 
 --batch_size 6 
 --optimizer AdamW 
 --low_memory_foot_print_batch_mode 0 
 --bert_model bert_base_multilingual_cased  # in [bert_base_multilingual_cased, camembert]
 --lr 5e-05 --epochs 30 
 --dropout_classifier 0.2 --weight_decay 0.0 
 --seed 1 
 --fine_tuning_strategy standart 
 --multi_task_loss_ponderation pos-pos=1.0,parsing-types=1,parsing-heads=1,mlm-wordpieces_inputs_words=1.0, 
 --overall_report_dir ./transfer//downstream/finetune/checkpoints/0101-report
  --overall_label 001-model
 --model_id_pref 001-model
```

