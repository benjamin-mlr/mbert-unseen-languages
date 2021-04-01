import os
from transfer.downstream.finetune.env.imports import os, OrderedDict
from transfer.downstream.finetune.env.dir.project_directories import BERT_MODELS_DIRECTORY

assert os.path.isdir(BERT_MODELS_DIRECTORY), \
    "ERROR : {} does not exist : it should host the bert models tar.gz and vocabulary ".format(BERT_MODELS_DIRECTORY)


BERT_MODEL_DIC = {

                   "bert-base-multilingual-cased-bm":
                     {
                      "model": os.path.join(os.environ.get("PRETRAINED", "."), "tuned/10928823-bm-bert-base-multilingual-cased"),
                      "encoder": "BertModel",
                      "tokenizer": "BertTokenizer",
                      "vocab": os.path.join(BERT_MODELS_DIRECTORY, "bert-base-multilingual-cased-vocab.txt"),                      
                      "vocab_size": 119547,
                      "state_dict_mapping": {"bert": "encoder", "cls": "head.mlm"},
                      },

                   "bert-base-multilingual-cased-erzya_latin":
                     {
                      "model": os.path.join(os.environ.get("PRETRAINED", "."), "tuned/10923449--bert-base-multilingual-cased"),
                      "encoder": "BertModel",
                      "tokenizer": "BertTokenizer",
                      "vocab": os.path.join(BERT_MODELS_DIRECTORY, "bert-base-multilingual-cased-vocab.txt"),                      
                      "vocab_size": 119547,
                      "state_dict_mapping": {"bert": "encoder", "cls": "head.mlm"},
                      },

                  "bert-base-multilingual-cased-erzya":
                     {
                      "model": os.path.join(os.environ.get("PRETRAINED", "."), "tuned/10923447--bert-base-multilingual-cased"),
                      "encoder": "BertModel",
                      "tokenizer": "BertTokenizer",
                      "vocab": os.path.join(BERT_MODELS_DIRECTORY, "bert-base-multilingual-cased-vocab.txt"),                      
                      "vocab_size": 119547,
                      "state_dict_mapping": {"bert": "encoder", "cls": "head.mlm"},
                      },

                  "bert-base-multilingual-cased-buryat":
                     {
                      "model": os.path.join(os.environ.get("PRETRAINED", "."), "tuned/10923446--bert-base-multilingual-cased"),
                      "encoder": "BertModel",
                      "tokenizer": "BertTokenizer",
                      "vocab": os.path.join(BERT_MODELS_DIRECTORY, "bert-base-multilingual-cased-vocab.txt"),                      
                      "vocab_size": 119547,
                      "state_dict_mapping": {"bert": "encoder", "cls": "head.mlm"},
                      },

                  "bert-base-multilingual-cased-buryat_latin":
                     {
                      "model": os.path.join(os.environ.get("PRETRAINED", "."), "tuned/10923445--bert-base-multilingual-cased"),
                      "encoder": "BertModel",
                      "tokenizer": "BertTokenizer",
                      "vocab": os.path.join(BERT_MODELS_DIRECTORY, "bert-base-multilingual-cased-vocab.txt"),                      
                      "vocab_size": 119547,
                      "state_dict_mapping": {"bert": "encoder", "cls": "head.mlm"},
                      },

                  "bert-base-multilingual-cased-medow_latin":
                     {
                      "model": os.path.join(os.environ.get("PRETRAINED", "."), "tuned/10923442--bert-base-multilingual-cased"),
                      "encoder": "BertModel",
                      "tokenizer": "BertTokenizer",
                      "vocab": os.path.join(BERT_MODELS_DIRECTORY, "bert-base-multilingual-cased-vocab.txt"),                      
                      "vocab_size": 119547,
                      "state_dict_mapping": {"bert": "encoder", "cls": "head.mlm"},
                      },


                  "bert-base-multilingual-cased-medow":
                     {
                      "model": os.path.join(os.environ.get("PRETRAINED", "."), "tuned/10923436--bert-base-multilingual-cased"),
                      "encoder": "BertModel",
                      "tokenizer": "BertTokenizer",
                      "vocab": os.path.join(BERT_MODELS_DIRECTORY, "bert-base-multilingual-cased-vocab.txt"),                      
                      "vocab_size": 119547,
                      "state_dict_mapping": {"bert": "encoder", "cls": "head.mlm"},
                      },


                  "bert-base-multilingual-cased-wolo_cyr":
                     {
                      "model": os.path.join(os.environ.get("PRETRAINED", "."), "tuned/10923441--bert-base-multilingual-cased"),
                      "encoder": "BertModel",
                      "tokenizer": "BertTokenizer",
                      "vocab": os.path.join(BERT_MODELS_DIRECTORY, "bert-base-multilingual-cased-vocab.txt"),                      
                      "vocab_size": 119547,
                      "state_dict_mapping": {"bert": "encoder", "cls": "head.mlm"},
                      },

                  "bert-base-multilingual-cased-wolof":
                     {
                      "model": os.path.join(os.environ.get("PRETRAINED", "."), "tuned/10923440--bert-base-multilingual-cased"),
                      "encoder": "BertModel",
                      "tokenizer": "BertTokenizer",
                      "vocab": os.path.join(BERT_MODELS_DIRECTORY, "bert-base-multilingual-cased-vocab.txt"),                      
                      "vocab_size": 119547,
                      "state_dict_mapping": {"bert": "encoder", "cls": "head.mlm"},
                      },

                   "bert-base-multilingual-cased-mingrelian_latin":
                     {
                      "model": os.path.join(os.environ.get("PRETRAINED", "."), "tuned/10923435--bert-base-multilingual-cased"),
                      "encoder": "BertModel",
                      "tokenizer": "BertTokenizer",
                      "vocab": os.path.join(BERT_MODELS_DIRECTORY, "bert-base-multilingual-cased-vocab.txt"),                      
                      "vocab_size": 119547,
                      "state_dict_mapping": {"bert": "encoder", "cls": "head.mlm"},
                      },



                    "bert-base-multilingual-cased-mingrelian":
                     {
                      "model": os.path.join(os.environ.get("PRETRAINED", "."), "tuned/10923433--bert-base-multilingual-cased"),
                      "encoder": "BertModel",
                      "tokenizer": "BertTokenizer",
                      "vocab": os.path.join(BERT_MODELS_DIRECTORY, "bert-base-multilingual-cased-vocab.txt"),                      
                      "vocab_size": 119547,
                      "state_dict_mapping": {"bert": "encoder", "cls": "head.mlm"},
                      },

                      "bert-base-multilingual-cased-livvi_cyr":
                     {
                      "model": os.path.join(os.environ.get("PRETRAINED", "."), "tuned/10923432--bert-base-multilingual-cased"),
                      "encoder": "BertModel",
                      "tokenizer": "BertTokenizer",
                      "vocab": os.path.join(BERT_MODELS_DIRECTORY, "bert-base-multilingual-cased-vocab.txt"),                      
                      "vocab_size": 119547,
                      "state_dict_mapping": {"bert": "encoder", "cls": "head.mlm"},
                      },

                    "bert-base-multilingual-cased-livvi":
                     {
                      "model": os.path.join(os.environ.get("PRETRAINED", "."), "tuned/10923431--bert-base-multilingual-cased"),
                      "encoder": "BertModel",
                      "tokenizer": "BertTokenizer",
                      "vocab": os.path.join(BERT_MODELS_DIRECTORY, "bert-base-multilingual-cased-vocab.txt"),                      
                      "vocab_size": 119547,
                      "state_dict_mapping": {"bert": "encoder", "cls": "head.mlm"},
                      },

                  "bert-base-multilingual-cased-swiss":
                     {
                      "model": os.path.join(os.environ.get("PRETRAINED", "."), "tuned/10897746--bert-base-multilingual-cased"),
                      "encoder": "BertModel",
                      "tokenizer": "BertTokenizer",
                      "vocab": os.path.join(BERT_MODELS_DIRECTORY, "bert-base-multilingual-cased-vocab.txt"),                      
                      "vocab_size": 119547,
                      "state_dict_mapping": {"bert": "encoder", "cls": "head.mlm"},
                      },
                  
                  "bert-base-multilingual-cased-fao":
                     {
                      "model": os.path.join(os.environ.get("PRETRAINED", "."), "tuned/10897743--bert-base-multilingual-cased/checkpoint-647000"),
                      "encoder": "BertModel",
                      "tokenizer": "BertTokenizer",
                      "vocab": os.path.join(BERT_MODELS_DIRECTORY, "bert-base-multilingual-cased-vocab.txt"),                      
                      "vocab_size": 119547,
                      "state_dict_mapping": {"bert": "encoder", "cls": "head.mlm"},
                      },


                  "bert-base-multilingual-cased-ckb":
                     {
                      "model": os.path.join(os.environ.get("PRETRAINED", "."), "tuned/10898039--bert-base-multilingual-cased/checkpoint-103000"),
                      "encoder": "BertModel",
                      "tokenizer": "BertTokenizer",
                      "vocab": os.path.join(BERT_MODELS_DIRECTORY, "bert-base-multilingual-cased-vocab.txt"),                      
                      "vocab_size": 119547,
                      "state_dict_mapping": {"bert": "encoder", "cls": "head.mlm"},
                      },
                  "bert-base-multilingual-cased-ckb_translit":
                     {
                      "model": os.path.join(os.environ.get("PRETRAINED", "."), "tuned/10907086--bert-base-multilingual-cased/checkpoint-759000"),
                      "encoder": "BertModel",
                      "tokenizer": "BertTokenizer",
                      "vocab": os.path.join(BERT_MODELS_DIRECTORY, "bert-base-multilingual-cased-vocab.txt"),                      
                      "vocab_size": 119547,
                      "state_dict_mapping": {"bert": "encoder", "cls": "head.mlm"},
                      },

                  "bert-base-multilingual-cased-sd":
                     {
                      "model": os.path.join(os.environ.get("PRETRAINED", "."), "tuned/10907069--bert-base-multilingual-cased/checkpoint-865000"),
                      "encoder": "BertModel",
                      "tokenizer": "BertTokenizer",
                      "vocab": os.path.join(BERT_MODELS_DIRECTORY, "bert-base-multilingual-cased-vocab.txt"),                      
                      "vocab_size": 119547,
                      "state_dict_mapping": {"bert": "encoder", "cls": "head.mlm"},
                      },
                  "bert-base-multilingual-cased-naija":
                     {
                      "model": os.path.join(os.environ.get("PRETRAINED", "."), "tuned/10907068--bert-base-multilingual-cased"),
                      "encoder": "BertModel",
                      "tokenizer": "BertTokenizer",
                      "vocab": os.path.join(BERT_MODELS_DIRECTORY, "bert-base-multilingual-cased-vocab.txt"),                      
                      "vocab_size": 119547,
                      "state_dict_mapping": {"bert": "encoder", "cls": "head.mlm"},
                      },

                  "xlm-roberta-base-ug_select":
                     {
                      "model": os.path.join(os.environ.get("PRETRAINED", "."), "tuned/10875161--xlm-roberta-base/checkpoint-168000"),
                      "vocab": os.path.join(BERT_MODELS_DIRECTORY, "xlm-roberta-base"),#, "roberta-base-merges.txt"),
                      #"vocab_size": 250002,
                      "encoder": "RobertaModel",
                      "tokenizer": "XLMRobertaTokenizer",
                      "wordpiece_flag": "▁",
                      "flag_is_first_token": True,
                      },

                  "xlm-roberta-base-mt_with_ar_translit":
                     {
                      "model": os.path.join(os.environ.get("PRETRAINED", "."), "tuned/10875557--xlm-roberta-base/checkpoint-115000"),
                      "vocab": os.path.join(BERT_MODELS_DIRECTORY, "xlm-roberta-base"),#, "roberta-base-merges.txt"),
                      #"vocab_size": 250002,
                      "encoder": "RobertaModel",
                      "tokenizer": "XLMRobertaTokenizer",
                      "wordpiece_flag": "▁",
                      "flag_is_first_token": True,
                      },


                  "xlm-roberta-base-mt_with_mt_translit":
                     {
                      "model": os.path.join(os.environ.get("PRETRAINED", "."), "tuned/10875543--xlm-roberta-base/checkpoint-97000"),
                      "vocab": os.path.join(BERT_MODELS_DIRECTORY, "xlm-roberta-base"),#, "roberta-base-merges.txt"),
                      #"vocab_size": 250002,
                      "encoder": "RobertaModel",
                      "tokenizer": "XLMRobertaTokenizer",
                      "wordpiece_flag": "▁",
                      "flag_is_first_token": True,
                      },


                  "xlm-roberta-base-mt_concat_permute_2":
                     {
                      "model": os.path.join(os.environ.get("PRETRAINED", "."), "tuned/10875542--xlm-roberta-base/checkpoint-172000"),
                      "vocab": os.path.join(BERT_MODELS_DIRECTORY, "xlm-roberta-base"),#, "roberta-base-merges.txt"),
                      #"vocab_size": 250002,
                      "encoder": "RobertaModel",
                      "tokenizer": "XLMRobertaTokenizer",
                      "wordpiece_flag": "▁",
                      "flag_is_first_token": True,
                      },
                  

                  "xlm-roberta-base-mt_concat_permute_1":
                     {
                      "model": os.path.join(os.environ.get("PRETRAINED", "."), "tuned/10875541--xlm-roberta-base/checkpoint-239000"),
                      "vocab": os.path.join(BERT_MODELS_DIRECTORY, "xlm-roberta-base"),#, "roberta-base-merges.txt"),
                      #"vocab_size": 250002,
                      "encoder": "RobertaModel",
                      "tokenizer": "XLMRobertaTokenizer",
                      "wordpiece_flag": "▁",
                      "flag_is_first_token": True,
                      },


                  "xlm-roberta-base-select_10873902":
                     {
                      "model": os.path.join(os.environ.get("PRETRAINED", "."), "tuned/10875539--xlm-roberta-base/checkpoint-167000"),
                      "vocab": os.path.join(BERT_MODELS_DIRECTORY, "xlm-roberta-base"),#, "roberta-base-merges.txt"),
                      #"vocab_size": 250002,
                      "encoder": "RobertaModel",
                      "tokenizer": "XLMRobertaTokenizer",
                      "wordpiece_flag": "▁",
                      "flag_is_first_token": True,
                      },



                   "xlm-roberta-base-mt_ar_it":
                     {
                      "model": os.path.join(os.environ.get("PRETRAINED", "."), "tuned/10869057--xlm-roberta-base"),
                      "vocab": os.path.join(BERT_MODELS_DIRECTORY, "xlm-roberta-base"),#, "roberta-base-merges.txt"),
                      #"vocab_size": 250002,
                      "encoder": "RobertaModel",
                      "tokenizer": "XLMRobertaTokenizer",
                      "wordpiece_flag": "▁",
                      "flag_is_first_token": True,
                      },

                   "xlm-roberta-base-mt_en":
                     {
                      "model": os.path.join(os.environ.get("PRETRAINED", "."), "tuned/10869056--xlm-roberta-base"),
                      "vocab": os.path.join(BERT_MODELS_DIRECTORY, "xlm-roberta-base"),#, "roberta-base-merges.txt"),
                      #"vocab_size": 250002,
                      "encoder": "RobertaModel",
                      "tokenizer": "XLMRobertaTokenizer",
                      "wordpiece_flag": "▁",
                      "flag_is_first_token": True,
                      },

                  "xlm-roberta-base-mt_en_ar":
                     {
                      "model": os.path.join(os.environ.get("PRETRAINED", "."), "tuned/10869055--xlm-roberta-base"),
                      "vocab": os.path.join(BERT_MODELS_DIRECTORY, "xlm-roberta-base"),#, "roberta-base-merges.txt"),
                      #"vocab_size": 250002,
                      "encoder": "RobertaModel",
                      "tokenizer": "XLMRobertaTokenizer",
                      "wordpiece_flag": "▁",
                      "flag_is_first_token": True,
                      },
                      
                  "xlm-roberta-base-mt_ar_2_50k":
                     {
                      "model": os.path.join(os.environ.get("PRETRAINED", "."), "tuned/10858703-mt_ar_50k_2-xlm-roberta-base"),
                      "vocab": os.path.join(BERT_MODELS_DIRECTORY, "xlm-roberta-base"),#, "roberta-base-merges.txt"),
                      #"vocab_size": 250002,
                      "encoder": "RobertaModel",
                      "tokenizer": "XLMRobertaTokenizer",
                      "wordpiece_flag": "▁",
                      "flag_is_first_token": True,
                      },
                    
                    "xlm-roberta-base-mt_50k":
                     {
                      "model": os.path.join(os.environ.get("PRETRAINED", "."), "tuned/10858702-mt-xlm-roberta-base"),
                      "vocab": os.path.join(BERT_MODELS_DIRECTORY, "xlm-roberta-base"),#, "roberta-base-merges.txt"),
                      #"vocab_size": 250002,
                      "encoder": "RobertaModel",
                      "tokenizer": "XLMRobertaTokenizer",
                      "wordpiece_flag": "▁",
                      "flag_is_first_token": True,
                      
                      },

                      "xlm-roberta-base-ug_ar":
                     {
                      "model": os.path.join(os.environ.get("PRETRAINED", "."), "tuned/10858649-ug_ar_2-xlm-roberta-base/checkpoint-283000"),
                      "vocab": os.path.join(BERT_MODELS_DIRECTORY, "xlm-roberta-base"),#, "roberta-base-merges.txt"),
                      #"vocab_size": 250002,
                      "encoder": "RobertaModel",
                      "tokenizer": "XLMRobertaTokenizer",
                      "wordpiece_flag": "▁",
                      "flag_is_first_token": True,
                      
                      },

                    
                     "xlm-roberta-base-ug_tr_az_kk_en_select":
                     {
                      "model": os.path.join(os.environ.get("PRETRAINED", "."), "tuned/10858705-ug_50k_select_kk_az_tr_en-xlm-roberta-base/backup-checkpoint-143000"),
                      "vocab": os.path.join(BERT_MODELS_DIRECTORY, "xlm-roberta-base"),#, "roberta-base-merges.txt"),
                      #"vocab_size": 250002,
                      "encoder": "RobertaModel",
                      "tokenizer": "XLMRobertaTokenizer",
                      "wordpiece_flag": "▁",
                      "flag_is_first_token": True,
                      
                      },

                     "xlm-roberta-base-ug_tr_az_kk":
                      {
                      "model": os.path.join(os.environ.get("PRETRAINED", "."), "tuned/10857784-ug_tr_az_kk-xlm-roberta-base/checkpoint-344000"),
                      "vocab": os.path.join(BERT_MODELS_DIRECTORY, "xlm-roberta-base"),#, "roberta-base-merges.txt"),
                      #"vocab_size": 250002,
                      "encoder": "RobertaModel",
                      "tokenizer": "XLMRobertaTokenizer",
                      "wordpiece_flag": "▁",
                      "flag_is_first_token": True,
                      #"state_dict_mapping": {"roberta": "encoder",  # "lm_head":,
                      #                       "lm_head.decoder": "head.mlm.predictions.decoder",
                      #                       "lm_head.dense": "head.mlm.predictions.transform.dense",
                      ##                       "lm_head.bias": "head.mlm.predictions.bias",
                      #                       "lm_head.layer_norm": "head.mlm.predictions.transform.LayerNorm"},
                      },
                     
                    "xlm-roberta-base-ug_tr_az_kk_translit":
                      {
                      "model": os.path.join(os.environ.get("PRETRAINED", "."), "tuned/10857782-ug_tr_az_kk_translit-xlm-roberta-base/checkpoint-341000"),
                      "vocab": os.path.join(BERT_MODELS_DIRECTORY, "xlm-roberta-base"),#, "roberta-base-merges.txt"),
                      #"vocab_size": 250002,
                      "encoder": "RobertaModel",
                      "tokenizer": "XLMRobertaTokenizer",
                      "wordpiece_flag": "▁",
                      "flag_is_first_token": True,
                      #"state_dict_mapping": {"roberta": "encoder",  # "lm_head":,
                      #                       "lm_head.decoder": "head.mlm.predictions.decoder",
                      #                       "lm_head.dense": "head.mlm.predictions.transform.dense",
                      ##                       "lm_head.bias": "head.mlm.predictions.bias",
                      #                       "lm_head.layer_norm": "head.mlm.predictions.transform.LayerNorm"},
                      },
                     
                     
                        "xlm-roberta-base-ug_ug_translit":
                      {
                      "model": os.path.join(os.environ.get("PRETRAINED", "."), "tuned/10857778-ug_ug_translit-xlm-roberta-base/checkpoint-338000"),
                      "vocab": os.path.join(BERT_MODELS_DIRECTORY, "xlm-roberta-base"),#, "roberta-base-merges.txt"),
                      #"vocab_size": 250002,
                      "encoder": "RobertaModel",
                      "tokenizer": "XLMRobertaTokenizer",
                      "wordpiece_flag": "▁",
                      "flag_is_first_token": True,
                      #"state_dict_mapping": {"roberta": "encoder",  # "lm_head":,
                      #                       "lm_head.decoder": "head.mlm.predictions.decoder",
                      #                       "lm_head.dense": "head.mlm.predictions.transform.dense",
                      ##                       "lm_head.bias": "head.mlm.predictions.bias",
                      #                       "lm_head.layer_norm": "head.mlm.predictions.transform.LayerNorm"},
                      },


                      "xlm-roberta-base-az":
                      {
                      "model": os.path.join(os.environ.get("PRETRAINED", "."), "tuned/10843630-az-xlm-roberta-base"),
                      "vocab": os.path.join(BERT_MODELS_DIRECTORY, "xlm-roberta-base"),#, "roberta-base-merges.txt"),
                      #"vocab_size": 250002,
                      "encoder": "RobertaModel",
                      "tokenizer": "XLMRobertaTokenizer",
                      "wordpiece_flag": "▁",
                      "flag_is_first_token": True,
                      #"state_dict_mapping": {"roberta": "encoder",  # "lm_head":,
                      #                       "lm_head.decoder": "head.mlm.predictions.decoder",
                      #                       "lm_head.dense": "head.mlm.predictions.transform.dense",
                      ##                       "lm_head.bias": "head.mlm.predictions.bias",
                      #                       "lm_head.layer_norm": "head.mlm.predictions.transform.LayerNorm"},
                      },

                      "xlm-roberta-base-ug_tr_kk_az_200k":
                      {
                      "model": os.path.join(os.environ.get("PRETRAINED", "."), "tuned/10843638-ug_tr_kk_az-xlm-roberta-base/checkpoint-147000"),
                      "vocab": os.path.join(BERT_MODELS_DIRECTORY, "xlm-roberta-base"),#, "roberta-base-merges.txt"),
                      #"vocab_size": 250002,
                      "encoder": "RobertaModel",
                      "tokenizer": "XLMRobertaTokenizer",
                      "wordpiece_flag": "▁",
                      "flag_is_first_token": True,
                      #"state_dict_mapping": {"roberta": "encoder",  # "lm_head":,
                      #                       "lm_head.decoder": "head.mlm.predictions.decoder",
                      #                       "lm_head.dense": "head.mlm.predictions.transform.dense",
                      ##                       "lm_head.bias": "head.mlm.predictions.bias",
                      #                       "lm_head.layer_norm": "head.mlm.predictions.transform.LayerNorm"},
                      },

                    
                      "xlm-roberta-base-ug_en_200k":
                      {
                      "model": os.path.join(os.environ.get("PRETRAINED", "."), "tuned/10843644-ug_en-xlm-roberta-base/checkpoint-147000"),
                      "vocab": os.path.join(BERT_MODELS_DIRECTORY, "xlm-roberta-base"),#, "roberta-base-merges.txt"),
                      #"vocab_size": 250002,
                      "encoder": "RobertaModel",
                      "tokenizer": "XLMRobertaTokenizer",
                      "wordpiece_flag": "▁",
                      "flag_is_first_token": True,
                      #"state_dict_mapping": {"roberta": "encoder",  # "lm_head":,
                      #                       "lm_head.decoder": "head.mlm.predictions.decoder",
                      #                       "lm_head.dense": "head.mlm.predictions.transform.dense",
                      ##                       "lm_head.bias": "head.mlm.predictions.bias",
                      #                       "lm_head.layer_norm": "head.mlm.predictions.transform.LayerNorm"},
                      },


                      "xlm-roberta-base-ug_en_100k":
                      {
                      "model": os.path.join(os.environ.get("PRETRAINED", "."), "tuned/10843653-ug_en-xlm-roberta-base"),
                      "vocab": os.path.join(BERT_MODELS_DIRECTORY, "xlm-roberta-base"),#, "roberta-base-merges.txt"),
                      #"vocab_size": 250002,
                      "encoder": "RobertaModel",
                      "tokenizer": "XLMRobertaTokenizer",
                      "wordpiece_flag": "▁",
                      "flag_is_first_token": True,
                      #"state_dict_mapping": {"roberta": "encoder",  # "lm_head":,
                      #                       "lm_head.decoder": "head.mlm.predictions.decoder",
                      #                       "lm_head.dense": "head.mlm.predictions.transform.dense",
                      ##                       "lm_head.bias": "head.mlm.predictions.bias",
                      #                       "lm_head.layer_norm": "head.mlm.predictions.transform.LayerNorm"},
                      },



                      "xlm-roberta-base-ug_tr_100k":
                      {
                      "model": os.path.join(os.environ.get("PRETRAINED", "."), "tuned/10830666-ug_tr_100k-xlm-roberta-base/backup-checkpoint-181000"),
                      "vocab": os.path.join(BERT_MODELS_DIRECTORY, "xlm-roberta-base"),#, "roberta-base-merges.txt"),
                      #"vocab_size": 250002,
                      "encoder": "RobertaModel",
                      "tokenizer": "XLMRobertaTokenizer",
                      "wordpiece_flag": "▁",
                      "flag_is_first_token": True,
                      #"state_dict_mapping": {"roberta": "encoder",  # "lm_head":,
                      #                       "lm_head.decoder": "head.mlm.predictions.decoder",
                      #                       "lm_head.dense": "head.mlm.predictions.transform.dense",
                      ##                       "lm_head.bias": "head.mlm.predictions.bias",
                      #                       "lm_head.layer_norm": "head.mlm.predictions.transform.LayerNorm"},
                      },

                      "xlm-roberta-base-ug_hu_100k":
                      {
                      "model": os.path.join(os.environ.get("PRETRAINED","."), "tuned/10830647-ug_hu_100k-xlm-roberta-base/backup-checkpoint-219000"),
                      "vocab": os.path.join(BERT_MODELS_DIRECTORY, "xlm-roberta-base"),#, "roberta-base-merges.txt"),
                      #"vocab_size": 250002,
                      "encoder": "RobertaModel",
                      "tokenizer": "XLMRobertaTokenizer",
                      "wordpiece_flag": "▁",
                      "flag_is_first_token": True,
                      #"state_dict_mapping": {"roberta": "encoder",  # "lm_head":,
                      #                       "lm_head.decoder": "head.mlm.predictions.decoder",
                      #                       "lm_head.dense": "head.mlm.predictions.transform.dense",
                      ##                       "lm_head.bias": "head.mlm.predictions.bias",
                      #                       "lm_head.layer_norm": "head.mlm.predictions.transform.LayerNorm"},
                      },

                    'robert-bm': { 'model': os.path.join(os.environ.get('PRETRAINED','.'), '11043877_job-627c7_model-bm'), 'encoder': 'RobertaModel', 'tokenizer': 'RobertaTokenizerFast', 'vocab': os.path.join(os.environ.get('PRETRAINED','.'),'bm' ,'tokenizer'), 'wordpiece_flag': '▁', 'flag_is_first_token': True, },
                    'robert-wo': { 'model': os.path.join(os.environ.get('PRETRAINED','.'), '11043877_job-b2501_model-wo'), 'encoder': 'RobertaModel', 'tokenizer': 'RobertaTokenizerFast', 'vocab': os.path.join(os.environ.get('PRETRAINED','.'),'wo' ,'tokenizer'), 'wordpiece_flag': '▁', 'flag_is_first_token': True, },
                    'robert-xmf': { 'model': os.path.join(os.environ.get('PRETRAINED','.'), '11043877_job-7140a_model-xmf'), 'encoder': 'RobertaModel', 'tokenizer': 'RobertaTokenizerFast', 'vocab': os.path.join(os.environ.get('PRETRAINED','.'),'xmf' ,'tokenizer'), 'wordpiece_flag': '▁', 'flag_is_first_token': True, },
                    'robert-xmf_latin': { 'model': os.path.join(os.environ.get('PRETRAINED','.'), '11043877_job-9b1ff_model-xmf_latin'), 'encoder': 'RobertaModel', 'tokenizer': 'RobertaTokenizerFast', 'vocab': os.path.join(os.environ.get('PRETRAINED','.'),'xmf_latin' ,'tokenizer'), 'wordpiece_flag': '▁', 'flag_is_first_token': True, },
                    'robert-olo': { 'model': os.path.join(os.environ.get('PRETRAINED','.'), '11043878_job-1e7de_model-olo'), 'encoder': 'RobertaModel', 'tokenizer': 'RobertaTokenizerFast', 'vocab': os.path.join(os.environ.get('PRETRAINED','.'),'olo' ,'tokenizer'), 'wordpiece_flag': '▁', 'flag_is_first_token': True, },
                    'robert-olo_cyr': { 'model': os.path.join(os.environ.get('PRETRAINED','.'), '11043878_job-eb7cf_model-olo_cyr'), 'encoder': 'RobertaModel', 'tokenizer': 'RobertaTokenizerFast', 'vocab': os.path.join(os.environ.get('PRETRAINED','.'),'olo_cyr' ,'tokenizer'), 'wordpiece_flag': '▁', 'flag_is_first_token': True, },
                    'robert-mhr': { 'model': os.path.join(os.environ.get('PRETRAINED','.'), '11043878_job-a1136_model-mhr'), 'encoder': 'RobertaModel', 'tokenizer': 'RobertaTokenizerFast', 'vocab': os.path.join(os.environ.get('PRETRAINED','.'),'mhr' ,'tokenizer'), 'wordpiece_flag': '▁', 'flag_is_first_token': True, },
                    'robert-mhr_translit': { 'model': os.path.join(os.environ.get('PRETRAINED','.'), '11043878_job-1865e_model-mhr_translit'), 'encoder': 'RobertaModel', 'tokenizer': 'RobertaTokenizerFast', 'vocab': os.path.join(os.environ.get('PRETRAINED','.'),'mhr_translit' ,'tokenizer'), 'wordpiece_flag': '▁', 'flag_is_first_token': True, },
                    'robert-bxu': { 'model': os.path.join(os.environ.get('PRETRAINED','.'), '11043879_job-393e0_model-bxu'), 'encoder': 'RobertaModel', 'tokenizer': 'RobertaTokenizerFast', 'vocab': os.path.join(os.environ.get('PRETRAINED','.'),'bxu' ,'tokenizer'), 'wordpiece_flag': '▁', 'flag_is_first_token': True, },
                    'robert-bxu_translit': { 'model': os.path.join(os.environ.get('PRETRAINED','.'), '11043879_job-b94a0_model-bxu_translit'), 'encoder': 'RobertaModel', 'tokenizer': 'RobertaTokenizerFast', 'vocab': os.path.join(os.environ.get('PRETRAINED','.'),'bxu_translit' ,'tokenizer'), 'wordpiece_flag': '▁', 'flag_is_first_token': True, },
                    'robert-myv': { 'model': os.path.join(os.environ.get('PRETRAINED','.'), '11043879_job-6463e_model-myv'), 'encoder': 'RobertaModel', 'tokenizer': 'RobertaTokenizerFast', 'vocab': os.path.join(os.environ.get('PRETRAINED','.'),'myv' ,'tokenizer'), 'wordpiece_flag': '▁', 'flag_is_first_token': True, },
                    'robert-myv_translit': { 'model': os.path.join(os.environ.get('PRETRAINED','.'), '11043879_job-a3d72_model-myv_translit'), 'encoder': 'RobertaModel', 'tokenizer': 'RobertaTokenizerFast', 'vocab': os.path.join(os.environ.get('PRETRAINED','.'),'myv_translit' ,'tokenizer'), 'wordpiece_flag': '▁', 'flag_is_first_token': True, },
                    'robert-naija': { 'model': os.path.join(os.environ.get('PRETRAINED','.'), '11043880_job-6f5cb_model-naija/checkpoint-575000/'), 'encoder': 'RobertaModel', 'tokenizer': 'RobertaTokenizerFast', 'vocab': os.path.join(os.environ.get('PRETRAINED','.'),'myv_translit' ,'tokenizer'), 'wordpiece_flag': '▁', 'flag_is_first_token': True, },


                      "robert-ckb_translit":
                        {
                          "model":  os.path.join(os.environ.get("PRETRAINED","."), "10907080_job-95259_model-ckb_translit/checkpoint-246000"),
                          "encoder": "RobertaModel",
                          "tokenizer": "RobertaTokenizerFast",
                          "vocab": os.path.join(os.environ.get("PRETRAINED","."), "ckb_translit","tokenizer"),
                          "wordpiece_flag": "▁",
                          "flag_is_first_token": True,
                       },
                      
                      "robert-sd":
                        {
                          "model": os.path.join(os.environ.get("PRETRAINED","."), "10907080_job-9b0f9_model-sd/checkpoint-258000"),
                          "encoder": "RobertaModel",
                          "tokenizer": "RobertaTokenizerFast",
                          "vocab": os.path.join(os.environ.get("PRETRAINED","."), "sd","tokenizer"),
                          "wordpiece_flag": "▁",
                          "flag_is_first_token": True,
                       },

                    "robert-ug_100k_ug_en_2_100k":
                        {
                          "model": os.path.join(os.environ.get("PRETRAINED","."), "10844316_job-3ce24_model-ug_en_2_100k"),
                          "encoder": "RobertaModel",
                          "tokenizer": "RobertaTokenizerFast",
                          "vocab": os.path.join(os.environ.get("PRETRAINED","."), "ug_en_2_100k","tokenizer"),
                          "wordpiece_flag": "▁",
                          "flag_is_first_token": True,
                       },

                    "robert-ug_100k_tr_kk_az_100k_3_33k":
                        {
                          "model": os.path.join(os.environ.get("PRETRAINED","."), "10844316_job-0fdeb_model-ug_tr_kk_az_3_33k"),
                          "encoder": "RobertaModel",
                          "tokenizer": "RobertaTokenizerFast",
                          "vocab": os.path.join(os.environ.get("PRETRAINED","."), "ug_tr_kk_az_3_33k","tokenizer"),
                          "wordpiece_flag": "▁",
                          "flag_is_first_token": True,
                       },


                       

                      "xlm-roberta-base-ug_100k_az_100k_tr_100k":
                      {
                      "model": os.path.join(os.environ.get("PRETRAINED","."), "tuned/10844306-ug_100k_az_100k_tr_100k-xlm-roberta-base/checkpoint-689000"),
                      "vocab": os.path.join(BERT_MODELS_DIRECTORY, "xlm-roberta-base"),#, "roberta-base-merges.txt"),
                      #"vocab_size": 250002,
                      "encoder": "RobertaModel",
                      "tokenizer": "XLMRobertaTokenizer",
                      "wordpiece_flag": "▁",
                      "flag_is_first_token": True,
                      #"state_dict_mapping": {"roberta": "encoder",  # "lm_head":,
                      #                       "lm_head.decoder": "head.mlm.predictions.decoder",
                      #                       "lm_head.dense": "head.mlm.predictions.transform.dense",
                      ##                       "lm_head.bias": "head.mlm.predictions.bias",
                      #                       "lm_head.layer_norm": "head.mlm.predictions.transform.LayerNorm"},
                      },

                      
                      

                      "robert-ug_tr_az_kk_translit":
                        {
                          "model": os.path.join(os.environ.get("PRETRAINED","."), "10846756_job-34b49_model-ug_tr_az_kk_translit/checkpoint-470000"),
                          "encoder": "RobertaModel",
                          "tokenizer": "RobertaTokenizerFast",
                          "vocab": os.path.join(os.environ.get("PRETRAINED","."), "ug_ug_translit","tokenizer"),
                          "wordpiece_flag": "▁",
                          "flag_is_first_token": True,
                       },

                    "robert-ug_ug_translit":
                        {
                          "model": os.path.join(os.environ.get("PRETRAINED","."), "10846756_job-2b180_model-ug_ug_translit"),
                          "encoder": "RobertaModel",
                          "tokenizer": "RobertaTokenizerFast",
                          "vocab": os.path.join(os.environ.get("PRETRAINED","."), "ug_ug_translit","tokenizer"),
                          "wordpiece_flag": "▁",
                          "flag_is_first_token": True,
                       },





                       "bert-base-multilingual-cased-ug_50k_select_kk_az_tr_en":
                      {
                     "model": os.path.join(os.environ.get("PRETRAINED","."),  "tuned/10858745-ug_50k_select_kk_az_tr_en-bert-base-multilingual-cased/checkpoint-170000"),
                     "encoder": "BertModel",
                      "tokenizer": "BertTokenizer",
                      "vocab": os.path.join(BERT_MODELS_DIRECTORY, "bert-base-multilingual-cased-vocab.txt"),                      
                      "vocab_size": 119547,
                      "state_dict_mapping": {"bert": "encoder", "cls": "head.mlm"},
                      },

                      "bert-base-multilingual-cased-mt":
                      {
                     "model": os.path.join(os.environ.get("PRETRAINED","."),  "tuned/10858701-mt-bert-base-multilingual-cased"),
                     "encoder": "BertModel",
                      "tokenizer": "BertTokenizer",
                      "vocab": os.path.join(BERT_MODELS_DIRECTORY, "bert-base-multilingual-cased-vocab.txt"),                      
                      "vocab_size": 119547,
                      "state_dict_mapping": {"bert": "encoder", "cls": "head.mlm"},
                      },

                       "bert-base-multilingual-cased-ug_ar":
                      {
                     "model": os.path.join(os.environ.get("PRETRAINED","."),  "tuned/10858647-ug_ar_2-bert-base-multilingual-cased/checkpoint-415000"),
                     "encoder": "BertModel",
                      "tokenizer": "BertTokenizer",
                      "vocab": os.path.join(BERT_MODELS_DIRECTORY, "bert-base-multilingual-cased-vocab.txt"),                      
                      "vocab_size": 119547,
                      "state_dict_mapping": {"bert": "encoder", "cls": "head.mlm"},
                      },


                      "bert-base-multilingual-cased-ug_ug_translit":
                      {
                     "model": os.path.join(os.environ.get("PRETRAINED","."),  "tuned/10846720-ug_ug_translit-bert-base-multilingual-cased/checkpoint-381000"),
                     "encoder": "BertModel",
                      "tokenizer": "BertTokenizer",
                      "vocab": os.path.join(BERT_MODELS_DIRECTORY, "bert-base-multilingual-cased-vocab.txt"),                      
                      "vocab_size": 119547,
                      "state_dict_mapping": {"bert": "encoder", "cls": "head.mlm"},
                      },

                      "bert-base-multilingual-cased-ug_translit":
                      {
                     "model": os.path.join(os.environ.get("PRETRAINED","."),  "tuned/10846718-ug_translit-bert-base-multilingual-cased"),
                     "encoder": "BertModel",
                      "tokenizer": "BertTokenizer",
                      "vocab": os.path.join(BERT_MODELS_DIRECTORY, "bert-base-multilingual-cased-vocab.txt"),                      
                      "vocab_size": 119547,
                      "state_dict_mapping": {"bert": "encoder", "cls": "head.mlm"},
                      },


                      "bert-base-multilingual-cased-ug_100k_az_100k_tr_100K":
                      {
                     "model": os.path.join(os.environ.get("PRETRAINED","."), "tuned/10844307-ug_100k_az_100k_tr_100k-bert-base-multilingual-cased/"),
                     "encoder": "BertModel",
                      "tokenizer": "BertTokenizer",
                      "vocab": os.path.join(BERT_MODELS_DIRECTORY, "bert-base-multilingual-cased-vocab.txt"),                      
                      "vocab_size": 119547,
                      "state_dict_mapping": {"bert": "encoder", "cls": "head.mlm"},
                      },



                      "xlm-roberta-base-ug_100k_tr_kk_az_3_33k_en_100k":
                      {
                      "model": os.path.join(os.environ.get("PRETRAINED","."), "10844305-ug_100k_tr_kk_az_3_33k_en_100k-xlm-roberta-base/checkpoint-452000/"),
                      "vocab": os.path.join(BERT_MODELS_DIRECTORY, "xlm-roberta-base"),#, "roberta-base-merges.txt"),
                      #"vocab_size": 250002,
                      "encoder": "RobertaModel",
                      "tokenizer": "XLMRobertaTokenizer",
                      "wordpiece_flag": "▁",
                      "flag_is_first_token": True,
                      #"state_dict_mapping": {"roberta": "encoder",  # "lm_head":,
                      #                       "lm_head.decoder": "head.mlm.predictions.decoder",
                      #                       "lm_head.dense": "head.mlm.predictions.transform.dense",
                      ##                       "lm_head.bias": "head.mlm.predictions.bias",
                      #                       "lm_head.layer_norm": "head.mlm.predictions.transform.LayerNorm"},
                      },
                      
                    "bert-base-multilingual-cased-ug_100k_tr_kk_az_3_33k_en_100k":
                      {
                     "model": os.path.join(os.environ.get("PRETRAINED","."), "tuned/10844304-ug_100k_tr_kk_az_3_33k_en_100k-bert-base-multilingual-cased/checkpoint-656000"),
                     "encoder": "BertModel",
                      "tokenizer": "BertTokenizer",
                      "vocab": os.path.join(BERT_MODELS_DIRECTORY, "bert-base-multilingual-cased-vocab.txt"),                      
                      "vocab_size": 119547,
                      "state_dict_mapping": {"bert": "encoder", "cls": "head.mlm"},
                      },

                    "bert-base-multilingual-cased-ug_tr_az_kk_translit":
                      {
                     "model": os.path.join(os.environ.get("PRETRAINED","."), "tuned/10846757-ug_tr_az_kk_translit-bert-base-multilingual-cased/checkpoint-448000"),
                     "encoder": "BertModel",
                      "tokenizer": "BertTokenizer",
                      "vocab": os.path.join(BERT_MODELS_DIRECTORY, "bert-base-multilingual-cased-vocab.txt"),                      
                      "vocab_size": 119547,
                      "state_dict_mapping": {"bert": "encoder", "cls": "head.mlm"},
                      },


                      "bert-base-multilingual-cased-az_100k":
                      {
                     "model": os.path.join(os.environ.get("PRETRAINED","."), "tuned/10843629-az-bert-base-multilingual-cased/"),
                     "encoder": "BertModel",
                      "tokenizer": "BertTokenizer",
                      "vocab": os.path.join(BERT_MODELS_DIRECTORY, "bert-base-multilingual-cased-vocab.txt"),                      
                      "vocab_size": 119547,
                      "state_dict_mapping": {"bert": "encoder", "cls": "head.mlm"},
                      },


                      "bert-base-multilingual-cased-tr_100k":
                      {
                     "model": os.path.join(os.environ.get("PRETRAINED","."), "tuned/10843633-tr-bert-base-multilingual-cased/checkpoint-229000"),
                     "encoder": "BertModel",
                      "tokenizer": "BertTokenizer",
                      "vocab": os.path.join(BERT_MODELS_DIRECTORY, "bert-base-multilingual-cased-vocab.txt"),                      
                      "vocab_size": 119547,
                      "state_dict_mapping": {"bert": "encoder", "cls": "head.mlm"},
                      },



                      "bert-base-multilingual-cased-ug_tr_kk_az_200k":
                      {
                     "model": os.path.join(os.environ.get("PRETRAINED","."), "tuned/10843641-ug_tr_kk_az-bert-base-multilingual-cased/checkpoint-229000"),
                     "encoder": "BertModel",
                      "tokenizer": "BertTokenizer",
                      "vocab": os.path.join(BERT_MODELS_DIRECTORY, "bert-base-multilingual-cased-vocab.txt"),                      
                      "vocab_size": 119547,
                      "state_dict_mapping": {"bert": "encoder", "cls": "head.mlm"},
                      },



                      "bert-base-multilingual-cased-ug_en_100k":
                      {
                     "model": os.path.join(os.environ.get("PRETRAINED","."), "tuned/10843654-ug_en-bert-base-multilingual-cased/checkpoint-224000"),
                     "encoder": "BertModel",
                      "tokenizer": "BertTokenizer",
                      "vocab": os.path.join(BERT_MODELS_DIRECTORY, "bert-base-multilingual-cased-vocab.txt"),                      
                      "vocab_size": 119547,
                      "state_dict_mapping": {"bert": "encoder", "cls": "head.mlm"},
                      },
                  
                    "bert-base-multilingual-cased-ug_hu_100k":
                      {
                     "model": os.path.join(os.environ.get("PRETRAINED","."), "tuned/10830646-ug_hu_100k-bert-base-multilingual-cased"),
                     "encoder": "BertModel",
                      "tokenizer": "BertTokenizer",
                      "vocab": os.path.join(BERT_MODELS_DIRECTORY, "bert-base-multilingual-cased-vocab.txt"),                      
                      "vocab_size": 119547,
                      "state_dict_mapping": {"bert": "encoder", "cls": "head.mlm"},
                      },

                    
                    "bert-base-multilingual-cased-ug_tr_100k":
                      {
                     "model": os.path.join(os.environ.get("PRETRAINED","."), "tuned/10830665-ug_tr_100k-bert-base-multilingual-cased/back-checkpoint-268000"),
                     "encoder": "BertModel",
                      "tokenizer": "BertTokenizer",
                      "vocab": os.path.join(BERT_MODELS_DIRECTORY, "bert-base-multilingual-cased-vocab.txt"),                      
                      "vocab_size": 119547,
                      "state_dict_mapping": {"bert": "encoder", "cls": "head.mlm"},
                      },


                    "camembert-base-narabizi":{
                            "encoder": "RobertaModel",

                            "model": os.path.join(os.environ.get("PRETRAINED","."), "tuned/10822460-narabizi-camembert-base"),
                            "tokenizer": "CamembertTokenizer",
                            "vocab":os.path.join(os.environ.get("PRETRAINED","."), "tuned/10822460-narabizi-camembert-base"),
                            "wordpiece_flag": "▁",
                            "flag_is_first_token": True,
                            "state_dict_mapping": {"roberta": "encoder",
                                                   "lm_head.decoder": "head.mlm.predictions.decoder",
                                                   "lm_head.dense": "head.mlm.predictions.transform.dense",
                                                   "lm_head.bias": "head.mlm.predictions.bias",
                                                   "lm_head.layer_norm": "head.mlm.predictions.transform.LayerNorm"}
                  },
                    
                    "KB-bert-base-swedish-cased-fao":{
                        "model": os.path.join(os.environ.get("PRETRAINED","."), "tuned/10819361-fao-KB-bert-base-swedish-cased"),
                        "vocab":"KB/bert-base-swedish-cased",
                        "encoder": "BertModel",
                        "tokenizer": "BertTokenizer",
                        "state_dict_mapping": {"bert": "encoder", "cls": "head.mlm"},
                     
                      },

                    "bert-base-german-cased-gsw":{
                        "model": os.path.join(os.environ.get("PRETRAINED","."), "tuned/10819358-gsw-bert-base-german-cased"),
                        "vocab":"bert-base-german-cased",
                        "encoder": "BertModel",
                        "tokenizer": "BertTokenizer",
                        "state_dict_mapping": {"bert": "encoder", "cls": "head.mlm"},
                     
                      },

                    "asafaya-bert-base-arabic-ug":{
                        "model": os.path.join(os.environ.get("PRETRAINED","."), "tuned/10811974-ug-bert-base-multilingual-cased"),
                        "encoder": "BertModel",
                        "vocab":"asafaya/bert-base-arabic",
                        "tokenizer": "BertTokenizer",
                        "state_dict_mapping": {"bert": "encoder", "cls": "head.mlm"},
                     
                      },

                        "bert_base_multilingual_cased-ug_100ep":
                      {
                    "model": os.path.join(os.environ.get("PRETRAINED","."), "tuned/10845017-ug-bert-base-multilingual-cased/checkpoint-1058000"),
                    
                     "encoder": "BertModel",
                      "tokenizer": "BertTokenizer",
                      "vocab": os.path.join(BERT_MODELS_DIRECTORY, "bert-base-multilingual-cased-vocab.txt"),                      
                      "vocab_size": 119547,
                      "state_dict_mapping": {"bert": "encoder", "cls": "head.mlm"},
                     
                      },


                      "bert_base_multilingual_cased-ug":
                      {
                    "model": os.path.join(os.environ.get("PRETRAINED","."), "tuned/10811974-ug-bert-base-multilingual-cased"),
                    
                     "encoder": "BertModel",
                      "tokenizer": "BertTokenizer",
                      "vocab": os.path.join(BERT_MODELS_DIRECTORY, "bert-base-multilingual-cased-vocab.txt"),                      
                      "vocab_size": 119547,
                      "state_dict_mapping": {"bert": "encoder", "cls": "head.mlm"},
                     
                      },

                    "xlm-roberta-base-malt-100k":
                      {
                      "model": os.path.join(os.environ.get("PRETRAINED","."), "tuned/10812170-mt-xlm-roberta-base/checkpoint-423000"),
                      "vocab": os.path.join(BERT_MODELS_DIRECTORY, "xlm-roberta-base"),#, "roberta-base-merges.txt"),
                      #"vocab_size": 250002,
                      "encoder": "RobertaModel",
                      "tokenizer": "XLMRobertaTokenizer",
                      "wordpiece_flag": "▁",
                      "flag_is_first_token": True,
                      #"state_dict_mapping": {"roberta": "encoder",  # "lm_head":,
                      #                       "lm_head.decoder": "head.mlm.predictions.decoder",
                      #                       "lm_head.dense": "head.mlm.predictions.transform.dense",
                      ##                       "lm_head.bias": "head.mlm.predictions.bias",
                      #                       "lm_head.layer_norm": "head.mlm.predictions.transform.LayerNorm"},
                      },


                      "bert_base_multilingual_cased-narabizi": {
                            "model": os.path.join(os.environ.get("PRETRAINED","."), "tuned/10812194-narabizi-bert-base-multilingual-cased/"),
                            "encoder": "BertModel",
                            "tokenizer": "BertTokenizer",
                            "vocab": os.path.join(BERT_MODELS_DIRECTORY, "bert-base-multilingual-cased-vocab.txt"),                      
                            "vocab_size": 119547,
                            "state_dict_mapping": {"bert": "encoder", "cls": "head.mlm"},
                  },

       
                    "xlm-roberta-base-narabizi":
                        {
                          "model": os.path.join(os.environ.get("PRETRAINED","."), "tuned/10812188-narabizi-xlm-roberta-base"),

                      "vocab": os.path.join(BERT_MODELS_DIRECTORY, "xlm-roberta-base"),#, "roberta-base-merges.txt"),
                      #"vocab_size": 250002,
                      "encoder": "RobertaModel",
                      "tokenizer": "XLMRobertaTokenizer",
                      "wordpiece_flag": "▁",
                      "flag_is_first_token": True,
                      #"state_dict_mapping": {"roberta": "encoder",  # "lm_head":,
                      #                       "lm_head.decoder": "head.mlm.predictions.decoder",
                      #                       "lm_head.dense": "head.mlm.predictions.transform.dense",
                      ##                       "lm_head.bias": "head.mlm.predictions.bias",
                      #                       "lm_head.layer_norm": "head.mlm.predictions.transform.LayerNorm"},
                      },

                      "xlm-roberta-base-ug":
                        {
                          "model": os.path.join(os.environ.get("PRETRAINED","."), "tuned/10812024-ug-xlm-roberta-base"),

                      "vocab": os.path.join(BERT_MODELS_DIRECTORY, "xlm-roberta-base"),#, "roberta-base-merges.txt"),
                      #"vocab_size": 250002,
                      "encoder": "RobertaModel",
                      "tokenizer": "XLMRobertaTokenizer",
                      "wordpiece_flag": "▁",
                      "flag_is_first_token": True,
                      #"state_dict_mapping": {"roberta": "encoder",  # "lm_head":,
                      #                       "lm_head.decoder": "head.mlm.predictions.decoder",
                      #                       "lm_head.dense": "head.mlm.predictions.transform.dense",
                      ##                       "lm_head.bias": "head.mlm.predictions.bias",
                      #                       "lm_head.layer_norm": "head.mlm.predictions.transform.LayerNorm"},
                      },


                      "xlm-roberta-base-am":
                        {
                          "model": os.path.join(os.environ.get("PRETRAINED","."), "tuned/10812186-am-xlm-roberta-base"),

                      "vocab": os.path.join(BERT_MODELS_DIRECTORY, "xlm-roberta-base"),#, "roberta-base-merges.txt"),
                      #"vocab_size": 250002,
                      "encoder": "RobertaModel",
                      "tokenizer": "XLMRobertaTokenizer",
                      "wordpiece_flag": "▁",
                      "flag_is_first_token": True,
                      #"state_dict_mapping": {"roberta": "encoder",  # "lm_head":,
                      #                       "lm_head.decoder": "head.mlm.predictions.decoder",
                      #                       "lm_head.dense": "head.mlm.predictions.transform.dense",
                      ##                       "lm_head.bias": "head.mlm.predictions.bias",
                      #                       "lm_head.layer_norm": "head.mlm.predictions.transform.LayerNorm"},
                      },



                      

                    "robert-ckb":
                        {
                          "model": os.path.join(os.environ.get("PRETRAINED","."), "10829903_job-04c76_model-ckb"),
                          "encoder": "RobertaModel",
                          "tokenizer": "RobertaTokenizerFast",
                          "vocab": os.path.join(os.environ.get("PRETRAINED","."), "ckb","tokenizer"),
                          "wordpiece_flag": "▁",
                          "flag_is_first_token": True,
                       },


                    "robert-malt-1M":
                        {
                          "model": os.path.join(os.environ.get("PRETRAINED","."), "10812159_job-d4b2c_model-mt"),
                          "encoder": "RobertaModel",
                          "tokenizer": "RobertaTokenizerFast",
                          "vocab": os.path.join(os.environ.get("PRETRAINED","."), "mt","tokenizer"),
                          "wordpiece_flag": "▁",
                          "flag_is_first_token": True,
                       },

                       "robert-malt-100k":
                        {
                          "model": os.path.join(os.environ.get("PRETRAINED","."), "10812159_job-245b3_model-mt"),
                          "encoder": "RobertaModel",
                          "tokenizer": "RobertaTokenizerFast",
                          "vocab": os.path.join(os.environ.get("PRETRAINED","."), "mt","tokenizer"),
                          "wordpiece_flag": "▁",
                          "flag_is_first_token": True,
                       },

                           "robert-malt-5M":
                        {
                          "model": os.path.join(os.environ.get("PRETRAINED","."), "10812159_job-4e0f7_model-mt/checkpoint-344000"),
                          "encoder": "RobertaModel",
                          "tokenizer": "RobertaTokenizerFast",
                          "vocab": os.path.join(os.environ.get("PRETRAINED","."), "mt","tokenizer"),
                          "wordpiece_flag": "▁",
                          "flag_is_first_token": True,
                       },




                      "robert-fao":
                      {
                          "model": os.path.join(os.environ.get("PRETRAINED","."), "fao"),
                          "encoder": "RobertaModel",
                          "tokenizer": "RobertaTokenizerFast",
                          "vocab": os.path.join(os.environ.get("PRETRAINED","."), "fao","tokenizer"),
                          "wordpiece_flag": "▁",
                          "flag_is_first_token": True,
                       },

                      "robert-gsw":
                      {
                          "model": os.path.join(os.environ.get("PRETRAINED","."), "gsw"),
                          "encoder": "RobertaModel",
                          "tokenizer": "RobertaTokenizerFast",
                          "vocab": os.path.join(os.environ.get("PRETRAINED","."), "gsw","tokenizer"),
                          "wordpiece_flag": "▁",
                          "flag_is_first_token": True,
                       },

                    "robert-malt":
                      {
                          "model": os.path.join(os.environ.get("PRETRAINED","."), "mt"),
                          "encoder": "RobertaModel",
                          "tokenizer": "RobertaTokenizerFast",
                          "vocab": os.path.join(os.environ.get("PRETRAINED","."), "mt","tokenizer"),
                          "wordpiece_flag": "▁",
                          "flag_is_first_token": True,
                       },
                    "robert-ug":
                      {
                          "model": os.path.join(os.environ.get("PRETRAINED","."), "ug"),
                          "encoder": "RobertaModel",
                          "tokenizer": "RobertaTokenizerFast",
                          "vocab": os.path.join(os.environ.get("PRETRAINED","."), "ug","tokenizer"),
                          "wordpiece_flag": "▁",
                          "flag_is_first_token": True,
                       },
                       "robert-tl":
                      {
                          "model": os.path.join(os.environ.get("PRETRAINED","."), "tl"),
                          "encoder": "RobertaModel",
                          "tokenizer": "RobertaTokenizerFast",
                          "vocab": os.path.join(os.environ.get("PRETRAINED","."), "tl","tokenizer"),
                          "wordpiece_flag": "▁",
                          "flag_is_first_token": True,
                       },
                      "robert-am":
                      {
                          "model": os.path.join(os.environ.get("PRETRAINED","."), "am"),
                          "encoder": "RobertaModel",
                          "tokenizer": "RobertaTokenizerFast",
                          "vocab": os.path.join(os.environ.get("PRETRAINED","."), "am","tokenizer"),
                          "wordpiece_flag": "▁",
                          "flag_is_first_token": True,
                       },

                       "robert-narabizi":
                      {
                          "model": os.path.join(os.environ.get("PRETRAINED","."), "narabizi"),
                          "encoder": "RobertaModel",
                          "tokenizer": "RobertaTokenizerFast",
                          "vocab": os.path.join(os.environ.get("PRETRAINED","."), "narabizi","tokenizer"),
                          "wordpiece_flag": "▁",
                          "flag_is_first_token": True,
                       },

                 "bert-cased-multitask": {"vocab": os.path.join(BERT_MODELS_DIRECTORY, "bert-base-cased-vocab.txt"),
                                           "model": os.path.join(BERT_MODELS_DIRECTORY, "bert-cased-multitask.tar.gz"),
                                           "vocab_size": 28996,
                                           "state_dict_mapping": {"bert": "encoder", "cls": "head.mlm"},

                                           },
                  "bert-cased": {"vocab": os.path.join(BERT_MODELS_DIRECTORY, "bert-base-cased-vocab.txt"),
                                 "model": os.path.join(BERT_MODELS_DIRECTORY, "bert-base-cased"),#.tar.gz"),
                                 "vocab_size": 28996,
                                 "state_dict_mapping": {"bert": "encoder", "cls": "head.mlm"},
                                 "encoder": "BertModel",
                                 "tokenizer": "BertTokenizer",
                                 },
                  "random":  {
                      "vocab": os.path.join(BERT_MODELS_DIRECTORY, "bert-base-cased-vocab.txt"),
                      "model": None,
                      "vocab_size": 28996,
                              },
                  "bert_base_multilingual_cased": {
                      "encoder": "BertModel",
                      "tokenizer": "BertTokenizer",
                      "vocab": os.path.join(BERT_MODELS_DIRECTORY, "bert-base-multilingual-cased","bert-base-multilingual-cased-vocab.txt"),
                      "model": os.path.join(BERT_MODELS_DIRECTORY, "bert-base-multilingual-cased"),#.tar.gz),
                      "vocab_size": 119547,
                      "state_dict_mapping": {"bert": "encoder", "cls": "head.mlm"},

                  },

                  "camembert-base": {
                      "vocab": os.path.join(BERT_MODELS_DIRECTORY, "camembert.base.hugs", "sentencepiece.bpe.model"),
                      "model": os.path.join(BERT_MODELS_DIRECTORY, "camembert.base.hugs"),
                      "vocab_size": 32005,
                      "encoder": "RobertaModel",
                      "tokenizer": "CamembertTokenizer",
                      "wordpiece_flag": "▁",
                      "flag_is_first_token": True,
                      "state_dict_mapping": {"roberta": "encoder",
                                             "lm_head.decoder": "head.mlm.predictions.decoder",
                                             "lm_head.dense": "head.mlm.predictions.transform.dense",
                                             "lm_head.bias": "head.mlm.predictions.bias",
                                             "lm_head.layer_norm": "head.mlm.predictions.transform.LayerNorm"}
                  },

                  "camembert-cased-1": {
                      "vocab": os.path.join(BERT_MODELS_DIRECTORY,"camembert.base.hugs", "sentencepiece.bpe.model"),
                      "model": os.path.join(BERT_MODELS_DIRECTORY, "camembert.base.hugs"),
                      "vocab_size": 32005,
                      "encoder": "RobertaModel",
                      "tokenizer": "CamembertTokenizer",
                      "wordpiece_flag": "▁",
                      "flag_is_first_token": True,
                      "state_dict_mapping": {"roberta": "encoder",
                                             "lm_head.decoder": "head.mlm.predictions.decoder",
                                             "lm_head.dense": "head.mlm.predictions.transform.dense",
                                             "lm_head.bias": "head.mlm.predictions.bias",
                                             "lm_head.layer_norm": "head.mlm.predictions.transform.LayerNorm"}
                  },

                  "camembert-cased-oscar-wwm-107075step": {
                      "vocab": os.path.join(BERT_MODELS_DIRECTORY, "roberta_base_oscar_wwm_step107075-hugs",
                                            "sentencepiece.bpe.model"),
                      "model": os.path.join(BERT_MODELS_DIRECTORY, "roberta_base_oscar_wwm_step107075-hugs"),
                      "vocab_size": 32005,
                      "encoder": "RobertaModel",
                      "tokenizer": "CamembertTokenizer",
                      "wordpiece_flag": "▁",
                      "flag_is_first_token": True,
                      "state_dict_mapping": {"roberta": "encoder",  # "lm_head":,
                                             "lm_head.decoder": "head.mlm.predictions.decoder",
                                             "lm_head.dense": "head.mlm.predictions.transform.dense",
                                             "lm_head.bias": "head.mlm.predictions.bias",
                                             "lm_head.layer_norm": "head.mlm.predictions.transform.LayerNorm"},
                  },
                    "camembert": {
                      "vocab": os.path.join(BERT_MODELS_DIRECTORY, "camembert-hugs",
                                            "sentencepiece.bpe.model"),
                      "model": os.path.join(BERT_MODELS_DIRECTORY, "camembert-hugs"),
                      "vocab_size": 32005,
                      "encoder": "RobertaModel",
                      "tokenizer": "CamembertTokenizer",
                      "wordpiece_flag": "▁",
                      "flag_is_first_token": True,
                      "state_dict_mapping": {"roberta": "encoder",  # "lm_head":,
                                             "lm_head.decoder": "head.mlm.predictions.decoder",
                                             "lm_head.dense": "head.mlm.predictions.transform.dense",
                                             "lm_head.bias": "head.mlm.predictions.bias",
                                             "lm_head.layer_norm": "head.mlm.predictions.transform.LayerNorm"},
                  },
                  "camembert-cased-ccnet-wwm-360000step": {
                      "vocab": os.path.join(BERT_MODELS_DIRECTORY, "roberta_base_ccnet_wwm_step360000-hugs",
                                            "sentencepiece.bpe.model"),
                      "model": os.path.join(BERT_MODELS_DIRECTORY, "roberta_base_ccnet_wwm_step360000-hugs"),
                      "vocab_size": 32005,
                      "encoder": "RobertaModel",
                      "tokenizer": "CamembertTokenizer",
                      "wordpiece_flag": "▁",
                      "flag_is_first_token": True,
                      "state_dict_mapping": {"roberta": "encoder",  # "lm_head":,
                                             "lm_head.decoder": "head.mlm.predictions.decoder",
                                             "lm_head.dense": "head.mlm.predictions.transform.dense",
                                             "lm_head.bias": "head.mlm.predictions.bias",
                                             "lm_head.layer_norm": "head.mlm.predictions.transform.LayerNorm"},
                  },
                      "roberta-base": {
                      "vocab": os.path.join(BERT_MODELS_DIRECTORY, "roberta-base"),#, "roberta-base-merges.txt"),
                      "model": os.path.join(BERT_MODELS_DIRECTORY, "roberta-base"),
                      "vocab_size": 50265,
                      "encoder": "RobertaModel",
                      "tokenizer": "RobertaTokenizer",
                      "wordpiece_flag": "▁",
                      "flag_is_first_token": True,
                      "state_dict_mapping": {"roberta": "encoder",  # "lm_head":,
                                             "lm_head.decoder": "head.mlm.predictions.decoder",
                                             "lm_head.dense": "head.mlm.predictions.transform.dense",
                                             "lm_head.bias": "head.mlm.predictions.bias",
                                             "lm_head.layer_norm": "head.mlm.predictions.transform.LayerNorm"},
                      },

                    "xlm-roberta-base": {
                      "vocab": os.path.join(BERT_MODELS_DIRECTORY, "xlm-roberta-base"),#, "roberta-base-merges.txt"),
                      "model": os.path.join(BERT_MODELS_DIRECTORY, "xlm-roberta-base"),
                      "vocab_size": 250002,
                      "encoder": "RobertaModel",
                      "tokenizer": "XLMRobertaTokenizer",
                      "wordpiece_flag": "▁",
                      "flag_is_first_token": True,
                      "state_dict_mapping": {"roberta": "encoder",  # "lm_head":,
                                             "lm_head.decoder": "head.mlm.predictions.decoder",
                                             "lm_head.dense": "head.mlm.predictions.transform.dense",
                                             "lm_head.bias": "head.mlm.predictions.bias",
                                             "lm_head.layer_norm": "head.mlm.predictions.transform.LayerNorm"},
                      },


                      "camembert-cased-oscar-wpm-101029step": {
                      "vocab": os.path.join(BERT_MODELS_DIRECTORY, "roberta_base_oscar_step101029-hugs", "sentencepiece.bpe.model"),
                      "model": os.path.join(BERT_MODELS_DIRECTORY, "roberta_base_oscar_step101029-hugs"),
                      "vocab_size": 32005,
                      "encoder": "RobertaModel",
                      "tokenizer": "CamembertTokenizer",
                      "wordpiece_flag": "▁",
                      "flag_is_first_token": True,
                      "state_dict_mapping": {"roberta": "encoder",  # "lm_head":,
                                             "lm_head.decoder": "head.mlm.predictions.decoder",
                                             "lm_head.dense": "head.mlm.predictions.transform.dense",
                                             "lm_head.bias": "head.mlm.predictions.bias",
                                             "lm_head.layer_norm": "head.mlm.predictions.transform.LayerNorm"},
                  },

                  "camembert-cased-ccnet-_ccnet_wwm_total125000_complete_step100346": {
                      "vocab": os.path.join(BERT_MODELS_DIRECTORY, "roberta_base_ccnet_wwm_total125000_complete_doc_step101402.pytorch","sentencepiece.bpe.model"),
                      "model": os.path.join(BERT_MODELS_DIRECTORY, "roberta_base_ccnet_wwm_total125000_complete_doc_step101402.pytorch"),
                      "vocab_size": 32005,
                      "encoder": "RobertaModel",
                      "tokenizer": "CamembertTokenizer",
                      "wordpiece_flag": "▁",
                      "flag_is_first_token": True,
                      "state_dict_mapping": {"roberta": "encoder",  # "lm_head":,
                                             "lm_head.decoder": "head.mlm.predictions.decoder",
                                             "lm_head.dense": "head.mlm.predictions.transform.dense",
                                             "lm_head.bias": "head.mlm.predictions.bias",
                                             "lm_head.layer_norm": "head.mlm.predictions.transform.LayerNorm"},
                  },

                  "roberta_base_ccnet_wwm_total1000000_complete_lr0.0007_step448000": {
                      "vocab": os.path.join(BERT_MODELS_DIRECTORY,
                                            "roberta_base_ccnet_wwm_total1000000_complete_lr0.0007_step448000", "sentencepiece.bpe.model"),
                      "model": os.path.join(BERT_MODELS_DIRECTORY,
                                            "roberta_base_ccnet_wwm_total1000000_complete_lr0.0007_step448000"),
                      "vocab_size": 32005,
                      "encoder": "RobertaModel",
                      "tokenizer": "CamembertTokenizer",
                      "wordpiece_flag": "▁",
                      "flag_is_first_token": True,
                      "state_dict_mapping": {"roberta": "encoder",  # "lm_head":,
                                             "lm_head.decoder": "head.mlm.predictions.decoder",
                                             "lm_head.dense": "head.mlm.predictions.transform.dense",
                                             "lm_head.bias": "head.mlm.predictions.bias",
                                             "lm_head.layer_norm": "head.mlm.predictions.transform.LayerNorm"},
                  },



                "roberta_base_ccnet_4gb_wwm_total125000_complete_lr0.0007_step123813": {
                      "vocab": os.path.join(BERT_MODELS_DIRECTORY,
                                            "roberta_base_ccnet_4gb_wwm_total125000_complete_lr0.0007_step123813",
                                            "sentencepiece.bpe.model"),
                      "model": os.path.join(BERT_MODELS_DIRECTORY,
                                            "roberta_base_ccnet_4gb_wwm_total125000_complete_lr0.0007_step123813"),
                      "vocab_size": 32005,
                      "encoder": "RobertaModel",
                      "tokenizer": "CamembertTokenizer",
                      "wordpiece_flag": "▁",
                      "flag_is_first_token": True,
                      "state_dict_mapping": {"roberta": "encoder",  # "lm_head":,
                                             "lm_head.decoder": "head.mlm.predictions.decoder",
                                             "lm_head.dense": "head.mlm.predictions.transform.dense",
                                             "lm_head.bias": "head.mlm.predictions.bias",
                                             "lm_head.layer_norm": "head.mlm.predictions.transform.LayerNorm"},
                  },
                "camembert_ccnet_4gb_100k": {
                      "vocab": os.path.join(BERT_MODELS_DIRECTORY,
                                            "roberta_base_ccnet_4gb_wwm_total125000_complete_lr0.0007_step100000",
                                            "sentencepiece.bpe.model"),
                      "model": os.path.join(BERT_MODELS_DIRECTORY,
                                            "roberta_base_ccnet_4gb_wwm_total125000_complete_lr0.0007_step100000"),
                      "vocab_size": 32005,
                      "encoder": "RobertaModel",
                      "tokenizer": "CamembertTokenizer",
                      "wordpiece_flag": "▁",
                      "flag_is_first_token": True,
                      "state_dict_mapping": {"roberta": "encoder",  # "lm_head":,
                                             "lm_head.decoder": "head.mlm.predictions.decoder",
                                             "lm_head.dense": "head.mlm.predictions.transform.dense",
                                             "lm_head.bias": "head.mlm.predictions.bias",
                                             "lm_head.layer_norm": "head.mlm.predictions.transform.LayerNorm"},

                  },
                  "camembert_wiki_4gb": {
                      "vocab": os.path.join(BERT_MODELS_DIRECTORY,
                                            "roberta_base_frwiki_pedro_wwm_total125000_complete_lr0.0007_step100000",
                                            "sentencepiece.bpe.model"),
                      "model": os.path.join(BERT_MODELS_DIRECTORY,
                                            "roberta_base_frwiki_pedro_wwm_total125000_complete_lr0.0007_step100000"),
                      "vocab_size": 32005,
                      "encoder": "RobertaModel",
                      "tokenizer": "CamembertTokenizer",
                      "wordpiece_flag": "▁",
                      "flag_is_first_token": True,
                      "state_dict_mapping": {"roberta": "encoder",  # "lm_head":,
                                             "lm_head.decoder": "head.mlm.predictions.decoder",
                                             "lm_head.dense": "head.mlm.predictions.transform.dense",
                                             "lm_head.bias": "head.mlm.predictions.bias",
                                             "lm_head.layer_norm": "head.mlm.predictions.transform.LayerNorm"},

                  },
                  "camembert_oscar_4gb": {
                      "vocab": os.path.join(BERT_MODELS_DIRECTORY,
                                            "roberta_base_oscar_4gb_wwm_total125000_complete_lr0.0007_step100000",
                                            "sentencepiece.bpe.model"),
                      "model": os.path.join(BERT_MODELS_DIRECTORY,
                                            "roberta_base_oscar_4gb_wwm_total125000_complete_lr0.0007_step100000"),
                      "vocab_size": 32005,
                      "encoder": "RobertaModel",
                      "tokenizer": "CamembertTokenizer",
                      "wordpiece_flag": "▁",
                      "flag_is_first_token": True,
                      "state_dict_mapping": {"roberta": "encoder",  # "lm_head":,
                                             "lm_head.decoder": "head.mlm.predictions.decoder",
                                             "lm_head.dense": "head.mlm.predictions.transform.dense",
                                             "lm_head.bias": "head.mlm.predictions.bias",
                                             "lm_head.layer_norm": "head.mlm.predictions.transform.LayerNorm"},

                  },

                  "roberta_base_ccnet_16gb_wwm_total125000_complete_lr0.0007_step124000": {
                      "vocab": os.path.join(BERT_MODELS_DIRECTORY,
                                            "roberta_base_ccnet_16gb_wwm_total125000_complete_lr0.0007_step124000",
                                            "sentencepiece.bpe.model"),
                      "model": os.path.join(BERT_MODELS_DIRECTORY,
                                            "roberta_base_ccnet_16gb_wwm_total125000_complete_lr0.0007_step124000"),
                      "vocab_size": 32005,
                      "encoder": "RobertaModel",
                      "tokenizer": "CamembertTokenizer",
                      "wordpiece_flag": "▁",
                      "flag_is_first_token": True,
                      "state_dict_mapping": {"roberta": "encoder",  # "lm_head":,
                                             "lm_head.decoder": "head.mlm.predictions.decoder",
                                             "lm_head.dense": "head.mlm.predictions.transform.dense",
                                             "lm_head.bias": "head.mlm.predictions.bias",
                                             "lm_head.layer_norm": "head.mlm.predictions.transform.LayerNorm"},
                  },

                "roberta_base_oscar_100mb_wwm_total125000_complete_lr0.0007_step15600":{
                      "vocab": os.path.join(BERT_MODELS_DIRECTORY,
                                            "roberta_base_oscar_100mb_wwm_total125000_complete_lr0.0007_step15600",
                                            "sentencepiece.bpe.model"),
                      "model": os.path.join(BERT_MODELS_DIRECTORY,
                                            "roberta_base_oscar_100mb_wwm_total125000_complete_lr0.0007_step15600"),
                      "vocab_size": 32005,
                      "encoder": "RobertaModel",
                      "tokenizer": "CamembertTokenizer",
                      "wordpiece_flag": "▁",
                      "flag_is_first_token": True,
                      "state_dict_mapping": {"roberta": "encoder",  # "lm_head":,
                                             "lm_head.decoder": "head.mlm.predictions.decoder",
                                             "lm_head.dense": "head.mlm.predictions.transform.dense",
                                             "lm_head.bias": "head.mlm.predictions.bias",
                                             "lm_head.layer_norm": "head.mlm.predictions.transform.LayerNorm"},
                    },

                  "roberta_large_ccnet_wwm_total125000_complete_lr0.0005_step125000": {
                      "vocab": os.path.join(BERT_MODELS_DIRECTORY,
                                            "roberta_large_ccnet_wwm_total125000_complete_lr0.0005_step125000",
                                            "sentencepiece.bpe.model"),
                      "model": os.path.join(BERT_MODELS_DIRECTORY,
                                            "roberta_large_ccnet_wwm_total125000_complete_lr0.0005_step125000"),
                      "vocab_size": 32005,
                      "encoder": "RobertaModel",
                      "tokenizer": "CamembertTokenizer",
                      "wordpiece_flag": "▁",
                      "flag_is_first_token": True,
                      "state_dict_mapping": {"roberta": "encoder",  # "lm_head":,
                                             "lm_head.decoder": "head.mlm.predictions.decoder",
                                             "lm_head.dense": "head.mlm.predictions.transform.dense",
                                             "lm_head.bias": "head.mlm.predictions.bias",
                                             "lm_head.layer_norm": "head.mlm.predictions.transform.LayerNorm"},
                  },

                  "xlm-mlm-enfr-1024": {
                      "vocab": os.path.join(BERT_MODELS_DIRECTORY, "xlm-mlm-enfr-1024"),
                      "model": os.path.join(BERT_MODELS_DIRECTORY, "xlm-mlm-enfr-1024"),
                      "vocab_size": 1,
                      "encoder": "XLMModel",
                      "tokenizer": "XLMTokenizer",
                      "wordpiece_flag": "</w>",
                      "flag_is_first_token": False,
                      "state_dict_mapping": {"transformer":"encoder"}
                  },

                  "xlm-mlm-tlm-xnli15-1024": {
                      "vocab": os.path.join(BERT_MODELS_DIRECTORY, "xlm-mlm-tlm-xnli15-1024"),
                      "model": os.path.join(BERT_MODELS_DIRECTORY, "xlm-mlm-tlm-xnli15-1024"),
                      "vocab_size": 1,
                      "encoder": "XLMModel",
                      "tokenizer": "XLMTokenizer",
                      "wordpiece_flag": "</w>",
                      "flag_is_first_token": False,
                      "state_dict_mapping": {"transformer": "encoder"}
                  },


                  "xlm-mlm-xnli15-1024": {
                      "vocab": os.path.join(BERT_MODELS_DIRECTORY, "xlm-mlm-xnli15-1024"),
                      "model": os.path.join(BERT_MODELS_DIRECTORY, "xlm-mlm-xnli15-1024"),
                      "vocab_size": 1,
                      "encoder": "XLMModel",
                      "tokenizer": "XLMTokenizer",
                      "wordpiece_flag": "</w>",
                      "flag_is_first_token": False,
                      "state_dict_mapping": {"transformer": "encoder"}
                  },

                    "xlm-mlm-17-1280": {
                      "vocab": os.path.join(BERT_MODELS_DIRECTORY, "xlm-mlm-17-1280"),
                      "model": os.path.join(BERT_MODELS_DIRECTORY, "xlm-mlm-17-1280"),
                      "vocab_size": 1,
                      "encoder": "XLMModel",
                      "tokenizer": "XLMTokenizer",
                      "wordpiece_flag": "</w>",
                      "flag_is_first_token": False,
                      "state_dict_mapping": {"transformer": "encoder"}
                  }

                  }

for step in ["8361", "16721", "25082", "33439", "41802", "50163", "58525", "66886", "75250", "83615", "91979", "100346", "108710", "117077"]:
    checkpoint_name = "roberta_base_ccnet_wwm_total125000_complete_step{}".format(step)
    #checkpoint_name = checkpoint_name
    BERT_MODEL_DIC[checkpoint_name] = {"vocab": os.path.join(BERT_MODELS_DIRECTORY,
                                                             checkpoint_name,
                                                             "sentencepiece.bpe.model"),
                                       "model": os.path.join(BERT_MODELS_DIRECTORY,
                                                             checkpoint_name),
                                       "vocab_size": 32005,
                                       "encoder": "RobertaModel",
                                       "tokenizer": "CamembertTokenizer",
                                       "wordpiece_flag": "▁",
                                       "flag_is_first_token": True,
                                       "state_dict_mapping": {"roberta": "encoder",  # "lm_head":,
                                                              "lm_head.decoder": "head.mlm.predictions.decoder",
                                                              "lm_head.dense": "head.mlm.predictions.transform.dense",
                                                              "lm_head.bias": "head.mlm.predictions.bias",
                                                              "lm_head.layer_norm": "head.mlm.predictions.transform.LayerNorm"
                                                                 }
    }

DIR_2_STAT_MAPPING = OrderedDict([(BERT_MODEL_DIC[key]["model"], BERT_MODEL_DIC[key].get("state_dict_mapping")) for key in BERT_MODEL_DIC])
