from transfer.downstream.finetune.env.imports import CrossEntropyLoss
from transfer.downstream.finetune.io_.dat.constants import PAD_ID_TAG, PAD_ID_HEADS
from transfer.downstream.finetune.model.constants import PAD_ID_LOSS_STANDART
#optimizer setting
AVAILABLE_OPTIMIZER = ["adam", "bahdanu-adadelta", "SGD", "AdamW"]
AVAILABLE_BERT_FINE_TUNING_STRATEGY = ["bert_out_first", "standart", "flexible_lr", "only_first_and_last",
                                       "freeze", "word_embeddings_freeze", "embeddings_freeze", "pos_embeddings_freeze",
                                       "attention_freeze_all", "dense_freeze_all",
                                       "attention_unfreeze_ponderation_only", "attention_freeze_ponderation", "encoder_freeze",
                                         "layer_specific"]
MULTI_TASK_LOSS_PONDERATION_PREDEFINED_MODE = []
AVAILALE_PENALIZATION_MODE = ["layer_wise", "pruning"]
AVAILABLE_INPUTS = ["word", "wordpiece", "char"]


# architecture setting
TASKS_PARAMETER = {

                   "pos": {"normalization": False,
                           "default_metric": "accuracy-pos",
                           "pred": ["pos_pred"],
                            "num_labels_mandatory":True,
                           # a list per prediction
                           "eval_metrics": [["accuracy-pos-pos"]],
                           "subsample-allowed": ["all", "NEED_NORM", "NORMED", "PRED_NEED_NORM", "PRED_NORMED", "InV", "OOV"],
                           "label": ["pos"],
                           # because its the label of mwe prediction
                           "input": "wordpieces_inputs_words",
                           "alignement": "wordpieces_inputs_words_alignement",
                           "head": "BertTokenHead",
                           "prediction_level": "word",
                           "loss": CrossEntropyLoss(ignore_index=PAD_ID_TAG, reduce="sum")
                           },
                   "mlm": {"normalization": False,
                           "mask_input": True,# means the sequence input is always masked following mlm (train and test!)
                           "default_metric": "accuracy-mlm",
                           "default-subsample": "mlm",
                           "subsample-allowed": ["all", "InV", "OOV", "mlm"],
                           "num_labels_mandatory": False,
                           # a list per prediction
                           "eval_metrics": [["accuracy-mlm-wordpieces_inputs_words"]],
                           "label": ["wordpieces_inputs_words"],
                           # because its the label of mwe prediction
                           "input": "input_masked",
                           "alignement": "wordpieces_inputs_words_alignement",
                           "original": "wordpieces_inputs_words",
                           "head": "BertOnlyMLMHead",
                           "prediction_level": "bpe",
                           "loss": CrossEntropyLoss(ignore_index=PAD_ID_LOSS_STANDART, reduce="sum")
                           },
                   "parsing": {
                       "normalization": False,
                       "default_metric": None,
                       "num_labels_mandatory": True,
                       "num_labels_mandatory_to_check": ["types"],
                       "eval_metrics": [["accuracy-parsing-heads"], ["accuracy-parsing-types"]],
                       "head": "BertGraphHead",
                        "subsample-allowed":  ["all", "InV", "OOV"],
                        # because its the label of mwe prediction
                       "input": "wordpieces_inputs_words",
                       "alignement": "wordpieces_inputs_words_alignement",
                       "label": ["heads", "types"],
                       "prediction_level": "word",
                       "loss": CrossEntropyLoss(ignore_index=PAD_ID_TAG, reduction="sum")
                   },
}

# reporting given prediction labels (task-label) provides the subsample to evaluate on
SAMPLES_PER_TASK_TO_REPORT = {
            "pos-pos": ["all", "NEED_NORM", "NORMED", "PRED_NEED_NORM", "PRED_NORMED", "InV", "OOV"],
            "parsing-heads": ["all", "InV", "OOV"],
            "parsing-types": ["all", "InV", "OOV"],
            "mlm-wordpieces_inputs_words": ["all", "InV", "OOV", "mlm"],
}

# label related
LABEL_PARAMETER = {"heads": {"graph_label": True, "pad_value": PAD_ID_HEADS, "bpe": False, "unicode_to_string": True,"default_input":"wordpieces_inputs_words",
                             "realignement_mode": "ignore_non_first_bpe"
                             },
                   "types": {"graph_label": False, "pad_value": PAD_ID_TAG, "bpe": False,"default_input": "wordpieces_inputs_words",
                             "realignement_mode": "ignore_non_first_bpe"},
                   "pos": {"graph_label": False, "pad_value": PAD_ID_TAG, "bpe": False, "realignement_mode": "ignore_non_first_bpe",
                           "default_input": "wordpieces_inputs_words"},
                   "wordpiece_normalization_target_aligned_with_word":
                       {"graph_label": False, "pad_value": PAD_ID_LOSS_STANDART, "bpe": True,
                        "realignement_mode": "detokenize_bpe", "default_input":"wordpiece_words_src_aligned_with_norm"},
                   "wordpieces_inputs_words": {"graph_label": False,
                                               "default_input": "wordpieces_inputs_words",
                                               "pad_value": PAD_ID_LOSS_STANDART, "bpe": True,
                                               "realignement_mode": "detokenize_bpe"},
                   }
