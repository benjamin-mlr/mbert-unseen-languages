from transfer.downstream.finetune.env.imports import OrderedDict, torch, pdb, re

from transfer.downstream.finetune.model.constants import PAD_ID_LOSS_STANDART

from transfer.downstream.finetune.model.settings import LABEL_PARAMETER

from transfer.downstream.finetune.io_.bert_iterators_tools.string_processing import from_bpe_token_to_str
import transfer.downstream.finetune.io_.bert_iterators_tools.alignement as alignement

from transfer.downstream.finetune.trainer.tools.epoch_run_fine_tuning_tools import get_task_name_based_on_logit_label




def get_prediction(logits_dic, topk):
    assert topk == 1
    predictions_topk_dic = OrderedDict()
    for logit_label, logits in logits_dic.items():
        #we handle parsing_types in a specific way
        if logit_label == "parsing-types":
            batch_size = logits.size(0)
            # getting predicted heads (to know which labels of the graph which should look at
            pred_heads = predictions_topk_dic["parsing-heads"][:, :, 0]
            # we extract from the logits only the one of the predicted heads (that are not PAD_ID_LOSS_STANDART : useless)
            logits = logits[(pred_heads != PAD_ID_LOSS_STANDART).nonzero()[:, 0],
                            (pred_heads != PAD_ID_LOSS_STANDART).nonzero()[:, 1], pred_heads[pred_heads != PAD_ID_LOSS_STANDART]]
            # we take the argmax label of this heads
            predictions_topk_dic[logit_label] = torch.argsort(logits, dim=-1, descending=True)[:, :topk]
            # only keeping the top 1 prediction
            # predictions_topk_dic[logit_label] = predictions_topk_dic[logit_label][:, 0]
            # reshaping
            predictions_topk_dic[logit_label] = predictions_topk_dic[logit_label].view(batch_size, -1, topk)
        else:
            predictions_topk_dic[logit_label] = torch.argsort(logits, dim=-1, descending=True)[:, :, :topk]

    return predictions_topk_dic


def get_bpe_string(predictions_topk_dic,
                   output_tokens_tensor_aligned_dic, input_tokens_tensor_per_task, topk,
                   tokenizer, task_to_label_dictionary, null_str, null_token_index, task_settings, mask_index, verbose):

    predict_dic = OrderedDict()
    source_preprocessed = OrderedDict()
    label_dic = OrderedDict()

    input_already_processed = []
    gold_already_processed = []
    for task_label in predictions_topk_dic:

        label = re.match("(.*)-(.*)", task_label)
        assert label is not None, "ERROR : {} task_label does not fit the right template (.*)-.* ".format(task_label)
        task = label.group(1)
        label = label.group(2)
        #task_settings[]
        sent_ls_top = from_bpe_token_to_str(predictions_topk_dic[task_label], topk, tokenizer=tokenizer,
                                            pred_mode=True, task=task, mask_index=mask_index,
                                            bpe_tensor_src=input_tokens_tensor_per_task["input_masked"] if task == "mlm" else None,
                                            label_dictionary=task_to_label_dictionary[task_label],
                                            get_bpe_string=LABEL_PARAMETER[label]["bpe"],
                                            label=label, null_token_index=null_token_index, null_str=null_str)
        # some tasks may share same outputs : we don't want to post-process them several times
        if label in gold_already_processed:
            continue
        if output_tokens_tensor_aligned_dic[label] is not None:
            gold_already_processed.append(label)
            gold = from_bpe_token_to_str(output_tokens_tensor_aligned_dic[label], topk, tokenizer=tokenizer, task=task,
                                         label_dictionary=task_to_label_dictionary[task_label],
                                         pred_mode=False, get_bpe_string=LABEL_PARAMETER[label]["bpe"], label=label,
                                         null_token_index=null_token_index, null_str=null_str)
            label_dic[label] = gold
        else:
            label_dic[label] = None

        predict_dic[task_label] = sent_ls_top
        input_label = task_settings[task]["input"]
        input_tokens_tensor = input_tokens_tensor_per_task[input_label]
        #for input_label, input_tokens_tensor in input_tokens_tensor_per_task.items():
        # some tasks may share same inputs : we don't want to post-process them several times
        if input_label in input_already_processed:
            continue
        input_already_processed.append(input_label)

        source_preprocessed[input_label] = from_bpe_token_to_str(input_tokens_tensor, topk, tokenizer=tokenizer, label_dictionary=task_to_label_dictionary[task_label], pred_mode=False, task=task, null_token_index=null_token_index, null_str=null_str, get_bpe_string=True, verbose=verbose)

    return source_preprocessed, label_dic, predict_dic


def get_detokenized_str(source_preprocessed_dict, input_alignement_with_raw, label_dic, predict_dic,
                        mask_str, end_token,
                        null_str, remove_mask_str_prediction, task_settings, batch=None,
                        flag_is_first_token=False, flag_word_piece_token="##", verbose=1):

    # de-BPE-tokenize
    predict_detokenize_dic = OrderedDict()
    label_detokenized_dic = OrderedDict()
    src_detokenized_dic = OrderedDict()

    for source_label, source_preprocessed in source_preprocessed_dict.items():

        if source_label == "wordpieces_inputs_words":
            _input_alignement_with_raw = input_alignement_with_raw
        else:
            # TODO !!! to make standart !
            _source_label = "wordpieces_inputs_words" if source_label == "input_masked" else source_label
            # cause it corresponds to parsing, pos inputs
            _input_alignement_with_raw = eval("batch.{}_alignement".format(_source_label))
            # we cut based on source to be consistent with former definition
            _input_alignement_with_raw = _input_alignement_with_raw[:, :len(source_preprocessed[0])]
        #pdb.set_trace()
        src_detokenized_dic[source_label] = alignement.realigne_multi(source_preprocessed,
                                                                      _input_alignement_with_raw,
                                                                      null_str=null_str, label="wordpieces_inputs_words",
                                                                      # normalize means we deal with bpe input not pos
                                                                      mask_str=mask_str, end_token=end_token,
                                                                      flag_is_first_token=flag_is_first_token,
                                                                      flag_word_piece_token=flag_word_piece_token,
                                                                      gold_sent=True, remove_mask_str=True,
                                                                      remove_extra_predicted_token=True,
                                                                      #remove_mask_str=remove_mask_str_prediction,
                                                                      keep_mask=True if source_label == "input_masked" else False,
                                                                      verbose=verbose
                                                                      )

    label_processed = []
    for label_task in predict_dic:
        label, task,  _continue, label_processed = get_task_name_based_on_logit_label(label_task, label_processed)
        if _continue:
            continue
        label_processed.append(label)
        _input_alignement_with_raw = eval("batch.{}".format(task_settings[task]["alignement"]))

        if label_dic[label] is not None:
            label_detokenized_dic[label] = alignement.realigne_multi(label_dic[label], _input_alignement_with_raw,
                                                                 remove_null_str=True, null_str=null_str,
                                                                 gold_sent=True, remove_extra_predicted_token=True,
                                                                 remove_mask_str=True,
                                                                 flag_is_first_token=flag_is_first_token,
                                                                 flag_word_piece_token=flag_word_piece_token,
                                                                 label=label,
                                                                 mask_str=mask_str, end_token=end_token)
        else:
            label_detokenized_dic[label] = None

        predict_detokenize_dic[label_task] = []
        # handle several prediction
        for sent_ls in predict_dic[label_task]:
            predict_detokenize_dic[label_task].append(alignement.realigne_multi(sent_ls, _input_alignement_with_raw,
                                                                                remove_null_str=True,
                                                                                label=label,
                                                                                flag_is_first_token=flag_is_first_token,
                                                                                flag_word_piece_token=flag_word_piece_token,
                                                                                remove_extra_predicted_token=True,
                                                                                null_str=null_str,
                                                                                mask_str=mask_str, end_token=end_token))
        if label_detokenized_dic[label] is not None:
            for e in range(len(label_detokenized_dic[label])):
                try:
                    assert len(predict_detokenize_dic[label_task][0][e]) == len(label_detokenized_dic[label][e]),\
                        "ERROR : for label {} len pred {} len gold {}".format(label, len(predict_detokenize_dic[label_task][0][e]), len(label_detokenized_dic[label][e]))
                except Exception as err:
                    print("ERROR : gold and pred are not aligned anymore", err)
                    raise(err)

    return src_detokenized_dic, label_detokenized_dic, predict_detokenize_dic


def get_aligned_output(label_per_task):
    output_tokens_tensor_aligned_dict = OrderedDict()
    for label in label_per_task:
        if label != "normalize" :
            output_tokens_tensor_aligned_dict[label] = label_per_task[label]

    return output_tokens_tensor_aligned_dict

