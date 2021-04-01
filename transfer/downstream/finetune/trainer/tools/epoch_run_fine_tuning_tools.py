from transfer.downstream.finetune.env.imports import torch, re, OrderedDict, pdb
from transfer.downstream.finetune.io_.logger import printing
from transfer.downstream.finetune.model.settings import TASKS_PARAMETER
from transfer.downstream.finetune.model.constants import SPECIAL_TOKEN_LS
from transfer.downstream.finetune.io_.dat.normalized_writer import write_conll, write_conll_multitask


def get_casing(case, batch, task_normalize_is, cls_token=None, sep_token=None):
    if case is not None:
        if case == "lower":
            batch.raw_input = [[word.lower() if word not in SPECIAL_TOKEN_LS+[cls_token, sep_token] else word for word in sent] for sent in batch.raw_input]
            if task_normalize_is:
                batch.raw_output = [[word.lower() if word not in SPECIAL_TOKEN_LS+[cls_token, sep_token] else word for word in sent] for sent in batch.raw_output]
    return batch


def logging_processing_data(_verbose, verbose, verbose_level, batch_raw_input, input_tokens_tensor, batch_raw_output, output_tokens_tensor, inp_bpe_tokenized, out_bpe_tokenized):
    printing("DATA : pre-tokenized input {} ", var=[batch_raw_input], verbose_level=verbose_level, verbose=_verbose)
    printing("DATA : BPEtokenized input ids {}", var=[input_tokens_tensor], verbose_level=3, verbose=verbose)

    printing("DATA : pre-tokenized output {} ", var=[batch_raw_output], verbose_level=verbose_level, verbose=_verbose)
    printing("DATA : BPE tokenized output ids  {}", var=[output_tokens_tensor], verbose_level=4, verbose=verbose)
    # BPE
    printing("DATA : BPE tokenized input  {}", var=[inp_bpe_tokenized], verbose_level=4, verbose=_verbose)
    printing("DATA : BPE tokenized output  {}", var=[out_bpe_tokenized], verbose_level=4, verbose=_verbose)



def log_data_src_label_pred(src_detokenized_dic, predict_detokenize_dic, label_detokenized_dic, tasks, verbose, verbose_level):

    if isinstance(verbose, int) or verbose == "alignment":
        if verbose == "alignment" or verbose >= verbose_level:
            for task in [_task for _tasks in tasks for _task in _tasks]:
                input_name = TASKS_PARAMETER[task]["input"]
                label_name_ls = TASKS_PARAMETER[task]["label"]

                for ind_src_sent, src_sent in enumerate(src_detokenized_dic[input_name]):
                    print("      ")
                    for label in label_name_ls:
                        try:
                            assert len(predict_detokenize_dic[task + "-" + label][0][ind_src_sent]) == len(label_detokenized_dic[label][ind_src_sent]), \
                                "ERROR pred {} label {} ".format(predict_detokenize_dic[task + "-" + label][ind_src_sent], label_detokenized_dic[label][ind_src_sent])
                            assert len(src_detokenized_dic[input_name][ind_src_sent]) == len(label_detokenized_dic[label][ind_src_sent]), "ERROR "
                            for ind_src, src in enumerate(src_sent):
                                to_print = "SRC : {} ,    ".format(src) + " ".join(["PRED:{}  GOLD:{} (label {})".format(predict_detokenize_dic[task + "-" + label][0][ind_src_sent][ind_src], label_detokenized_dic[label][ind_src_sent][ind_src], label) for label in label_name_ls])
                                printing(to_print, verbose=1, verbose_level=1)
                        except Exception as e:
                            print("ERROR : not aligned labels so cannot log ", e)



            #for sent_src, sent_gold, sent_pred in zip()


def print_align_bpe(source_preprocessed, gold, input_alignement_with_raw, labels_n_mask_prediction,
                    verbose, verbose_level):
    if labels_n_mask_prediction is None:
        labels_n_mask_prediction = [[None for _ in range(len(sent))] for sent in
                                    input_alignement_with_raw]
    if isinstance(verbose, int) or verbose == "alignement":
        if verbose == "alignement" or verbose >= verbose_level:
            assert len(source_preprocessed) == len(gold), ""
            assert len(input_alignement_with_raw) == len(gold), ""
            for sent_src, sent_gold, index_match_with_src, append_masks in zip(source_preprocessed,
                                                                               gold,
                                                                               input_alignement_with_raw,
                                                                               labels_n_mask_prediction):
                assert len(sent_src) == len(sent_gold)
                assert len(sent_src) == len(sent_gold)
                for src, gold_tok, index, masks in zip(sent_src, sent_gold, index_match_with_src,
                                                       append_masks):
                    printing("{}:{} --> {} (n_masks {})", var=[index, src, gold_tok, masks],
                             verbose=1, verbose_level=1)


def log_warning(counting_failure_parralel_bpe_batch, data_label, batch_i, batch, noisy_under_splitted,
                skipping_batch_n_to_1, aligned, noisy_over_splitted, skip_1_t_n, skipping_evaluated_batch, verbose):
    printing("WARNING {} aignement failure caused by parallel ", var=[counting_failure_parralel_bpe_batch],
             verbose=verbose, verbose_level=1)
    printing(
        "WARNING on {} : Out of {} batch of X sentences each {} skipped ({} batch aligned ; {} with at least 1 sentence noisy MORE SPLITTED ; {} with  LESS SPLITTED {} + SENT with skipped_1_to_n : {}) ",
        var=[data_label, batch_i, noisy_under_splitted + skipping_batch_n_to_1, aligned,
             noisy_over_splitted, noisy_under_splitted, "SKIPPED" if skip_1_t_n else "", skipping_batch_n_to_1],
        verbose=verbose, verbose_level=1)
    printing("WARNING on {} ON THE EVALUATION SIDE we skipped extra {} batch ",
             var=[data_label, skipping_evaluated_batch], verbose_level=1, verbose=verbose)


def tensorboard_loss_writer_batch_level_multi(writer, mode, model_id, _loss, batch_i, iter, loss_dic, tasks):

    writer.add_scalars("loss-batch-sum",
                       {"loss-{}-{}-bpe".format(mode, model_id): _loss.clone().cpu().data.numpy()
                       if not isinstance(_loss, int) else 0},
                       iter+batch_i)
    for label in loss_dic:
        writer.add_scalars("loss-batch-{}".format(label),
                           {"loss-{}-{}-bpe".format(mode, model_id): loss_dic[label].detach().clone().cpu().data.numpy()},
                           iter + batch_i)


def tensorboard_loss_writer_batch_level(writer, mode, model_id, _loss, batch_i, iter, loss_dic, task_normalize_is,  append_n_mask):
    writer.add_scalars("loss-batch-sum",
                       {"loss-{}-{}-bpe".format(mode, model_id): _loss.detach().clone().cpu().data.numpy()
                       if not isinstance(_loss, int) else 0},
                       iter+batch_i)

    if task_normalize_is:
        writer.add_scalars("loss-batch-norm",
                           {"loss-{}-{}-bpe".format(mode, model_id):
                                loss_dic["loss_task_1"].detach().clone().cpu().data.numpy()
                            },
                           iter + batch_i)
        if append_n_mask:

            writer.add_scalars("loss-batch-norm-pred_n_mask",
                               {"loss-{}-{}-pred_n_mask".format(mode, model_id):
                                    loss_dic["loss_task_n_mask_prediction"].detach().clone().cpu().data.numpy()
                                },
                               iter + batch_i)


def update_loss_dic_average(loss_dic_current, loss_dic_total):

    assert set(loss_dic_current.keys()).issubset(set(loss_dic_total.keys())), "ERROR : mismatch keys {} and {} ".format(loss_dic_current, loss_dic_total)

    for loss_label, value in loss_dic_current.items():
        loss_dic_total[loss_label] += value.item()

    return loss_dic_total


def tensorboard_loss_writer_epoch_level_multi(writer, mode, model_id, epoch,
                                              loss_dic, n_tokens_dic, data_label, penalization_dic=None, group_mapping=None,
                                              grad_report_dic=None, grad_std_report_dic=None, verbose=1):
    """
    NB : loss provided is already supposed to be average per batch
    :param writer:
    :param tasks:
    :param mode:
    :param model_id:
    :param epoch:
    :param n_batch_norm:
    :param n_batch_pos:
    :param append_n_mask:
    :param loss:
    :param loss_norm:
    :param loss_pos:
    :param loss_n_mask_prediction:
    :param batch_i:
    :return:
    """
    try:
        assert set(loss_dic.keys()) == set(n_tokens_dic.keys()), "ERROR keys mismatching between loss and n_tokens {} {}".format(loss_dic, n_tokens_dic)
    except Exception as e:
        print(e)
    if penalization_dic is not None:
        for penalize_lab, penalization in penalization_dic.items():
            penalization_val = penalization[2]
            n_param = penalization[0]
            ponderation = penalization[1]
            try:
                penalization_val_not_prune = penalization[3]
                penalization_val_prune = penalization[4]
            except:
                penalization_val_not_prune = 0
                penalization_val_prune = 0

            if group_mapping is not None:
                # we aggregate based on group mapping
                def get_group(group_mapping, penalize_lab):
                    for group_regex in group_mapping:
                        if re.match(group_regex, penalize_lab) is not None:
                            return group_regex
                group_layer = get_group(group_mapping, penalize_lab)
            else:
                group_layer = penalize_lab

            writer.add_scalars("penalization-{}-{}_ponderation".format(group_layer, ponderation),
                               {"{}-{}-{}".format("penalization", penalize_lab, model_id): penalization_val}, epoch)
            writer.add_scalars("penalization-{}-{}_ponderation".format(group_layer, ponderation),
                               {"{}-{}-{}".format("penalization-pruned", penalize_lab, model_id): penalization_val_prune}, epoch)
            writer.add_scalars("penalization-{}-{}_ponderation".format(group_layer, ponderation),
                               {"{}-{}-{}".format("penalization-not-prune", penalize_lab, model_id): penalization_val_not_prune}, epoch)

    if grad_report_dic is not None and grad_std_report_dic is not None:

        for layer, layer_grad_mean in grad_report_dic.items():
            writer.add_scalars("grad_norm",
                                {"{}-{}-{}".format("grad", layer, model_id): layer_grad_mean.data.numpy()}, epoch)
        for layer, layer_grad_std in grad_std_report_dic.items():
            writer.add_scalars("grad_std",
                               {"{}-{}-{}".format("grad", layer, model_id): layer_grad_std.data.numpy()}, epoch)

    loss_lab_not_reported_ls = []
    for loss_lab, loss_val in loss_dic.items():
        try:
            writer.add_scalars("loss-multitask-epoch-{}-{}".format(loss_lab, mode),
                               {"{}-{}-{}-{}".format("loss", mode, data_label, model_id): loss_val.detach()/n_tokens_dic[loss_lab]}, epoch)
        except:
            loss_lab_not_reported_ls.append((loss_lab, loss_val, n_tokens_dic[loss_lab]))
    if verbose>1:
        print("WARNING : could not report loss in "
          "tensorboard for epoch {}  data {} : task {} ".format(epoch, data_label, loss_lab_not_reported_ls))


def tensorboard_loss_writer_epoch_level(writer, tasks, mode, model_id, epoch, n_batch_norm, n_batch_pos, append_n_mask, loss, loss_norm, loss_pos, loss_n_mask_prediction, batch_i, data_label):
    """
    NB : loss provided is already supposed to be average per batch
    :param writer:get_penalization
    :param tasks:
    :param mode:
    :param model_id:
    :param epoch:
    :param n_batch_norm:
    :param n_batch_pos:
    :param append_n_mask:
    :param loss:
    :param loss_norm:
    :param loss_pos:
    :param loss_n_mask_prediction:
    :param batch_i:
    :return:
    """
    writer.add_scalars("loss-overall-epoch-{}-{}".format("_".join([task for _tasks in tasks for task in _tasks]), mode),
                       {"{}-{}-{}-{}".format("loss", mode, data_label, model_id): loss/batch_i}, epoch)
    if "normalize" in tasks:
        try:
            writer.add_scalars("loss-norm-epoch",
                       {"loss-{}-{}-bpe".format(mode, model_id): loss_norm.clone().cpu().data.numpy()/n_batch_norm},
                       epoch)
        except Exception as e:
            print("ERROR {} loss_pos is , n_batch_pos is {} coud not log ".format(e, loss_norm, n_batch_norm))
        if append_n_mask:
            writer.add_scalars("loss-n_mask_prediction-epoch",
                               {"loss-{}-{}-n_mask_prediction".format(mode,
                                model_id): loss_n_mask_prediction.clone().cpu().data.numpy()/n_batch_norm},
                               epoch)
    if "pos" in tasks:
        try:
            writer.add_scalars("loss-pos-epoch",
                       {"loss-{}-{}-bpe".format(mode, model_id): loss_pos.clone().cpu().data.numpy()/n_batch_pos},
                       epoch)
        except Exception as e:
            print("ERROR {} loss_pos is , n_batch_pos is {} coud not log ".format(e, loss_pos, n_batch_pos))


def writing_predictions_conll(dir_normalized, dir_normalized_original_only, dir_gold, dir_gold_original_only,
                              src_detokenized, inverse_writing, pred_detokenized_topk,  iter, batch_i,
                              new_file, gold_detokenized, verbose):

    write_conll(format="conll", dir_normalized=dir_normalized,
                dir_original=dir_normalized_original_only,
                src_text_ls=src_detokenized, inverse=inverse_writing,
                text_decoded_ls=pred_detokenized_topk[0],  # pred_pos_ls=None, src_text_pos=None,
                tasks=["normalize"], ind_batch=iter + batch_i, new_file=new_file,
                src_text_pos=src_detokenized, pred_pos_ls=gold_detokenized,
                verbose=verbose)
    write_conll(format="conll", dir_normalized=dir_gold, dir_original=dir_gold_original_only,
                src_text_ls=src_detokenized,
                src_text_pos=src_detokenized, pred_pos_ls=gold_detokenized,
                text_decoded_ls=gold_detokenized,  # pred_pos_ls=None, src_text_pos=None,
                tasks=["normalize"],
                ind_batch=iter + batch_i, new_file=new_file, verbose=verbose)
    new_file = False
    return new_file


def writing_predictions_conll_multi(dir_pred, dir_normalized_original_only,
                                    dir_gold, dir_gold_original_only,
                                    src_detokenized, pred_per_task,
                                    iter, batch_i, tasks, all_indexes,task_parameters,
                                    new_file, gold_per_tasks, verbose, cls_token=None, sep_token=None):

    write_conll_multitask(format="conll", dir_pred=dir_pred,
                          dir_original=dir_normalized_original_only,
                          src_text_ls=src_detokenized,
                          tasks=tasks, ind_batch=iter + batch_i, new_file=new_file,
                          pred_per_task=pred_per_task,
                          gold=False, cls_token=cls_token, sep_token=sep_token,
                          task_parameters=task_parameters,
                          all_indexes=all_indexes,
                          verbose=verbose)
    write_conll_multitask(format="conll", dir_pred=dir_gold,
                          dir_original=dir_gold_original_only, tasks=tasks,
                          src_text_ls=src_detokenized,
                          pred_per_task=gold_per_tasks, gold=True,
                          all_indexes=all_indexes,cls_token=cls_token, sep_token=sep_token,
                          task_parameters=task_parameters,
                          ind_batch=iter + batch_i, new_file=new_file, verbose=verbose)

    return False


def get_task_name_based_on_logit_label(logit_label, label_processed):
    match = re.match("(.*)-(.*)", logit_label)
    assert match  is not None, "ERROR {}".format(logit_label)
    label = match.group(2)
    task = match .group(1)
    #else:
    #    label = logit_label
    _continue = False
    if label in label_processed:
        _continue = True
    else:
        _continue = False
        label_processed.append(label)
    return label, task, _continue, label_processed


def get_task_label(tasks, task_settings):
    list_label_score = []
    for task in tasks:
        #list_label_score.extend(task_settings[task]["label"])
        list_label_score.extend([task+"-"+labe for labe in task_settings[task]["label"]])
    return list_label_score


def init_score_token_sent_dict(samples_per_task_reporting, tasks, agg_func_ls, compute_intersection_score, task_settings):

    init_samples_per_task = {}
    #import pdb
    #pdb.set_trace()

    labels = get_task_label(tasks, task_settings)

    for task in labels:
        # TODO : standartize make more standart
        #if task.startswith("mwe") or task.startswith("mlm"):
        init_samples_per_task[task] = samples_per_task_reporting[task].copy()
        #else:
        #    init_samples_per_task[task] = samples_per_task_reporting["normalize"].copy()

    if compute_intersection_score:
        for task in labels:
            for ind, sam in enumerate(init_samples_per_task[task][1:]):
                for ind_2 in range(ind):
                    init_samples_per_task[task].append(sam+"-n-"+init_samples_per_task[task][ind_2+1])

    score_dic = {task: {agg_func: {sample: 0 for sample in init_samples_per_task[task]} for agg_func in agg_func_ls} for task in labels}

    n_tokens_dic = {task: {agg_func: {sample: 0 for sample in init_samples_per_task[task]} for agg_func in agg_func_ls} for task in labels}
    n_sents_dic = {task: {agg_func: {sample: 0 for sample in init_samples_per_task[task]} for agg_func in agg_func_ls} for task in labels}
    if "normalize" in tasks:
        for extra_label in ["n_masks_pred", "normalize_pred"]:
            score_dic[extra_label] = {"sum": {sample: 0 for sample in samples_per_task_reporting[extra_label]}}
            n_tokens_dic[extra_label] = {"sum": {sample: 0 for sample in samples_per_task_reporting[extra_label]}}
            n_sents_dic[extra_label] = {"sum": {sample: 0 for sample in samples_per_task_reporting[extra_label]}}

    return score_dic, n_tokens_dic, n_sents_dic


def dimension_check_label(label_per_task, input_tokens_tensor):
    for task, labels in label_per_task.items():
        labels.size(0) == input_tokens_tensor.size(
            0), "task {} output_tokens_tensor_aligned.size(0) {} input_tokens_tensor.size() {}".format(task,
                                                                                                       labels.size(),
                                                                                                       input_tokens_tensor.size())
        labels.size(1) == input_tokens_tensor.size(
            1), "task {} output_tokens_tensor_aligned.size(1) {} input_tokens_tensor.size() {}".format(task,
                                                                                                       labels.size(
                                                                                                           1),
                                                                                                       input_tokens_tensor.size(
                                                                                                           1))


def extend_input(masks, input, input_alignement_with_raw, mask_token_index, use_gpu):
    """
    extend input based on predicted masks
    :param masks: predicted number of inputs
    :param input:
    :param input_alignement_with_raw:
    :param mask_token_index:
    :param use_gpu:
    :return:
    """
    assert masks.size(0) == input.size(0)
    assert masks.size(1) == input.size(1)
    extended_input = []
    extended_alignement = []
    max_len = 0
    for ind_sent in range(masks.size(0)):
        extended_input_sent = []
        extended_alignement_sent = []
        for ind_tok in range(input.size(1)):
            # we account 0 prediction as 1 for prediction
            if masks[ind_sent, ind_tok].item() > 1:
                extended_input_sent.append(input[ind_sent, ind_tok].item())
                extended_input_sent.extend([mask_token_index for _ in range(masks[ind_sent, ind_tok]-1)])
                extended_alignement_sent.extend([input_alignement_with_raw[ind_sent, ind_tok].item() for _ in range(masks[ind_sent, ind_tok])])
            else:
                extended_input_sent.append(input[ind_sent, ind_tok].item())
                extended_alignement_sent.extend([input_alignement_with_raw[ind_sent, ind_tok].item() for _ in range(max(masks[ind_sent, ind_tok],1))])
            max_len = max(len(extended_input_sent), max_len)
        extended_input.append(extended_input_sent)
        extended_alignement.append(extended_alignement_sent)
    # add padding
    extended_input = [sent + [0 for _ in range(max_len - len(sent))] for sent in extended_input]
    extended_alignement_sent = [sent_alignement + [1000 for _ in range(max_len - len(sent_alignement))] for
                                sent_alignement in extended_alignement]

    extended_input_torch = torch.tensor(extended_input)
    extended_alignement_sent_torch = torch.tensor(extended_alignement_sent)

    if use_gpu:
        extended_input_torch = extended_input_torch.cuda()
        extended_alignement_sent_torch = extended_alignement_sent_torch.cuda()

    return extended_input_torch, extended_alignement_sent_torch


def count_tokens(task_ls, n_tokens_counter_per_task, label_per_task, label_paremeter):
    n_tokens_counter_current_per_task = OrderedDict()
    """"get exact number of non-pad tokens for each tasks"""
    for task in task_ls:
        for label in TASKS_PARAMETER[task]["label"]:
            n_tokens_counter_current_per_task[task + "-" + label] = (label_per_task[label] != label_paremeter[label]["pad_value"]).sum().item()
            n_tokens_counter_per_task[task + "-" + label] += n_tokens_counter_current_per_task[task + "-" + label]
    ## TODO : handle in a more standart way
    n_tokens_all = n_tokens_counter_current_per_task[task+"-"+label]
    return n_tokens_counter_per_task, n_tokens_counter_current_per_task, n_tokens_all


def loss_mean(loss_dic, n_tokens_counter_current_per_task):
    for val in loss_dic:
        loss_dic[val] /= n_tokens_counter_current_per_task[val]
    return loss_dic
