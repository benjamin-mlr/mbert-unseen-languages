from transfer.downstream.finetune.env.imports import pdb

from transfer.downstream.finetune.evaluate.score.compute_score import word_level_scoring, word_level_filter
from transfer.downstream.finetune.evaluate.score.aggregate_score import agg_func_batch_score

AVAILABLE_EVALUATION_SAMPLE_FILTER = ["all"]


def get_intersection_score(samples):
    intersected_samples = []
    sample_to_intersesct = [sam for sam in samples if sam != "all"]
    for ind_sample, _sample in enumerate(sample_to_intersesct):
        for ind_sample_2 in range(ind_sample):
            inter = _sample + "-n-" + sample_to_intersesct[ind_sample_2]
            intersected_samples.append(inter)
    return intersected_samples, sample_to_intersesct


def overall_word_level_metric_measure(task_label, pred_label,
                                      gold_sent_ls_dict,
                                      pred_sent_ls_topk_dict,
                                      topk, metric="exact_match", samples=None,
                                      src_detokenized=None, reference_word_dic=None,
                                      compute_intersection_score=True,
                                      mask_token=None, cls_token=None, sep_token=None,
                                      agg_func_ls=None):
    """
    'metric' based on a word level comparison of (pred,gold) : e.g : exact_match , edit
    'agg_func' based on a aggregation func to get the overall batch score : e.g : sum
    :param metric:
    :param agg_func:
    :return batch : score, number of token measured
    """
    if samples is None:
        samples = ["all"]
    if agg_func_ls is None:
        agg_func_ls = ["sum"]
    if task_label == "parsing_types":
        assert "parsing_heads" in gold_sent_ls_dict and "parsing_heads" in pred_sent_ls_topk_dict, \
            "ERROR : to compute the score of parsing_types : parsing_heads is required "
    assert isinstance(samples, list)
    assert len(set(samples) & set(AVAILABLE_EVALUATION_SAMPLE_FILTER)) > 0, \
        "ERROR : one of the samples in {} not supported {}".format(samples, AVAILABLE_EVALUATION_SAMPLE_FILTER)

    assert isinstance(agg_func_ls, list)

    pred_sent_ls_topk = pred_sent_ls_topk_dict[pred_label]
    gold_sent_ls = gold_sent_ls_dict[task_label]

    assert len(pred_sent_ls_topk) == topk, "ERROR topk not consistent with prediction list ".format(len(pred_sent_ls_topk), topk)
    overall_score_ls_sent = []
    intersected_samples = []

    if compute_intersection_score:
        intersected_samples, sample_to_intersesct = get_intersection_score(samples)

    overall_filter_ls = {sample: [] for sample in samples+intersected_samples}

    skipping_sent = 0

    for gold_ind_sent, gold_sent in enumerate(gold_sent_ls):
        try:
            assert len(gold_sent) == len(pred_sent_ls_topk[0][gold_ind_sent]), "ERROR : gold_sent : {} len {} and pred_sent_ls_topk[0][gold_ind_sent] {} len {} ".format(gold_sent, len(gold_sent), pred_sent_ls_topk[0][gold_ind_sent], len(pred_sent_ls_topk[0][gold_ind_sent]))
            # WARNING : this might not be true in POS mode for some cases (when mask bpe is used)
        except Exception as e:
            raise(e)
            print("ERROR (scoring/report) on task {} ".format(task_label), e)
            if len(gold_sent) > len(pred_sent_ls_topk[0][gold_ind_sent]):
                counter = 0
                n_to_solve = len(gold_sent) - len(pred_sent_ls_topk[0][gold_ind_sent])
                for ind in range(n_to_solve):
                    counter += gold_sent[-n_to_solve:][ind] == "_PAD_POS"
                if n_to_solve == counter:
                    gold_sent = gold_sent[:-n_to_solve]
                    src_detokenized[gold_ind_sent] = src_detokenized[gold_ind_sent][:-n_to_solve]
                    print("WARNING : we handled mismatch between pred len/src and gold len by cutting it based on "
                          "GOLD padding (SHOULD BE RAISED IN TASK POS)")
                    # NB : this should be handle properly :
                    #   the detokenization has a problem when dropout_bpe_mask is not null
                else:
                    if pred_sent_ls_topk[0][gold_ind_sent][-1] == "[SEP]":
                        pred_sent_ls_topk[0][gold_ind_sent] = pred_sent_ls_topk[0][gold_ind_sent]+["[SEP]" for _ in range(len(gold_sent) - len(pred_sent_ls_topk[0][gold_ind_sent]))]
                        print("APPENDING pred_sent_ls_topk[0] {} with {} ".format(len(gold_sent) - len(pred_sent_ls_topk[0][gold_ind_sent]), pred_sent_ls_topk[0][gold_ind_sent]))
                        assert len(gold_sent) == len(pred_sent_ls_topk[0][gold_ind_sent])
                    else:
                        print(Exception("ERROR {} : could not handled mismatch between pred {} len/src {} "
                                        "and gold len by cutting it based on GOLD padding (SHOULD BE RAISED IN TASK POS)".format(e, gold_sent, pred_sent_ls_topk[0][gold_ind_sent])))
                        skipping_sent += len(gold_sent_ls)
                        overall_score_ls_sent = [[0]]
                        break
            else:
                skipping_sent += len(gold_sent_ls)
                overall_score_ls_sent = [[0]]

                break
        if src_detokenized is not None and samples[0] != "all" and len(samples) > 1:
            # otherise we don't need src_detokenized
            assert len(gold_sent) == len(src_detokenized[gold_ind_sent]), "ERROR src_detokenized {} and gold_sent_ls for sent {} have different length ".format(gold_sent, src_detokenized[gold_ind_sent])

        score_sent = []
        filter_sent = {_sample: [] for _sample in samples}

        for ind_word in range(len(gold_sent)):
            gold_token = gold_sent[ind_word]
            topk_word_pred = [pred_sent_ls_topk[top][gold_ind_sent][ind_word] for top in range(topk)]
            # handling with LAS specificity ; a types is correct iif its label is correct and its head is correct
            if task_label == "types":
                gold_token_to_score = gold_token \
                    if gold_sent_ls_dict["heads"][gold_ind_sent][ind_word] == pred_sent_ls_topk_dict["parsing-heads"][0][gold_ind_sent][ind_word] else "ZERO-ING-SCORE-TYPES-AS-HEADS-IS-UNCORRECT"

            else:
                gold_token_to_score = gold_token
            score_sent.append(word_level_scoring(metric=metric, gold=gold_token_to_score, topk_pred=topk_word_pred, topk=topk))
            for ind_sample, _sample in enumerate(samples):
                try:

                    src = src_detokenized[gold_ind_sent][ind_word] if _sample != "all" and not _sample.startswith("n_masks") else None
                except Exception as e:
                    print("ERROR (scoring/report) handling src {} index ({},{}) ".format(src_detokenized,gold_ind_sent, ind_word), e)
                    pdb.set_trace()
                filter_sent[_sample].append(word_level_filter(sample=_sample,
                                                              gold=gold_token, topk_pred=topk_word_pred,
                                                              topk=topk, src=src, is_mwe=gold_sent_ls_dict[task_label][gold_ind_sent][ind_word],
                                                              word_reference_dic_ls=reference_word_dic,
                                                              mask_token=mask_token, cls_token=cls_token, sep_token=sep_token
                                                              ))
            if compute_intersection_score:
                for ind_sample, _sample in enumerate(sample_to_intersesct):
                    for ind_sample_2 in range(ind_sample):
                        inter = _sample+"-n-"+sample_to_intersesct[ind_sample_2]
                        if filter_sent.get(inter, None) is None:
                            filter_sent[inter] = []
                        filter_sent[inter].append(word_level_filter(sample=_sample,
                                                                    sample_2=sample_to_intersesct[ind_sample_2],
                                                                    gold=gold_token,
                                                                    topk_pred=topk_word_pred, topk=topk,
                                                                    src=src_detokenized[gold_ind_sent][ind_word],
                                                                    word_reference_dic_ls=reference_word_dic,
                                                                    is_mwe=gold_sent_ls_dict[task_label][gold_ind_sent][ind_word],
                                                                    cls_token=cls_token, sep_token=sep_token, mask_token=mask_token
                                                                    ))
        if compute_intersection_score:
            for _sample in samples+intersected_samples:
                overall_filter_ls[_sample].append(filter_sent[_sample])
        else:
            for _sample in samples:
                overall_filter_ls[_sample].append(filter_sent[_sample])
        overall_score_ls_sent.append(score_sent)

    result = {agg_func: {} for agg_func in agg_func_ls}

    for agg_func in agg_func_ls:
        for sample in samples+intersected_samples:
            try:

                result[agg_func][sample] = {
                    "score": agg_func_batch_score(overall_ls_sent_score=overall_score_ls_sent,
                                                  agg_func=agg_func, overall_filter=overall_filter_ls[sample]),
                    "agg_func": agg_func, "metric": "exact_match",
                    "n_tokens": agg_func_batch_score(overall_ls_sent_score=overall_score_ls_sent,
                                                     overall_filter=overall_filter_ls[sample],
                                                     agg_func="n_tokens"),
                    "n_sents": agg_func_batch_score(overall_ls_sent_score=overall_score_ls_sent,
                                                    overall_filter=overall_filter_ls[sample],
                                                    agg_func="n_sents")}

            except Exception as e:
                print(e)
                pdb.set_trace()

    return result, skipping_sent, samples+intersected_samples
