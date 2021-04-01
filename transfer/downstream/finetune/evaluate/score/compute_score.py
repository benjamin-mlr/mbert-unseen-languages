from transfer.downstream.finetune.env.imports import pdb

from transfer.downstream.finetune.model.constants import SPECIAL_TOKEN_LS


def word_level_scoring(metric, gold, topk_pred, topk):
    """
    compare a gold string and a list of candidate
    return a score based on it
    only exact_match supported so far
    (originally designed for bert eval)
    :param metric:
    :param gold:
    :param topk_pred:
    :param topk:
    :return:
    """
    assert metric in ["exact_match"], "metric is {} ".format(metric)
    if topk > 1:
        assert metric == "exact_match", "ERROR : only exact_match allows for looking into topk prediction "
    assert len(topk_pred) == topk, "ERROR : inconsinstent provided topk and what I got "
    if metric == "exact_match":
        for pred in topk_pred:
            if gold == pred:
                return 1
        return 0


def word_level_filter(gold, topk_pred, topk, src, sample="all",
                      sample_2=None, word_reference_dic_ls=None, is_mwe=None, mask_token=None,
                      cls_token=None, sep_token=None):
    """
    compare a gold string and a list of candidate
    return a score based on it
    only exact_match supported so far
    (originally designed for bert eval)
    :param metric:
    :param gold:
    :param topk_pred:
    :param topk:
    :return:
    """
    #assert sample in ["all", "NORMED", "NEED_NORM"]
    assert len(topk_pred) == topk, "ERROR : inconsinstent provided topk and what I got "

    if gold in SPECIAL_TOKEN_LS or gold in [sep_token, cls_token]:
        return 0

    if sample == "all":
        sample_1_filter = 1
    elif sample == "mlm":
        sample_1_filter = mask_token in src
    elif sample == "MWE":
        assert is_mwe is not None, "ERROR filter request is MWE but is_mwe not provided"
        sample_1_filter = is_mwe
    elif sample == "PRED_NORMED":
        sample_1_filter = src == topk_pred[0]
    elif sample == "PRED_NEED_NORM":
        sample_1_filter = src != topk_pred[0]
    elif sample == "NORMED":
        sample_1_filter = src == gold
    elif sample == "NEED_NORM":
        sample_1_filter = src != gold
    elif sample == "InV":
        assert word_reference_dic_ls is not None, "No word_reference_dic_ls provided"
        assert word_reference_dic_ls.get("InV", None) is not None, "No word_reference_dic_ls['InV'] provided"
        sample_1_filter = src in word_reference_dic_ls["InV"] or src.lower() in word_reference_dic_ls["InV"]
    elif sample == "OOV":
        assert word_reference_dic_ls is not None, "No word_reference_dic_ls provided"
        assert word_reference_dic_ls.get("InV", None) is not None, "No word_reference_dic_ls['InV'] provided"
        sample_1_filter = src not in word_reference_dic_ls["InV"] and src.lower() not in word_reference_dic_ls["InV"]
    elif sample.startswith("n_masks"):
        assert isinstance(eval(sample[-1]), int), "ERROR : sample {} do not fit in n_masks_N ".format(sample)
        sample_1_filter = gold == eval(sample[-1])

    if sample_2 is not None:
        assert sample_2 != sample, "we don't want reduncancies"
        assert sample != "all", " we don't want intersction with all "
        if sample_2 == "all":
            sample_2_filter = 1
        elif sample_2 == "NORMED":
            sample_2_filter = src == gold
        elif sample_2 == "NEED_NORM":
            sample_2_filter = src != gold
        elif sample_2 == "InV":
            assert word_reference_dic_ls is not None, "No word_reference_dic_ls provided"
            assert word_reference_dic_ls.get("InV", None) is not None, "No word_reference_dic_ls['InV'] provided"
            sample_2_filter = src in word_reference_dic_ls["InV"] or src.lower() in word_reference_dic_ls["InV"]
        elif sample_2 == "OOV":
            assert word_reference_dic_ls is not None, "No word_reference_dic_ls provided"
            assert word_reference_dic_ls.get("InV", None) is not None, "No word_reference_dic_ls['InV'] provided"
            sample_2_filter = src not in word_reference_dic_ls["InV"] and src.lower() not in word_reference_dic_ls["InV"]
        elif sample_2 == "PRED_NORMED":
            sample_2_filter = src == topk_pred[0]
        elif sample_2 == "PRED_NEED_NORM":
            sample_2_filter = src != topk_pred[0]
        elif sample_2 == "MWE":
            assert is_mwe is not None, "ERROR filter request is MWE but is_mwe not provided"
            sample_2_filter = is_mwe
    else:
        sample_2_filter = 1

    return sample_1_filter * sample_2_filter
