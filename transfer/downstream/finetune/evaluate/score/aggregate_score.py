from transfer.downstream.finetune.env.imports import pdb


def agg_func_batch_score(overall_ls_sent_score, overall_filter, agg_func):

    # sanity check
    try:
        assert len(overall_ls_sent_score) == len(overall_filter), "ERROR (scoring/agg_func_batch_score) : " \
                                                                  "filter uncorrect score:{} filter:{} one of " \
                                                                  "those had been miscomputed ".format(overall_ls_sent_score, overall_filter)
        for ind in range(len(overall_ls_sent_score)):
            assert len(overall_ls_sent_score[ind]) == len(overall_filter[ind]), "ERROR (scoring/agg_func_batch_score) : filter uncorrect " \
                                                                                "len sent score={} filter={} " \
                                                                                "(one of those has been miscomputed) ".format(overall_ls_sent_score, overall_filter[ind])
    except AssertionError as e:
        print(e)
        raise(e)

    # if filter 1 we keep otherise we ignore the token (and its score) in evaluation
    sum_ = sum([score for score_ls, filter_ls in zip(overall_ls_sent_score, overall_filter) for score, filter in zip(score_ls, filter_ls) if filter])
    n_tokens = sum([1 for score_ls, filter_ls in zip(overall_ls_sent_score, overall_filter) for _, filter in
                zip(score_ls, filter_ls) if filter])
    #n_tokens = sum([1 for score_ls in overall_ls_sent_score for _ in score_ls ])
    # at least one token not filter to keep the setnence
    n_sents = len([1 for sent, filter in zip(overall_ls_sent_score, overall_filter) if 1 in filter])

    if agg_func == "sum":
        return sum_
    elif agg_func == "n_tokens":
        return n_tokens
    elif agg_func == "n_sents":
        return n_sents
    elif agg_func == "mean":
        return sum_/n_tokens
    elif agg_func == "sum_mean_per_sent" and False:
        # TODO : filter
        sum_per_sent = [sum(score_ls) for score_ls in overall_ls_sent_score]
        token_per_sent = [len(score_ls) for score_ls in overall_ls_sent_score]
        sum_mean_per_sent_score = sum([sum_/token_len for sum_, token_len in zip(sum_per_sent, token_per_sent)])
        return sum_mean_per_sent_score
    else:
        raise(Exception("agg_func: {} not supported".format(agg_func)))

