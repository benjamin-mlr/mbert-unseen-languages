from transfer.downstream.finetune.env.imports import time
from transfer.downstream.finetune.io_.logger import printing
from transfer.downstream.finetune.model.settings import MULTI_TASK_LOSS_PONDERATION_PREDEFINED_MODE, TASKS_PARAMETER


def sanity_check_info_checkpoint(info_checkpoint, template):
    for key in template.keys():
        # git id is added on the fly as updated
        if key not in ["git_id", "other"]:
            assert key in info_checkpoint, "ERROR {} key is not in info_checkpoint".format(key)


def get_timing(former):
    if former is not None:
        return time.time() - former, time.time()
    else:
        return None, None


def sanity_check_loss_poneration(ponderation_dic, verbose=1):
    if isinstance(ponderation_dic, dict):
        for task in TASKS_PARAMETER:
            assert task in ponderation_dic, "ERROR : task {} is not related to a ponderation while it should ".format(task)
    elif isinstance(ponderation_dic,str):
        assert ponderation_dic in MULTI_TASK_LOSS_PONDERATION_PREDEFINED_MODE, "ERROR ponderation should be in {}".format(ponderation_dic,MULTI_TASK_LOSS_PONDERATION_PREDEFINED_MODE)
        printing("WARNING : COULD NOT SANITY CHECK ponderation_dic {} ", var=[ponderation_dic], verbose=verbose,
                 verbose_level=1)
    else:
        raise(Exception("ponderation_dic is neither string or dict {}".format(ponderation_dic)))





def sanity_check_checkpointing_metric(tasks, checkpointing_metric):
    standard_metric = "loss-dev-all"
    if len(tasks) > 1:
        assert checkpointing_metric == standard_metric, "ERROR : only {} supported in multitask setting so far".format(
            standard_metric)
    else:
        allowed_metric = [standard_metric, TASKS_PARAMETER[tasks[0]].get("default_metric", "NOT a metric")]
        assert checkpointing_metric in allowed_metric, "ERROR checkpointing_metric {} should be in {}".format(
            checkpointing_metric, allowed_metric)


def sanity_check_data_len(tokens_tensor, segments_tensors, tokenized_ls, aligned_index, raising_error=True):
    """
    BERT related checking that each batch of tokens, segments (sentence 1 or B), raw token and index (for realignement)
    have consistent length
    """
    n_sentence = len(tokens_tensor)
    try:
        assert len(segments_tensors) == n_sentence, "ERROR BATCH segments_tensors {} not same len as tokens ids {}".format(segments_tensors, n_sentence)
        assert len(tokenized_ls) == n_sentence, "ERROR BATCH  tokenized_ls {} not same len as tokens ids {}".format(tokenized_ls, n_sentence)
        assert len(aligned_index) == n_sentence, "ERROR BATCH aligned_index {} not same len as tokens ids {}".format(aligned_index, n_sentence)
    except AssertionError as e:
        if raising_error:
            raise(e)
        else:
            print(e)
    for index, segment, token_str, index in zip(tokens_tensor, segments_tensors, tokenized_ls, aligned_index):
        n_token = len(index)
        try:
            #assert len(segment) == n_token, "ERROR sentence {} segment not same len as index {}".format(segment, index)
            assert len(token_str) == n_token, "ERROR sentence {} token_str not same len as index {}".format(token_str, index)
            assert len(index) == n_token, "ERROR sentence {} index not same len as index {}".format(index, index)
        except AssertionError as e:
            if raising_error:
                raise(e)
            else:
                print(e)
