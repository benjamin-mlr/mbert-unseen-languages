#from env.importing import *
from env.importing import edit_distance, np
from io_.dat.constants import PAD_ID_CHAR
from env.project_variables import AVAILABLE_AGGREGATION_FUNC_AUX_TASKS


def get_transform_normalized_standart(cids_norm, cid_inputs, sent_index, word_index,
                                      task, pad_id_char=PAD_ID_CHAR):

    assert task in AVAILABLE_AGGREGATION_FUNC_AUX_TASKS, \
        "ERROR : get_transform_normalized_standart has been called but task {}  and {} ".format(task, AVAILABLE_AGGREGATION_FUNC_AUX_TASKS)

    norm = cids_norm[sent_index, word_index, :][cids_norm[sent_index, word_index, :] != pad_id_char]
    noisy = cid_inputs[sent_index, word_index, :][cid_inputs[sent_index, word_index, :] != pad_id_char]
    if "norm_not_norm" == task:
        # 1 means NORMED 0 means NEED_NORM
        return np.array_equal(norm, noisy)
    if "edit_prediction" == task:
        norm = "".join(norm.astype(str))
        noisy = "".join(noisy.astype(str))
        if edit_distance(noisy, norm)/max(len(noisy), len(norm)) > 3:
            print("EDIT distance ", edit_distance(noisy, norm)/max(len(noisy), len(norm)))
        return edit_distance(noisy, norm)/max(len(noisy), len(norm))
