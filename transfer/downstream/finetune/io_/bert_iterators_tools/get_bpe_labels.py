
from transfer.downstream.finetune.env.imports import pdb, OrderedDict, np, torch, time
from transfer.downstream.finetune.model.settings import LABEL_PARAMETER

from transfer.downstream.finetune.model.constants import PAD_ID_LOSS_STANDART, ROOT_HEADS_INDEX, END_HEADS_INDEX
from transfer.downstream.finetune.model.optimization.masking import dropout_mlm


def get_mask_input(input_tokens_tensor, use_gpu, pad):
    new_input = np.array(input_tokens_tensor.cpu())
    _input_mask = [[0 if new_input[ind_sent][ind_tok] == pad else 1 for ind_tok in range(len(new_input[ind_sent]))] for ind_sent in range(len(new_input))]
    input_mask = torch.Tensor(_input_mask).long()
    if use_gpu:
        input_mask = input_mask.cuda()

    return input_mask


# as CLS is appended at the begining of each sentences : we need to adjust the labels for it
CLS_ADJUST = 0


def get_bpe_label_word_level_task(labels, batch, input_tokens_tensor,
                                  input_alignement_with_raw,
                                  use_gpu, label_name, pad, graph_labels=False):

    if labels is not None:
        output_tokens_tensor = np.array(labels.cpu())
    else:
        output_tokens_tensor = None
    new_input = np.array(input_tokens_tensor.cpu())
    len_max = max([len(sent) for sent in new_input])
    new_input = [[inp for inp in sent] + [pad for _ in range(len_max - len(sent))] for sent in new_input]
    # we mask bpe token that have been split (we don't mask the first bpe token of each word)
    _input_mask = [[0 if new_input[ind_sent][ind_tok] == pad or input_alignement_with_raw[ind_sent][ind_tok-1] == input_alignement_with_raw[ind_sent][ind_tok] else 1 for ind_tok in range(len(new_input[ind_sent]))] for ind_sent in range(len(new_input))]
    cumulate_shift = None
    if graph_labels:
        # for each sentence : each bpe token : we count the number of multi-bpe token before it
        def get_cumulated_non_first_bpe_counter(sent):
            counter = 0
            new_sent = []
            counter_former = 0
            cumulated = 0
            for ind, token in enumerate(sent):
                if ind+1 < len(sent) and token == sent[ind+1] and token != 1000:
                    counter += 1
                elif token != 1000:
                    new_sent.append(counter_former+cumulated)
                    cumulated += counter_former
                    counter_former = counter
                    counter = 0
            return new_sent

        #def test_get_cumulated_non_first_bpe_counter():
        #    assert [0, 0, 0, 1, 1, 1, 3, 3, 3, 5, 5, 5] == get_cumulated_non_first_bpe_counter([0, 1, 2, 2 ,3, 4, 5, 5, 5, 6, 7, 8, 8, 8, 9, 10 ,11, 1000])
        #    assert [0, 0, 0, 1, 1, 1, 3, 3, 3, 5, 5, 5] == get_cumulated_non_first_bpe_counter([0, 1, 2, 2, 3, 4, 5, 5, 5, 6, 7, 8, 8, 8, 9, 10, 11])
        #    #print("TEST passed ")
        #test_get_cumulated_non_first_bpe_counter()

        cumulate_shift = [get_cumulated_non_first_bpe_counter(input_alignement_with_raw[ind_sent]) for ind_sent in range(len(input_alignement_with_raw))]

    output_tokens_tensor_new = []
    for ind_sent in range(len(_input_mask)):
        output_tokens_tensor_new_ls = []
        shift = 0
        for ind_tok in range(len(_input_mask[ind_sent])):
            mask = _input_mask[ind_sent][ind_tok]

            if labels is not None:
                try:
                    label = output_tokens_tensor[ind_sent, ind_tok - shift]
                    if graph_labels:
                        # as CLS is appended at the begining of each sentences : we need to adjust the labels for it
                        # CLS and SEQ points to the first token indexed by -1 so become 1
                        if label not in [ROOT_HEADS_INDEX, END_HEADS_INDEX] and cumulate_shift[ind_sent][label] > 0:
                            label += cumulate_shift[ind_sent][label]
                        label += CLS_ADJUST
                except Exception as e:
                    try:
                        assert input_alignement_with_raw[ind_sent][ind_tok] == 1000, "ERROR we should have reached the end of get labels also "
                        label = LABEL_PARAMETER[label_name]["pad_value"] #PAD_ID_TAG if not graph_labels else PAD_ID_HEADS # output_tokens_tensor[ind_sent, output_tokens_tensor.shape[1] - 1]
                    except Exception as f:
                        print("ERROR (get_bpe_labels): we reached the end of output labels but input is not done ! ", f)
                        print("ERROR ind_send:{} ind_tok {} shift {} output_tokens_tensor {} alignement {} -  {}".format(ind_sent, ind_tok, shift, output_tokens_tensor, input_alignement_with_raw[ind_sent], e))
                        print("ERROR ind_send ", batch.raw_input, batch.raw_output)
                        raise(e)

            if mask == 0 and labels is not None:
                # 1 for _PAD_POS fpr PAD_ID_HEADS 0
                pad = LABEL_PARAMETER[label_name]["pad_value"]#PAD_ID_TAG if not graph_labels else PAD_ID_HEADS
                output_tokens_tensor_new_ls.append(pad)
                shift += 1
            elif labels is not None:
                output_tokens_tensor_new_ls.append(label)
        output_tokens_tensor_new.append(output_tokens_tensor_new_ls)

    def sanity_test_parsing_label(labels, output_tokens_tensor_new, input_alignement_with_raw, cumulate_shift):
        for sent in range(labels.size(0)):
            ind_max = len(cumulate_shift[sent])-1
            for _ in range(5):
                ind = np.random.choice(range(ind_max))
                # the new label must be equal to the old one at the corresponding position + 1 + the number of non-first-bpe-token (original indexing of the label)
                if output_tokens_tensor_new[sent][ind] not in [ROOT_HEADS_INDEX+1, END_HEADS_INDEX, PAD_ID_LOSS_STANDART]:
                    try:
                        assert output_tokens_tensor_new[sent][ind] == labels[sent, int(input_alignement_with_raw[sent][ind])]+CLS_ADJUST+cumulate_shift[sent][labels[sent, int(input_alignement_with_raw[sent][ind])]], \
                        "ERROR sent {} ind word {} " \
                        "new {} and old {} cumulted {} ".format(sent, ind, output_tokens_tensor_new[sent][ind],
                                                            labels[sent, input_alignement_with_raw[sent][ind]], cumulate_shift[sent][ind])
                    except AssertionError as e:
                        print(e)
                        pdb.set_trace()
                    #print("TEST passed for sent {} word {}".format(sent, ind))

    if graph_labels and labels is not None:
        sanity_test_parsing_label(labels, output_tokens_tensor_new, input_alignement_with_raw, cumulate_shift)
    if labels is not None:
        output_tokens_tensor = torch.Tensor(output_tokens_tensor_new).long()
        head_mask = torch.Tensor(_input_mask).long()
    input_tokens_tensor = torch.Tensor(new_input).long()
    if use_gpu:
        if labels is not None:
            head_mask = head_mask.cuda()
            output_tokens_tensor = output_tokens_tensor.cuda()
        input_tokens_tensor = input_tokens_tensor.cuda()
    return output_tokens_tensor, head_mask, input_tokens_tensor, cumulate_shift


def get_label_per_bpe(tasks, batch, input_tokens_tensor, input_alignement_with_raw, use_gpu, tasks_parameters,
                      pad_index, vocab_len=None, masking_strategy=0,
                      mask_token_index=None, sep_token_index=None, cls_token_index=None, dropout_input_bpe=None):
    """
    returns input, input masks and output for each tasks
    (in regard to the task type , so far only word level is supported)
    """
    #  TODO : should be done in pytorch + reducancies with get_index
    label_per_task = OrderedDict()
    input_tokens_tensor_per_task = OrderedDict()
    token_type_ids = OrderedDict()
    input_mask_per_task = OrderedDict()
    input_mask, output_tokens_tensor = None, None
    cumulate_shift = None
    head_masks = OrderedDict()
    for simul_task in tasks:
        for task in simul_task:
            for task_batch_name in tasks_parameters[task]["label"]:
                task_batch = eval("batch.{}".format(task_batch_name)).clone()
                # why not is_mwe and n_masks also
                if task in ["parsing", "pos"]:
                    # for now we align parsing and tagging signal with raw input using
                    # get_bpe_label_word_level_task here
                    output_tokens_tensor, head_mask, input_tokens_tensor, _cumulate_shift = get_bpe_label_word_level_task(labels=task_batch,
                                                                                                         pad=pad_index,
                                                                                                         batch=batch,
                                                                                                         #input_tokens_tensor,
                                                                                                         #input_alignement_with_raw,
                                                                                                         use_gpu=use_gpu,
                                                                                                         label_name=task_batch_name,
                                                                                                         input_tokens_tensor=eval("batch.{}".format(tasks_parameters[task]["input"])).clone(),
                                                                                                         input_alignement_with_raw=eval("batch.{}_alignement".format(tasks_parameters[task]["input"])),
                                                                                                         graph_labels=LABEL_PARAMETER[task_batch_name].get("graph_label", False))

                    output_tokens_tensor_aligned = output_tokens_tensor#[:, : input_tokens_tensor.size(1)]
                    if task == "parsing" and task_batch_name == "heads":
                        cumulate_shift = _cumulate_shift
                else:
                    # for tokenization related tasks we already took care of alignement during CoNLLReader
                    output_tokens_tensor_aligned = task_batch
                    head_mask = None

                head_masks[task] = head_mask
                if output_tokens_tensor_aligned is not None:
                    output_tokens_tensor_aligned = output_tokens_tensor_aligned.contiguous()

                    if use_gpu:
                        output_tokens_tensor_aligned = output_tokens_tensor_aligned.cuda()
                # if the task has several label : we just appen the label name to the task in the label dictionary
                # ALL output padded with BERT pad are padded with LOSS pad (-1)
                label_per_task[task_batch_name] = output_tokens_tensor_aligned

            if not tasks_parameters[task].get("mask_input", False):
                #input_tokens_tensor_per_task[tasks_parameters[task]["input"]] = eval("batch.{}".format(tasks_parameters[task]["input"])).clone() if task not in ["parsing", "pos"] else input_tokens_tensor.clone()
                input_tokens_tensor_per_task[tasks_parameters[task]["input"]] = eval("batch.{}".format(tasks_parameters[task]["input"])).clone()

                # we dropout input for regulirization purpose here if needed
                if dropout_input_bpe is not None and dropout_input_bpe > 0:
                    input_tokens_tensor_per_task[tasks_parameters[task]["input"]] = dropout_mlm(
                        input_tokens_tensor_per_task[tasks_parameters[task]["input"]],
                        mask_token_index=mask_token_index,
                        sep_token_index=sep_token_index,
                        cls_token_index=cls_token_index,
                        pad_index=pad_index,
                        use_gpu=False,
                        dropout_mask=dropout_input_bpe,
                        dropout_random_bpe_of_masked=0.5, vocab_len=vocab_len)

                input_mask_per_task[tasks_parameters[task]["input"]] = (input_tokens_tensor_per_task[tasks_parameters[task]["input"]] != pad_index)
            else:# mlm
                # mask_input is for Mask Languag Model task  : which means Masking + replacing by random wordpiece
                assert masking_strategy is None
                #NB : dropout_input_bpe is ignored in MLM : set to 15% as Bert Paper
                assert tasks_parameters[task].get("original") is not None, \
                    "ERROR 'original' field is needed to get raw sequence before preprocssing for task {} ".format(task)
                input_tokens_tensor_per_task[tasks_parameters[task]["input"]] = dropout_mlm(eval("batch.{}".format(tasks_parameters[task]["original"])).clone(),
                                                                                            mask_token_index=mask_token_index,
                                                                                            sep_token_index=sep_token_index,
                                                                                            cls_token_index=cls_token_index,
                                                                                            pad_index=pad_index,
                                                                                            use_gpu=False,
                                                                                            dropout_mask=0.15,
                                                                                            dropout_random_bpe_of_masked=0.5,
                                                                                            vocab_len=vocab_len)
                # NB ; this mask is for PADDING !! (bad naming)
                input_mask_per_task[tasks_parameters[task]["input"]] = (input_tokens_tensor_per_task[tasks_parameters[task]["input"]] != pad_index)

            token_type_ids[tasks_parameters[task]["input"]] = torch.zeros_like(input_tokens_tensor_per_task[tasks_parameters[task]["input"]])
            if use_gpu:
                input_tokens_tensor_per_task[tasks_parameters[task]["input"]] = input_tokens_tensor_per_task[tasks_parameters[task]["input"]].cuda()
                input_mask_per_task[tasks_parameters[task]["input"]] = input_mask_per_task[tasks_parameters[task]["input"]].cuda()
                token_type_ids[tasks_parameters[task]["input"]] = token_type_ids[tasks_parameters[task]["input"]].cuda()

    return head_masks, input_tokens_tensor, token_type_ids, label_per_task, \
           input_tokens_tensor_per_task, input_mask_per_task, cumulate_shift
