from transfer.downstream.finetune.env.imports import pdb, torch, np
from transfer.downstream.finetune.io_.logger import printing
from transfer.downstream.finetune.trainer.tools.sanity_check import sanity_check_data_len


def rp_space_func(sent, replace_space_with=""):
    sent = [word if " " not in word else word.replace(" ", replace_space_with) for word in sent]
    return sent



def get_indexes(list_pretokenized_str, tokenizer, verbose, use_gpu,
                word_norm_not_norm=None):
    """
    from pretokenized string : it will bpe-tokenize it using BERT 'tokenizer'
    and then convert it to tokens ids
    :param list_pretokenized_str:
    :param tokenizer:
    :param verbose:
    :param use_gpu:
    :return:
    """
    all_tokenized_ls = [tokenizer.tokenize_origin(inp,) for inp in list_pretokenized_str]
    tokenized_ls = [tup[0] for tup in all_tokenized_ls]

    aligned_index = [tup[1] for tup in all_tokenized_ls]
    segments_ids = [[0 for _ in range(len(tokenized))] for tokenized in tokenized_ls]

    printing("DATA : bpe tokenized {} , {} {} ", var=[tokenized_ls, len(tokenized_ls),len(tokenized_ls[0])], verbose=verbose, verbose_level="raw_data")
    printing("DATA : bpe tokenized {} , {} {} ", var=[tokenized_ls, len(tokenized_ls),len(tokenized_ls[0])], verbose=verbose, verbose_level="alignement")
    ids_ls = [tokenizer.convert_tokens_to_ids(inp) for inp in tokenized_ls]
    max_sent_len = max([len(inp) for inp in tokenized_ls])
    ids_padded = [inp + [PAD_ID_BERT for _ in range(max_sent_len - len(inp))] for inp in ids_ls]
    aligned_index_padded = [[e for e in inp] + [1000 for _ in range(max_sent_len - len(inp))] for inp in aligned_index]
    segments_padded = [inp + [PAD_ID_BERT for _ in range(max_sent_len - len(inp))] for inp in segments_ids]

    if word_norm_not_norm is not None:
        mask = mask_group(word_norm_not_norm, bpe_aligned_index=aligned_index_padded)
    else:
        mask = [[1 for _ in inp]+[0 for _ in range(max_sent_len - len(inp))] for inp in segments_ids]
    mask = torch.LongTensor(mask)
    tokens_tensor = torch.LongTensor(ids_padded)
    segments_tensors = torch.LongTensor(segments_padded)
    if use_gpu:
        mask = mask.cuda()
        tokens_tensor = tokens_tensor.cuda()
        segments_tensors = segments_tensors.cuda()

    printing("DATA {}", var=[tokens_tensor], verbose=verbose, verbose_level=3)

    sanity_check_data_len(tokens_tensor, segments_tensors, tokenized_ls, aligned_index, raising_error=True)

    return tokens_tensor, segments_tensors, tokenized_ls, aligned_index_padded, mask


def preprocess_batch_string_for_bert(batch,start_token, end_token, rp_space=False):
    """
    adding starting and ending token in raw sentences
    :param batch:
    :return:
    """
    for i in range(len(batch)):
        try:
            batch[i][0] = start_token
        except:
            pdb.set_trace()
        batch[i][-1] = end_token
        if rp_space:
            batch[i] = rp_space_func(batch[i])
        batch[i] = " ".join(batch[i])
    return batch


def from_bpe_token_to_str(bpe_tensor,  topk, pred_mode, null_token_index, null_str, task, tokenizer=None,
                          bpe_tensor_src=None,
                          pos_dictionary=None, label="normalize",
                          label_dictionary=None, mask_index=None,
                          get_bpe_string=False, verbose=1):
    """
    it actually supports not only bpe token but also pos-token
    pred_mode allow to handle gold data also (which only have 2 dim and not three)
    :param bpe_tensor:
    :param topk: int : number of top prediction : will arrange them with all the top1 all the 2nd all the third...
    :param pred_mode: book
    :return:
    """
    assert label is not None or get_bpe_string, \
        "ERROR : task {} get_string {} : one of them should be defined or True".format(label, get_bpe_string)
    if task == "mlm" and pred_mode:
        assert bpe_tensor_src is not None and mask_index is not None, "ERROR bpe_tensor_src is needed to get not-predicted token as well as mask_index "
        predictions_topk_ls = [[[bpe_tensor[sent, word, top].item() if bpe_tensor_src[sent, word].item() == mask_index else bpe_tensor_src[sent, word].item() for word in range(bpe_tensor.size(1))] for sent in range(bpe_tensor.size(0))] for top in range(topk)]
    else:
        predictions_topk_ls = [[[bpe_tensor[sent, word, top].item() if pred_mode else bpe_tensor[sent, word].item() for word in range(bpe_tensor.size(1))] for sent in range(bpe_tensor.size(0))] for top in range(topk)]

    # here all labels that require the tokenizer (should factorize it in some way)
    if get_bpe_string:#label in ["normalize", "mwe_prediction", "input_masked"] or
        assert tokenizer is not None
        # requires task specific here : mlm only prediction we are interested in are
        # RM , special_extra_token=null_token_index, special_token_string=null_str
        sent_ls_top = [[tokenizer.convert_ids_to_tokens(sent_bpe) for sent_bpe in predictions_topk] for predictions_topk in predictions_topk_ls]

        printing("DATA : bpe string again {}", var=[sent_ls_top], verbose=verbose, verbose_level="raw_data")
    else:
        dictionary = label_dictionary

        if label_dictionary == "index":
            sent_ls_top = [[[token_ind for token_ind in sent_bpe] for sent_bpe in predictions_topk] for predictions_topk in predictions_topk_ls]
        else:
            try:
                sent_ls_top = [[[dictionary.instances[token_ind - 1] if token_ind > 0 else "UNK" for token_ind in sent_bpe] for sent_bpe in predictions_topk] for predictions_topk in predictions_topk_ls]
            # adding more information about the exe
            except Exception as e:
                print("{} : dictionary : {} and prediction {} (POS specificity was removed )".format(e, dictionary.instances, predictions_topk_ls))
                raise(e)

    if not pred_mode:
        sent_ls_top = sent_ls_top[0]

    return sent_ls_top


def input_normalization_processing(task_normalize_is, batch, norm_2_noise_training, norm_2_noise_eval):
    norm2noise_bool = False
    if (norm_2_noise_training is not None or norm_2_noise_eval) and task_normalize_is:
        portion_norm2noise = norm_2_noise_training if norm_2_noise_training is not None else 1.
        norm_2_noise_training = portion_norm2noise is not None
        rand = np.random.uniform(low=0, high=1, size=1)[0]
        norm2noise_bool = portion_norm2noise >= rand
        if norm2noise_bool:
            batch_raw_input = preprocess_batch_string_for_bert(batch.raw_output)
            printing("WARNING : input is gold norm", verbose_level=2, verbose=1)
        else:
            printing("WARNING : input is input", verbose_level=2, verbose=1)
            batch_raw_input = preprocess_batch_string_for_bert(batch.raw_input)
    else:
        printing("WARNING : input is input ", verbose_level=2, verbose=1)
        batch_raw_input = preprocess_batch_string_for_bert(batch.raw_input)
    return batch_raw_input,  norm2noise_bool, norm_2_noise_training

