from transfer.downstream.finetune.env.imports import pdb
from transfer.downstream.finetune.io_.logger import printing
from transfer.downstream.finetune.model.settings import LABEL_PARAMETER
from transfer.downstream.finetune.model.constants import NULL_STR, PADING_SYMBOLS


def align_bpe(n_bpe_target_minus_source, source_aligned, source_aligned_index, target_aligned, target_aligned_index,
              n_masks_to_add, src_token_len, bert_tokenizer, mask_token,
              mode="dummy", index_src=None, index_target=None, verbose=0):
    """
    align bpe of a given token using mode
    :return:
    """
    assert mode in ["dummy"]
    # dummy means appending with SPACE or MASK when needed
    if n_bpe_target_minus_source > 0:
        assert index_src is not None
        source_aligned_index.extend([index_src for _ in range(n_bpe_target_minus_source)])
        source_aligned.extend(
            bert_tokenizer.convert_tokens_to_ids([mask_token for _ in range(n_bpe_target_minus_source)]))

    elif n_bpe_target_minus_source < 0:
        assert index_target is not None
        # we add a NULL_STR (to be predicted) and index it as the former bpe token
        target_aligned_index.extend([index_target for _ in range(-n_bpe_target_minus_source)])
        target_aligned.extend(bert_tokenizer.convert_tokens_to_ids([NULL_STR for _ in range(-n_bpe_target_minus_source)]))

    n_masks_to_add.append(n_bpe_target_minus_source)
    n_masks_to_add.extend([-1 for _ in range(src_token_len - 1)])

    if verbose == "reader":
        printing("SRC appending word bpe align : {}\nTARGET appending word bpe align : {} \nN_MASKS------------ : {}",
                 var=[[mask_token for _ in range(n_bpe_target_minus_source)] if n_bpe_target_minus_source > 0 else "",
                      [NULL_STR for _ in range(-n_bpe_target_minus_source)] if n_bpe_target_minus_source < 0 else "",
                      [n_bpe_target_minus_source]+[-1 for _ in range(src_token_len - 1)]],
                 verbose_level="reader", verbose=verbose)

    return source_aligned, source_aligned_index, target_aligned, target_aligned_index, n_masks_to_add


def realigne_multi(ls_sent_str, input_alignement_with_raw, null_str, mask_str, label,
                   end_token,
                   remove_null_str=True, remove_mask_str=False, remove_extra_predicted_token=False,
                   keep_mask=False, gold_sent=False, flag_word_piece_token="##", flag_is_first_token=False,verbose=1):
    """
    # factorize with net realign
    ** remove_extra_predicted_token used iif pred mode **
    - detokenization of ls_sent_str based on input_alignement_with_raw index
    - we remove paddding and end detokenization at symbol [SEP] that we take as the end of sentence signal
    """

    assert len(ls_sent_str) == len(input_alignement_with_raw), \
        "ERROR : ls_sent_str {} : {} input_alignement_with_raw {} : {} ".format(ls_sent_str, len(ls_sent_str),
                                                                                input_alignement_with_raw, len(input_alignement_with_raw))
    new_sent_ls = []
    for sent, index_ls in zip(ls_sent_str, input_alignement_with_raw):

        assert len(sent) == len(index_ls), "ERROR : {} sent {} len {} and index_ls {} len {} not same len".format(label, sent, index_ls, len(sent), len(index_ls))

        former_index = -1
        new_sent = []
        former_token = ""
        for _i, (token, index) in enumerate(zip(sent, index_ls)):
            trigger_end_sent = False
            index = int(index)
            if remove_extra_predicted_token:
                if index == 1000 or index == -1:
                    # we reach the end according to gold data
                    # (this means we just stop looking at the prediciton of the model (we can do that because we assumed word alignement))
                    trigger_end_sent = True
                    if gold_sent:
                        # we sanity check that the alignement corredponds
                        try:
                            assert token in PADING_SYMBOLS, "WARNING 123 : breaking gold sequence on {} token not in {}".format(token , PADING_SYMBOLS)
                        except Exception as e:
                            if verbose>1:
                                print(e)
            if token == mask_str and not keep_mask:
                token = "X" if not remove_mask_str else ""
            # concatanating wordpieces
            if LABEL_PARAMETER[label]["realignement_mode"] == "detokenize_bpe":
                if index == former_index:
                    if token.startswith(flag_word_piece_token) and not flag_is_first_token:
                        former_token += token[len(flag_word_piece_token):]
                    else:
                        former_token += token
            # for sequence labelling : ignoring

            elif LABEL_PARAMETER[label]["realignement_mode"] == "ignore_non_first_bpe":
                # we just ignore bpe that are not first bpe of tokens
                if index == former_index:
                    pass
            if index != former_index or _i + 1 == len(index_ls):

                if not flag_is_first_token:
                    new_sent.append(former_token)
                elif flag_is_first_token and (isinstance(former_token, str) and former_token.startswith(flag_word_piece_token)):
                    new_sent.append(former_token[len(flag_word_piece_token):])
                else:
                    # is this last case possible
                    new_sent.append(former_token)
                former_token = token
                if trigger_end_sent:
                    break
            # if not pred mode : always not trigger_end_sent : True (required for the model to not stop too early if predict SEP too soon)
            # NEW CLEANER WAY OF BREAKING : should be generalize
            if remove_extra_predicted_token and trigger_end_sent:
                if not flag_is_first_token:
                    new_sent.append(former_token)
                elif flag_is_first_token and (isinstance(former_token, str) and former_token.startswith(flag_word_piece_token)):
                    new_sent.append(former_token[len(flag_word_piece_token):])
                else:
                    # is this last case possible
                    new_sent.append(former_token)
                break
            # TODO : SHOULD be cleaned
            # XLM (same first and end token) so not activated for </s>

            if ((former_token == end_token and end_token != "</s>") or _i + 1 == len(index_ls) and not remove_extra_predicted_token) or ((remove_extra_predicted_token and (former_token == end_token and trigger_end_sent) or _i + 1 == len(index_ls))):
                new_sent.append(token)
                break
            former_index = index

        new_sent_ls.append(new_sent[1:])
    #pdb.set_trace()
    return new_sent_ls

