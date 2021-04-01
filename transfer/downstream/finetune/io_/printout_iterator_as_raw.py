from transfer.downstream.finetune.io_.logger import printing


def outputing_raw_data_from_iterator(words, word_norm, chars, chars_norm, word_norm_not_norm, pos,
                                     verbose, print_raw, normalization, char_dictionary, word_dictionary,
                                     word_norm_dictionary,
                                     pos_dictionary):
    """
    printing real data on the fly for debugging, data sanity check, ...
    TODO : may factorize a few things here
    :param words:
    :param word_norm:
    :param chars:
    :param chars_norm:
    :param word_norm_not_norm:
    :param pos:
    :param verbose:
    :param print_raw:
    :param normalization:
    :param char_dictionary:
    :param word_dictionary:
    :param word_norm_dictionary:
    :param pos_dictionary:
    :return:
    """
    _verbose = verbose if isinstance(verbose, int) else 0
    if print_raw:
        _verbose = 5

    if _verbose >= 5:
        if word_norm_not_norm is not None:
            character_display = [
                " ".join([char_dictionary.get_instance(chars[sent, word_ind, char_i]) for char_i in range(chars.size(2))]) +
                " | NORM : {} |SENT {} WORD {}| ".format(word_norm_not_norm[sent, word_ind], sent, word_ind) for
                ind_sent, sent in enumerate(range(chars.size(0)))
                for ind_w, word_ind in enumerate(range(chars.size(1)))]
        else:
            character_display = [
                " ".join(
                    [char_dictionary.get_instance(chars[sent, word_ind, char_i]) for char_i in range(chars.size(2))])
                for ind_sent, sent in enumerate(range(chars.size(0)))
                for ind_w, word_ind in enumerate(range(chars.size(1)))]

        if word_norm is not None:
            assert word_norm_dictionary is not None
            word_norm_display = " ".join([word_norm_dictionary.get_instance(word_norm[sent, word_ind]) for word_ind in range(word_norm.size(1)) for sent in range(word_norm.size(0))])
        else:
            print("No word level normalized word (only char)")
            word_norm_display = ["NONE"]

        word_display = [word_dictionary.get_instance(words[batch, word_ind]) + " "
                        for batch in range(chars.size(0)) for word_ind in range(chars.size(1))]

        if pos_dictionary is not None:
            pos_display = [pos_dictionary.get_instance(pos[batch, 0]) + " " for batch in
                           range(chars.size(0))]
        else:
            pos_display = None

    else:
        word_display = []
        character_display = []
        pos_display = []
    if not normalization and chars is not None:
        chars_norm = chars.clone()

    # TODO add word_norm
    if _verbose >= 5:
        if word_norm_not_norm is not None:
            character_norm_display = [" ".join([char_dictionary.get_instance(chars_norm[sent, word_ind, char_i])
                                                for char_i in range(chars_norm.size(2))]) +
                                      "|  NORM : {} |SENT {} WORD {}| \n ".format(word_norm_not_norm[sent, word_ind], sent,
                                                                                  word_ind)
                                      for ind_sent, sent in enumerate(range(chars_norm.size(0)))
                                      for ind_w, word_ind in enumerate(range(chars_norm.size(1)))]
        else:
            character_norm_display = [" ".join([char_dictionary.get_instance(chars_norm[sent, word_ind, char_i])
                                                for char_i in range(chars_norm.size(2))])
                                      for ind_sent, sent in enumerate(range(chars_norm.size(0)))
                                      for ind_w, word_ind in enumerate(range(chars_norm.size(1)))]
        printing("Feeding source characters {} \n ------ Target characters {}  "
                 "(NB : the character vocabulary is the same at input and output)",
                 var=(character_display, character_norm_display),
                 verbose=_verbose, verbose_level=5)
        printing("Feeding source words {} ", var=[word_display], verbose=_verbose, verbose_level=5)
        printing("Feeding Word normalized (word level) {}", var=[word_norm_display], verbose=_verbose, verbose_level=5)
        printing("Feeding source pos {} ", var=[pos_display], verbose=_verbose, verbose_level=5)
        if chars is not None and chars_norm is not None:
            printing("TYPE {} char before batch chars_norm {} ", var=(chars.is_cuda, chars_norm.is_cuda), verbose=verbose, verbose_level=5)
