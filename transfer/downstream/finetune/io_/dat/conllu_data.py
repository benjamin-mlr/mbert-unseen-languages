from transfer.downstream.finetune.env.imports import os, codecs, torch, np, Variable
from transfer.downstream.finetune.io_.logger import printing
from transfer.downstream.finetune.io_.dat.constants import MAX_CHAR_LENGTH, NUM_CHAR_PAD, PAD_CHAR, PAD_POS, PAD_TYPE, ROOT_CHAR, ROOT_POS, PAD, \
  ROOT_TYPE, END_CHAR, END_POS, END_TYPE, _START_VOCAB, ROOT, PAD_ID_WORD, PAD_ID_CHAR, PAD_ID_TAG, DIGIT_RE, CHAR_START_ID, CHAR_START, CHAR_END_ID, PAD_ID_CHAR, PAD_ID_NORM_NOT_NORM, END,\
  MEAN_RAND_W2V, SCALE_RAND_W2V, PAD_ID_EDIT, PAD_ID_HEADS
from transfer.downstream.finetune.model.constants import PAD_ID_LOSS_STANDART
from transfer.downstream.finetune.io_.dat.conllu_reader import CoNLLReader
from transfer.downstream.finetune.io_.dat.dictionary import Dictionary
from transfer.downstream.finetune.io_.dat.conllu_get_normalization import get_normalized_token

from transfer.downstream.finetune.model.settings import  AVAILABLE_INPUTS
from transfer.downstream.finetune.env.vars import MAX_VOCABULARY_SIZE_WORD_DIC

#from io_.dat.conllu_get_normalization import get_normalized_token
#from io_.signal_aggregation import get_transform_normalized_standart


def fill_array(array, filling, row, pad):
  if array is not None:
    array[row, :len(filling)] = filling
    array[row, len(filling):] = pad
    return array
  return None


def fill_index_list(list_index, filling, pad, bucket_length_in_bpe):
  if list_index is not None and filling is not None:
    list_index.append(filling + [pad for _ in range(bucket_length_in_bpe - len(filling))])
  return list_index


def query_batch(index, batch):
  if batch is not None:
    return batch[index]
  return None


def load_dict(dict_path, train_path=None, dev_path=None, test_path=None,
              word_normalization=False, pos_specific_data_set=None,
              word_embed_dict=None, tasks=None,
              dry_run=0, expand_vocab=False, add_start_char=None,
              case=None, do_not_fill_dictionaries=False,
              force_new_dic=False, verbose=1):
  """
  Based on data, it create dictionaries
  a dictionary is a string to index mapping
  It's useful for Words, characters, POS tags (U-POS and X-POS) , Dependancies relations, normalized words (for lexical normalization)
  If the task does't require a given dictionary : it will be created but left empty
  :return:
  """

  create_new_dictionaries = False
  for dict_type in ["word", "character", "pos", "xpos", "type"]:
    if not os.path.isfile(os.path.join(dict_path, "{}.json".format(dict_type))):
      create_new_dictionaries  = True
      printing("WARNING : did not find dictionaries {} so creating it ", var=[dict_path], verbose=verbose, verbose_level=2)

  create_new_dictionaries = True if force_new_dic else create_new_dictionaries

  if create_new_dictionaries :
    assert isinstance(train_path, list) and (isinstance(dev_path, list) or dev_path is None), \
      "ERROR : TRAIN:{} not list or DEV:{} not list ".format(train_path, dev_path)
    assert train_path is not None and dev_path is not None and add_start_char is not None

    printing("Creating dictionary in {} ".format(dict_path), verbose=verbose, verbose_level=1)
    word_dictionary, word_norm_dictionary, char_dictionary, pos_dictionary, \
    xpos_dictionary, type_dictionary = create_dict(dict_path,
                                                   train_path=train_path,
                                                   dev_path=dev_path,
                                                   test_path= test_path,  dry_run=dry_run,
                                                   word_embed_dict=word_embed_dict,
                                                   expand_vocab_bool=expand_vocab, add_start_char=add_start_char,
                                                   pos_specific_data_set=pos_specific_data_set,
                                                   tasks=tasks,case=case, do_not_fill_dictionaries=do_not_fill_dictionaries,
                                                   word_normalization=word_normalization, verbose=verbose)
  else:
    assert train_path is None and dev_path is None and test_path is None and add_start_char is None, \
      "train_path {} dev_path {} test_path {} add_start_char {}".format(train_path,
                                                                        dev_path,
                                                                        test_path,
                                                                        add_start_char)
    printing("Loading dictionary from {} ".format(dict_path), verbose=verbose, verbose_level=1)
    word_dictionary = Dictionary('word', default_value=True, singleton=True)
    word_norm_dictionary = Dictionary('word_norm', default_value=True, singleton=True) if word_normalization else None
    char_dictionary = Dictionary('character', default_value=True)
    pos_dictionary = Dictionary('pos', default_value=True)
    xpos_dictionary = Dictionary('xpos', default_value=True)
    type_dictionary = Dictionary('type', default_value=True)
    dic_to_load_names = ["word", "character", "pos", "xpos", "type"]
    dict_to_load = [word_dictionary, char_dictionary, pos_dictionary, xpos_dictionary, type_dictionary]
    if word_normalization:
      dic_to_load_names.append("word_norm")
      dict_to_load.append(word_norm_dictionary)
    for name, dic in zip(dic_to_load_names, dict_to_load):
      dic.load(input_directory=dict_path, name=name)

  return word_dictionary, word_norm_dictionary, char_dictionary, pos_dictionary, xpos_dictionary, type_dictionary


def pos_specific_dic_builder(pos_specific_data_set, pos_dictionary, verbose=1):
  if pos_specific_data_set is not None:
    assert os.path.exists(pos_specific_data_set), "{} does not exist".format(pos_specific_data_set)
    with codecs.open(pos_specific_data_set, 'r', 'utf-8', errors='ignore') as file:
      li = 0
      for line in file:
        line = line.strip()
        if len(line) == 0 or line[0] == '#':
          continue
        tokens = line.split('\t')
        if '-' in tokens[0] or '.' in tokens[0]:
          continue
        pos = tokens[3]  # if tokens[4]=='_' else tokens[3]+'$$$'+tokens[4]
        #xpos = tokens[4]
        pos_dictionary.add(pos)
        #xpos_dictionary.add(xpos)
    printing("VOCABULARY : POS Vocabulary : pos dictionary built on {} ".format(pos_specific_data_set), verbose_level=1, verbose=verbose)
    return pos_dictionary
  printing("VOCABULARY : POS Vocabulary : pos dictionary untouched", verbose_level=1, verbose=verbose)
  return pos_dictionary


def create_dict(dict_path, train_path, dev_path, test_path, tasks,
                dry_run, word_normalization=False, expand_vocab_bool=False, add_start_char=0,
                min_occurence=0, pos_specific_data_set=None,word_embed_dict=None, case=None,
                verbose=1, do_not_fill_dictionaries=False,
               ):
  """
  Given train, dev, test treebanks and a word embedding matrix :
  - basic mode : create key_value instanes for each CHAR, WORD, U|X-POS , Relation with special cases for Roots, Padding and End symbols
  - expanding is done on dev set (we assume that dev set is accessible)
  - min_occurence : if <= considered as singleton otherwise ad
  - if expand_vocab == True : we also perform expansion on test set if test_path is not None and on dev_path
  - DEPRECIATED : based on word_embed_dict a new numpy matrix is created that will be used to th : ONLY expansion decision made on word_embed_dict
  - if pos_specific_data_set not None  :
      - build pos_dictionary from it
      - expand word dictionaries with it
  #WARNING singleton as been removed in a hardcoded way cause it was not clear what it was doing/done for

  TODO : to be tested : test based on a given conll --> vocab word is correct
      in regard to min_occurence and that the created matrix is correct also   (index --> vetor correct
  """
  printing("WARNING : CASING IS {} for dictionary ", var=[case], verbose=verbose, verbose_level=1)
  default_value = True
  if word_embed_dict is None:
    word_embed_dict = {}
  word_dictionary = Dictionary('word', default_value=default_value, singleton=True)
  word_norm_dictionary = Dictionary('word_norm', default_value=default_value, singleton=True) if word_normalization else None
  char_dictionary = Dictionary('character', default_value=default_value)
  pos_dictionary = Dictionary('pos', default_value=default_value)
  xpos_dictionary = Dictionary('xpos', default_value=default_value)
  type_dictionary = Dictionary('type', default_value=default_value)
  counter_match_train = 0
  counter_other_train = 0
  char_dictionary.add(PAD_CHAR)

  if add_start_char:
    char_dictionary.add(CHAR_START)

  char_dictionary.add(ROOT_CHAR)
  char_dictionary.add(END_CHAR)
  char_dictionary.add(END)

  pos_dictionary.add(PAD_POS)
  xpos_dictionary.add(PAD_POS)
  type_dictionary.add(PAD_TYPE)

  pos_dictionary.add(ROOT_POS)
  xpos_dictionary.add(ROOT_POS)
  type_dictionary.add(ROOT_TYPE)

  pos_dictionary.add(END_POS)
  xpos_dictionary.add(END_POS)
  type_dictionary.add(END_TYPE)

  vocab = dict()
  vocab_norm = dict()
  # read training file add to Vocab directly except for words (not word_norm)
  # ## for which we need filtering so we add them to vocab()

  assert isinstance(train_path, list)
  assert tasks is not None, "ERROR : we need tasks information along with dataset to know how to commute label dictionary"

  for train_dir, simultaneous_task_ls in zip(train_path, tasks):
    if do_not_fill_dictionaries:
      print("WARNING : do_not_fill_dictionaries is TRUE ")
      break
    printing("VOCABULARY : computing dictionary for word, char on {} for task {} ", var=[train_dir, simultaneous_task_ls], verbose=verbose, verbose_level=1)
    if len(set(simultaneous_task_ls) & set(["normalize", "all"])) > 0:
      printing("VOCABULARY : computing dictionary for normalized word also {} ", var=[train_dir, simultaneous_task_ls], verbose=verbose, verbose_level=1)
    # NB if pos OR parsing is here we actually compte the Dictionary for pos AND parsing
    elif len(set(simultaneous_task_ls) & set(["pos", "parsing", "all"])) > 0:
      printing("VOCABULARY : computing dictionary for pos and/or parsing word also ", verbose=verbose, verbose_level=1)
    with codecs.open(train_dir, 'r', 'utf-8', errors='ignore') as file:
      li = 0
      for line in file:
        line = line.strip()
        if len(line) == 0 or line[0] == '#':
          continue
        tokens = line.split('\t')
        if '-' in tokens[0] or '.' in tokens[0]:
          continue
        for char in tokens[1]:
          char_dictionary.add(char)
        word = DIGIT_RE.sub(b"0", str.encode(tokens[1])).decode()
        if case == "lower":
          word = word.lower()
        pos = tokens[3]  #if tokens[4]=='_' else tokens[3]+'$$$'+tokens[4]
        xpos = tokens[4]
        typ = tokens[7]
        #if pos_specific_data_set is None:
        # otherwise : pos-dictionary will be build with pos_specific_data_set
        #  pos_dictionary.add(pos)
        if len(set(simultaneous_task_ls) & set(["all", "pos", "parsing"]))>0:
          pos_dictionary.add(pos)
          xpos_dictionary.add(xpos)
          type_dictionary.add(typ)
        if word_normalization and len(set(simultaneous_task_ls) & set(["normalize", "all"])) > 0:
          raise(Exception("word_normalization not supported "))
          #token_norm, _ = get_normalized_token(tokens[9], 0, verbose=verbose)
          if case == "lower":
            token_norm = token_norm.lower()
          if token_norm in vocab_norm:
            vocab_norm[token_norm] += 1
          else:
            vocab_norm[token_norm] = 1
        if word in vocab:
          vocab[word] += 1
        else:
          vocab[word] = 1
        li = li + 1
        if dry_run and li == 100:
          break
  # collect singletons
  singletons = set([word for word, count in vocab.items() if count <= min_occurence])
  # if a singleton is in pretrained embedding dict, set the count to min_occur + c
  for word in vocab.keys():
    if word in word_embed_dict or word.lower() in word_embed_dict:
      # if words are in word_embed_dict we want them even they appear less then min_occurence
      vocab[word] += min_occurence
  for word_norm in vocab_norm.keys():
    # TODO : should do something if we allow word embedding on the target standart side
    pass
    #if word in word_embed_dict or word.lower() in word_embed_dict:
  vocab_norm_list = _START_VOCAB + sorted(vocab_norm, key=vocab_norm.get, reverse=True)
  # WARNING / same min_occurence for source and target word vocabulary
  vocab_norm_list = [word for word in vocab_norm_list if word in _START_VOCAB or vocab_norm[word] > min_occurence]
  if len(vocab_norm_list) > MAX_VOCABULARY_SIZE_WORD_DIC:
    printing("VOCABULARY : norm vocabulary cut to {}  tokens", var=[MAX_VOCABULARY_SIZE_WORD_DIC],verbose=verbose, verbose_level=1)
    vocab_norm_list = vocab_norm_list[:MAX_VOCABULARY_SIZE_WORD_DIC]

  vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
  # filter strictly above min_occurence
  vocab_list = [word for word in vocab_list if word in _START_VOCAB or vocab[word] > min_occurence]
  max_vocabulary_size = MAX_VOCABULARY_SIZE_WORD_DIC
  if len(vocab_list) > max_vocabulary_size:
    printing("VOCABULARY : target vocabulary cut to {} tokens", var=[MAX_VOCABULARY_SIZE_WORD_DIC], verbose=verbose, verbose_level=1)
    vocab_list = vocab_list[:max_vocabulary_size]
  word_dictionary.inv_ls = vocab_list
  printing("VOCABULARY INV added {} ".format(vocab_list), verbose_level=3, verbose=verbose)

  pos_dictionary = pos_specific_dic_builder(pos_specific_data_set, pos_dictionary, verbose=verbose)

  def expand_vocab(data_paths):
    counter_match_dev = 0
    expand = 0
    vocab_set = set(vocab_list)
    vocab_norm_set = set(vocab_norm_list)
    for data_path in data_paths:
      with codecs.open(data_path, 'r', 'utf-8', errors='ignore') as file:
        li = 0
        for line in file:
          line = line.strip()
          if len(line) == 0 or line[0] == '#':
            continue
          tokens = line.split('\t')
          if '-' in tokens[0] or '.' in tokens[0]:
            continue
          for char in tokens[1]:
            char_dictionary.add(char)
          word = DIGIT_RE.sub(b"0", str.encode(tokens[1])).decode()
          if case == "lower":
            word = word.lower()
          pos = tokens[3] # if tokens[4]=='_' else tokens[3]+'$$$'+tokens[4]
          xpos = tokens[4]
          typ = tokens[7]
          # TODO SOMEHITNG
          if word_normalization:
            token_norm, _ = get_normalized_token(tokens[9], 0, verbose=0)
          if word_normalization:
            # TODO : add word_norm_embed_dict to allow expansion !
            if False and word_norm not in vocab_norm_set:
              vocab_norm_set.add(word_norm)
              vocab_norm_list.append(word_norm)
          # TODO : ANswer : WHY WOULD WE LIKE TO EXPAND IT ON DEV, TEST ?
          #if pos_specific_data_set is None:
          #  pos_dictionary.add(pos)
          #xpos_dictionary.add(xpos)
          #type_dictionary.add(typ)
          # if word not already in vocab_set (loaded as trained and each time expand_vocab was called :
          # but found in new dataset and appear in word_embed_dict then we add it to vocab # otherwise not need to load them to vocab (they won't have any representation)
          # but found in new dataset and appear in word_embed_dict then we add it to vocab # otherwise not need to load them to vocab (they won't have any representation)
          if word not in vocab_set and (word in word_embed_dict or word.lower() in word_embed_dict):
            vocab_set.add(word)
            expand += 1
            vocab_list.append(word)
          li = li + 1
          if dry_run and li == 100:
            break
        printing("VOCABULARY EXPAND word source vocabulary expanded of {} tokens based on {} ", var=[expand, data_path],
                 verbose=verbose, verbose_level=0)
  if expand_vocab_bool:
    assert len(word_embed_dict)>0, "ERROR : how do you want to expand if no wod embedding dict"
    if isinstance(dev_path, str):
      dev_path = [dev_path]
    expand_vocab(dev_path)
    printing("VOCABULARY : EXPANDING vocabulary on {} ", var=[dev_path], verbose_level=0, verbose=verbose)
    if test_path is not None:
      if isinstance(test_path, str):
        test_path = [test_path]
      printing("VOCABULARY : EXPANDING vocabulary on {} ", var=[test_path], verbose_level=0, verbose=verbose)
      expand_vocab(test_path)
      # TODO : word_norm should be handle spcecifically
  # TODO : what is singletons for ?
  singletons = []
  if word_norm_dictionary is not None:
    for word_norm in vocab_norm_list:
      word_norm_dictionary.add(word_norm)

  for word in vocab_list:
    word_dictionary.add(word)
    if word in word_embed_dict:
      counter_match_train += 1
    else:
      counter_other_train +=1
    if word in singletons :
      word_dictionary.add_singleton(word_dictionary.get_index(word))
  word_dictionary.save(dict_path)
  if word_norm_dictionary is not None:
    word_norm_dictionary.save(dict_path)
    word_norm_dictionary_size = word_norm_dictionary.size()
    word_norm_dictionary.close()
  else:
    word_norm_dictionary_size = 0
  char_dictionary.save(dict_path)
  pos_dictionary.save(dict_path)
  xpos_dictionary.save(dict_path)
  type_dictionary.save(dict_path)
  word_dictionary.close()
  char_dictionary.close()
  pos_dictionary.close()
  xpos_dictionary.close()
  type_dictionary.close()
  if word_embed_dict != {}:
    printing("VOCABULARY EXPANSION Match with preexisting word embedding {} match in train and dev and {} no match tokens ".format(
    counter_match_train, counter_other_train), verbose=1, verbose_level=1)
  else:
    printing(
      "VOCABULARY WORDS was not expanded on dev or test cause no external word embedding dict wa provided", verbose=verbose, verbose_level=1)
  printing("VOCABULARY : {} word {} word_norm {} char {} xpos {} pos {} type encoded in vocabulary "
           "(including default token, an special tokens)",
           var=[word_dictionary.size(), word_norm_dictionary_size, char_dictionary.size(), xpos_dictionary.size(),
                pos_dictionary.size(), type_dictionary.size()],
           verbose=verbose, verbose_level=1)

  return word_dictionary, word_norm_dictionary, char_dictionary, pos_dictionary, xpos_dictionary, type_dictionary


def get_decile(reader, bucketing_level, symbolic_root, symbolic_end, must_get_norm, word_decoder, tasks, input_level_ls, verbose):
    """
    creates data-driven buckets of data based on sentences - word or wordpiece - length
    Buckets are based on hardcoded percentiles [0, 20, 40, 50, 60, 70, 80, 90, 95, 99, 100]
    """
    inst_size_ls = []
    inst = reader.getNext(symbolic_root=symbolic_root, symbolic_end=symbolic_end, must_get_norm=must_get_norm,
                          word_decoder=word_decoder, tasks=tasks, input_level_ls=input_level_ls)
    counter_corrupted = 0
    while inst is not None:
      if inst == "CORRUPTED":
        inst = reader.getNext(symbolic_root=symbolic_root, symbolic_end=symbolic_end, must_get_norm=must_get_norm,
                              input_level_ls=input_level_ls,
                              word_decoder=word_decoder, tasks=tasks, verbose=verbose)
        counter_corrupted+=1
        print("WARNING GET DECILE skipping one CORRUPTED sentences in reader {} in total ".format(counter_corrupted))
        continue
      if bucketing_level == "word":
        inst_size = inst.length()
        inst_size_ls.append(inst_size)
      elif bucketing_level == "wordpiece":
        inst_size = inst.sentence_word_piece.length()
        inst_size_ls.append(inst_size)

      inst = reader.getNext(symbolic_root=symbolic_root, symbolic_end=symbolic_end, must_get_norm=must_get_norm,
                            input_level_ls=input_level_ls, word_decoder=word_decoder, tasks=tasks, verbose=verbose)
    inst_size_ls = np.array(inst_size_ls)
    percentiles_to_get = [0, 20, 40, 50, 60, 70, 80, 90, 95, 99, 100]
    percentiles = [int(np.percentile(inst_size_ls, percentile)) for percentile in percentiles_to_get]
    printing("CONLLU DATA decile " + " ".join(["<percentile={} : {}".format(percentile, value) for percentile, value in zip(percentiles, percentiles_to_get)]), verbose=verbose, verbose_level=1)
    return percentiles


def get_bucket_level(input_level_ls, verbose):

    if "wordpiece" in input_level_ls:
      bucketing_level = "wordpiece"
    elif "word" in input_level_ls:
      bucketing_level = "word"
      printing("ITERATOR : bucketing is done based on {} level", var=[bucketing_level], verbose_level=1, verbose=verbose)
    else:
      raise(Exception("neither wordpiece or word found in input_level_ls {} ".format(input_level_ls)))
    return bucketing_level


def init_bucket(bucket, verbose):
    if bucket:
      bucket_length_words = [12, 21, 24, 29, 32, 36, 41, 43, 50, 59, -1]
      # in bpe
      # TODO : should do two passed (one to compute buckets based on deciles other)
      buckets_length_bpe_words = [12, 21, 24, 29, 32, 36, 41, 43, 50, 59, -1] # based on lexnotm15 trainign data (english ugc)
      #printing("WARNING : bucket limited to 40", verbose=verbose, verbose_level=1)
    else:
      #
      buckets_length_bpe_words = [-1]
      bucket_length_words = [-1]
      printing("WARNING : for validation we don't bucket the data : bucket len is {} (-1 means will be based "
               "on max sent length lenght) ", var=bucket_length_words[0], verbose=verbose, verbose_level=1)

    assert len(bucket_length_words) == len(buckets_length_bpe_words), "ERROR bucket word level and bpe level should be same len "

    if buckets_length_bpe_words[-1] == -1 or bucket_length_words[-1] == -1:
        assert buckets_length_bpe_words[-1] == bucket_length_words[-1], \
          "ERROR : if last bucket is -1 should be the same for both word buckt and bpe bucket"
    return bucket_length_words, buckets_length_bpe_words


def read_data(source_path, word_dictionary, char_dictionary, pos_dictionary, xpos_dictionary, type_dictionary, max_size=None,
              word_norm_dictionary=None,
              word_decoder=False,
              normalization=False, bucket=False, max_char_len=None,
              symbolic_root=False, symbolic_end=False, dry_run=False, tasks=None,
              must_get_norm=True, bert_tokenizer=None, n_token_by_batch = 200,
              input_level_ls=None,get_data_computed_bucket=True,
              run_mode="train",
              verbose=0):
  """
  Given vocabularies , data_file :
  - creates a  list of bucket
  - each bucket is a list of unicode encoded worrds, character, pos tags, relations, ... based on DependancyInstances()
   and Sentence() objects
  """
  assert input_level_ls is not None and set(input_level_ls).issubset(AVAILABLE_INPUTS)

  printing('DATA READER : Reading data from %s' % source_path, verbose_level=1, verbose=verbose)
  counter = 0
  reader = CoNLLReader(source_path, word_dictionary, char_dictionary, pos_dictionary, type_dictionary, xpos_dictionary,
                       max_char_len=max_char_len, bert_tokenizer=bert_tokenizer,
                       lemma_dictionary=None, word_norm_dictionary=word_norm_dictionary)

  bucketing_level = get_bucket_level(input_level_ls, verbose)

  assert "wordpiece" == bucketing_level, "ERROR only wordpiece supported for now "
  if get_data_computed_bucket and bucket:
    buckets_length_bpe_words = get_decile(reader, bucketing_level, symbolic_root=symbolic_root, symbolic_end=symbolic_end,
                       must_get_norm=must_get_norm, word_decoder=word_decoder, tasks=tasks,
                       input_level_ls=input_level_ls, verbose=verbose)
    #buckets_length_bpe_words = _buck
  else:
    bucket_length_words, buckets_length_bpe_words = init_bucket(bucket, verbose)
    #_buck = buckets_length_bpe_words

  printing("DATA iterator based on {} tasks with bucketing {}", var=[tasks, buckets_length_bpe_words], verbose_level=1, verbose=verbose)

  reader = CoNLLReader(source_path, word_dictionary, char_dictionary, pos_dictionary, type_dictionary, xpos_dictionary,
                       max_char_len=max_char_len, bert_tokenizer=bert_tokenizer,
                       lemma_dictionary=None, word_norm_dictionary=word_norm_dictionary)

  inst = reader.getNext(symbolic_root=symbolic_root, symbolic_end=symbolic_end, must_get_norm=must_get_norm,
                        word_decoder=word_decoder, tasks=tasks, input_level_ls=input_level_ls,
                        run_mode=run_mode)

  last_bucket_id = len(buckets_length_bpe_words) - 1
  data = [[] for _ in buckets_length_bpe_words]
  max_char_length = [0 for _ in buckets_length_bpe_words]
  max_char_norm_length = [0 for _ in buckets_length_bpe_words] if normalization else None
  counter_corrupted = 0
  inst_size_ls = []
  # batch_sizes mean is in average the number of token in each batch for each bucket
  batch_sizes_mean = int(np.mean([n_token_by_batch/max_len for max_len in buckets_length_bpe_words]))

  while inst is not None and (not dry_run or counter < 100):
    if inst == "CORRUPTED":
      inst = reader.getNext(symbolic_root=symbolic_root, symbolic_end=symbolic_end, must_get_norm=must_get_norm,
                            input_level_ls=input_level_ls,run_mode=run_mode,
                            word_decoder=word_decoder, tasks=tasks, verbose=verbose)
      print("WARNING skipping one CORRUPTED sentences in reader {} in total ".format(counter_corrupted))
      counter_corrupted += 1
      continue

    printing("Sentence : counter {} inst : {}".format(counter, inst.sentence.raw_lines[1]),
             verbose=verbose, verbose_level=5)

    sent = inst.sentence
    sent_word_piece = inst.sentence_word_piece

    if bucketing_level == "word":
      inst_size = inst.length()
    elif bucketing_level == "wordpiece":
      inst_size = inst.sentence_word_piece.length()
      inst_size_ls.append(inst_size)

    for bucket_id, bucket_size in enumerate(buckets_length_bpe_words):
      if inst_size < bucket_size or bucket_id == last_bucket_id:
        data[bucket_id].append([sent.all_indexes, sent.word_ids, sent.word_norm_ids, sent.char_id_seqs, sent.char_norm_ids_seq, inst.pos_ids, inst.heads, inst.type_ids,
                                counter, sent.words, sent.word_norm, sent.raw_lines, inst.xpos_ids,
                                sent_word_piece.word_piece_raw_tokens,
                                sent_word_piece.word_piece_raw_tokens_aligned,
                                sent_word_piece.word_piece_words,
                                sent_word_piece.word_piece_lemmas,
                                sent_word_piece.word_piece_normalization,
                                sent_word_piece.word_piece_normalization_index,
                                sent_word_piece.word_piece_raw_tokens_aligned_index,
                                sent_word_piece.word_piece_words_index,
                                sent_word_piece.word_piece_raw_tokens_index,
                                sent_word_piece.is_mwe,
                                sent_word_piece.n_masks_to_add_in_raw_label,
                                sent_word_piece.is_first_bpe_of_token,
                                sent_word_piece.is_first_bpe_of_norm,
                                sent_word_piece.is_first_bpe_of_words,
                                sent_word_piece.word_piece_normalization_target_aligned_with_word,
                                sent_word_piece.word_piece_normalization_target_aligned_with_word_index,
                                sent_word_piece.word_piece_words_src_aligned_with_norm,
                                sent_word_piece.word_piece_words_src_aligned_with_norm_index,
                                sent_word_piece.to_norm,
                                sent_word_piece.n_masks_to_append_src_to_norm,
                                sent_word_piece.n_words])
        if "char" in input_level_ls:
          max_char_len = max([len(char_seq) for char_seq in sent.char_seqs])
          if normalization :
            max_char_norm_len = max([len(char_norm_seq) for char_norm_seq in sent.char_norm_ids_seq])
        # defining maximum characters lengh per bucket both for normalization and
        # we define a max_char_len per bucket !
          if max_char_length[bucket_id] < max_char_len:
            max_char_length[bucket_id] = max_char_len
          if normalization:
            if max_char_norm_length[bucket_id] < max_char_norm_len:
              max_char_norm_length[bucket_id] = max_char_norm_len

        if bucket_id == last_bucket_id and buckets_length_bpe_words[last_bucket_id] < len(sent_word_piece.word_piece_words):
          buckets_length_bpe_words[last_bucket_id] = len(sent_word_piece.word_piece_words)+2
        if buckets_length_bpe_words[bucket_id] == -1:
          buckets_length_bpe_words[bucket_id] = len(sent_word_piece.word_piece_words)+2
        break
    inst = reader.getNext(symbolic_root=symbolic_root, symbolic_end=symbolic_end, input_level_ls=input_level_ls,
                          run_mode=run_mode,
                          must_get_norm=must_get_norm, word_decoder=word_decoder, tasks=tasks)
    counter += 1
    if inst is None or not (not dry_run or counter < 100):
      printing("Breaking : breaking because inst {} counter<100 {} dry {} ".format(inst is None, counter < 100, dry_run),
               verbose=verbose, verbose_level=3)
  reader.close()

  return data, {"buckets_length_bpe_words": buckets_length_bpe_words if bert_tokenizer is not None else None,
                "max_char_length": max_char_length,
                "max_char_norm_length": max_char_norm_length, "n_sent": counter,
                "mean_batch_size": batch_sizes_mean}, buckets_length_bpe_words


def read_data_to_variable(source_path, word_dictionary, char_dictionary, pos_dictionary, xpos_dictionary,
                          type_dictionary, pad_id,max_size=None, symbolic_root=False,word_norm_dictionary=None,
                          symbolic_end=False,dry_run=False,
                          verbose=0, normalization=False, bucket=True, word_decoder=False,
                          tasks=None, max_char_len=None, must_get_norm=True, bert_tokenizer=None,
                          input_level_ls=None, run_mode="train",
                          add_end_char=0, add_start_char=0):
  """
  Given data ovject form read_variable creates array-like  variables for character, word, pos, relation, heads ready to be fed to a network
  """
  if max_char_len is None:
    max_char_len = MAX_CHAR_LENGTH
  if "norm_not_norm" in tasks:
    assert normalization, "norm_not_norm can't be set without normalisation info"
  printing("WARNING symbolic root {} is and symbolic end is {} ", var=[symbolic_root, symbolic_end], verbose=verbose, verbose_level=1)
  data, max_char_length_dic, _buckets = read_data(source_path, word_dictionary, char_dictionary, pos_dictionary,
                                                  xpos_dictionary, type_dictionary, bucket=bucket, word_norm_dictionary=word_norm_dictionary,
                                                  verbose=verbose, max_size=max_size, normalization=normalization,
                                                  symbolic_root=symbolic_root,
                                                  word_decoder=word_decoder, tasks=tasks,max_char_len=max_char_len,
                                                  must_get_norm=must_get_norm,
                                                  bert_tokenizer=bert_tokenizer,
                                                  input_level_ls=input_level_ls,
                                                  run_mode=run_mode,
                                                  symbolic_end=symbolic_end, dry_run=dry_run)

  max_char_length = max_char_length_dic["max_char_length"]
  max_char_norm_length = max_char_length_dic["max_char_norm_length"]

  printing("DATA MAX_CHAR_LENGTH set to {}".format(max_char_len), verbose=verbose, verbose_level=1)

  bucket_sizes = [len(data[b]) for b in range(len(_buckets))]
  printing("DATA checking bucket {} are resp. len {} (n_sent) ", var=[bucket_sizes, _buckets], verbose=verbose, verbose_level=1)
  data_variable = []

  ss = [0] * len(_buckets)
  ss1 = [0] * len(_buckets)

  for bucket_id in range(len(_buckets)):
    bucket_size = bucket_sizes[bucket_id]
    if bucket_size == 0:
      data_variable.append((1, 1))
      continue
    bucket_length = _buckets[bucket_id]+10
    # NB "WARNING : adding 10 dimension to bucket_length "for avoiding over loading of words buckets due to bpe bucketing (bucket id {})".format(bucket_id))
    if max_char_length_dic["buckets_length_bpe_words"] is not None:
      bucket_length_in_bpe = max_char_length_dic["buckets_length_bpe_words"][bucket_id]
    char_length = min(max_char_len+NUM_CHAR_PAD, max_char_length[bucket_id] + NUM_CHAR_PAD)
    if max_char_len+NUM_CHAR_PAD < max_char_length[bucket_id] + NUM_CHAR_PAD:
      printing("WARNING : Iterator conllu_data CUTTING bucket {} to max allowed length {}",
               var=[max_char_len+NUM_CHAR_PAD],
               verbose=verbose, verbose_level=1)

    wid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64) if "word" in input_level_ls else None
    cid_inputs = np.empty([bucket_size, bucket_length, char_length], dtype=np.int64) if "char" in input_level_ls else None
    pid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64) if "pos" in tasks else None
    xpid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64) if "pos" in tasks else None
    hid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64) if "parsing" in tasks else None
    tid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64) if "parsing" in tasks else None

    wordpieces_inputs_raw_tokens = np.empty([bucket_size, bucket_length_in_bpe], dtype=np.int64) if "wordpiece" in input_level_ls else None

    is_mwe_label = np.empty([bucket_size, bucket_length_in_bpe], dtype=np.int64) if "mwe_detection" in tasks else None
    n_masks_to_app_in_raw_label = np.empty([bucket_size, bucket_length_in_bpe], dtype=np.int64) if "n_masks_mwe" in tasks is not None else None

    wordpieces_words = np.empty([bucket_size, bucket_length_in_bpe], dtype=np.int64) if "wordpiece" in input_level_ls else None
    wordpieces_raw_aligned_with_words = np.empty([bucket_size, bucket_length_in_bpe], dtype=np.int64) if "wordpiece" in input_level_ls and "mwe_prediction" in tasks else None

    wordpiece_normalization = np.empty([bucket_size, bucket_length_in_bpe], dtype=np.int64) if "wordpiece" in input_level_ls and "normalize" in tasks else None
    ind_wordpiece_normalization_alignement_index = [] if "wordpiece" in input_level_ls and "normalize" in tasks else None

    bucket_length_in_bpe_noisy_aligned = bucket_length_in_bpe+5
    wordpiece_normalization_target_aligned_with_word = np.empty([bucket_size, bucket_length_in_bpe_noisy_aligned ],dtype=np.int64) if "wordpiece" in input_level_ls and "normalize" in tasks else None
    ind_wordpiece_normalization_target_aligned_with_word_index = [] if "wordpiece" in input_level_ls and "normalize" in tasks else None
    # TODO : should bucket_length_in_bpe programatically (not by hardcoding)
    wordpiece_words_src_aligned_with_norm = np.empty([bucket_size, bucket_length_in_bpe_noisy_aligned],dtype=np.int64) if "wordpiece" in input_level_ls and "normalize" in tasks else None
    ind_wordpiece_words_src_aligned_with_norm_index = [] if "wordpiece" in input_level_ls and "normalize" in tasks else None

    n_masks_for_norm = np.empty([bucket_size, bucket_length_in_bpe], dtype=np.int64) if "wordpiece" in input_level_ls and "normalize" in tasks else None
    to_norm_np = np.empty([bucket_size, bucket_length_in_bpe], dtype=np.int64) if "wordpiece" in input_level_ls and "normalize" in tasks else None

    ind_wordpieces_words_alignement_index = [] if "wordpiece" in input_level_ls else None
    ind_wordpieces_inputs_raw_tokens_alignement_index = [] if "wordpiece" in input_level_ls and "mwe_prediction" in tasks else None
    ind_wordpieces_raw_aligned_alignement_index = [] if "wordpiece" in input_level_ls and "mwe_prediction" in tasks else None
    all_indexes = []

    if normalization:
      char_norm_length = min(max_char_len+NUM_CHAR_PAD, max_char_norm_length[bucket_id] + NUM_CHAR_PAD)
      cids_norm = np.empty([bucket_size, bucket_length, char_norm_length], dtype=np.int64)

      wid_norm_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64) if word_decoder else None

      word_norm_not_norm = np.empty([bucket_size, bucket_length], dtype=np.int64) if "norm_not_norm" in tasks else None
      edit = np.empty([bucket_size, bucket_length], dtype=np.float32) if "edit_prediction" in tasks else None

    masks_inputs = np.zeros([bucket_size, bucket_length], dtype=np.float32)
    single_inputs = np.zeros([bucket_size, bucket_length], dtype=np.int64)

    lengths_inputs = np.empty(bucket_size, dtype=np.int64)
    
    order_inputs = np.empty(bucket_size, dtype=np.int64)
    raw_word_inputs, raw_lines = [], []
    words_normalized_str = []

    n_total_word = 0

    for i, inst in enumerate(data[bucket_id]):
      ss[bucket_id] += 1
      ss1[bucket_id] = bucket_length
      indexes, wids, wids_norm, cid_seqs, cid_norm_seqs, pids, hids, tids, orderid, word_raw, normalized_str, lines, xpids, \
      word_piece_raw_tokens, word_piece_raw_tokens_aligned, word_piece_words, word_piece_lemmas, \
      word_piece_normalization, word_piece_normalization_index,\
      word_piece_raw_tokens_aligned_index, word_piece_words_index, word_piece_raw_tokens_index, \
      is_mwe, n_masks_to_add_in_raw_label, is_first_bpe_of_token, is_first_bpe_of_norm, is_first_bpe_of_words, \
      word_piece_normalization_target_aligned_with_word, word_piece_normalization_target_aligned_with_word_index,\
      word_piece_words_src_aligned_with_norm, word_piece_words_src_aligned_with_norm_index,\
      to_norm, n_masks_to_append_src_to_norm, n_words = inst

      if n_words is not None:
        n_total_word += n_words

      inst_size = len(wids)
      lengths_inputs[i] = inst_size
      order_inputs[i] = orderid
      raw_word_inputs.append(word_raw)
      words_normalized_str.append(normalized_str)
      # word piece for raw token

      wordpieces_inputs_raw_tokens = fill_array(wordpieces_inputs_raw_tokens, word_piece_raw_tokens, i, pad_id)
      # word segmentaition related
      n_masks_to_app_in_raw_label = fill_array(n_masks_to_app_in_raw_label, n_masks_to_add_in_raw_label, i, PAD_ID_LOSS_STANDART)
      is_mwe_label = fill_array(is_mwe_label, is_mwe, i, PAD_ID_LOSS_STANDART)
      ind_wordpieces_inputs_raw_tokens_alignement_index = fill_index_list(ind_wordpieces_inputs_raw_tokens_alignement_index, word_piece_raw_tokens_index, PAD_ID_LOSS_STANDART, bucket_length_in_bpe)
      # word piece for (syntax) word
      wordpieces_words = fill_array(wordpieces_words, word_piece_words, i, pad_id)
      wordpieces_raw_aligned_with_words = fill_array(wordpieces_raw_aligned_with_words, word_piece_raw_tokens_aligned, i, pad_id)
      ind_wordpieces_words_alignement_index = fill_index_list(ind_wordpieces_words_alignement_index, word_piece_raw_tokens_aligned_index, PAD_ID_LOSS_STANDART, bucket_length_in_bpe)
      #ind_wordpieces_words_alignement_index = fill_index_list(ind_wordpieces_words_alignement_index, word_piece_words_index, PAD_ID_LOSS_STANDART, bucket_length_in_bpe)
      all_indexes = fill_index_list(all_indexes, indexes, PAD_ID_LOSS_STANDART, bucket_length)

      # for normalisation
      wordpiece_normalization = fill_array(wordpiece_normalization, word_piece_normalization, i, pad_id)
      ind_wordpiece_normalization_alignement_index = fill_index_list(ind_wordpiece_normalization_alignement_index, word_piece_normalization_index, PAD_ID_LOSS_STANDART,bucket_length_in_bpe)

      wordpiece_normalization_target_aligned_with_word = fill_array(wordpiece_normalization_target_aligned_with_word, word_piece_normalization_target_aligned_with_word, i, pad_id)
      ind_wordpiece_normalization_target_aligned_with_word_index = fill_index_list(ind_wordpiece_normalization_target_aligned_with_word_index,  word_piece_normalization_target_aligned_with_word_index, PAD_ID_LOSS_STANDART, bucket_length_in_bpe_noisy_aligned)

      wordpiece_words_src_aligned_with_norm = fill_array(wordpiece_words_src_aligned_with_norm, word_piece_words_src_aligned_with_norm,  i, pad_id)
      ind_wordpiece_words_src_aligned_with_norm_index = fill_index_list(ind_wordpiece_words_src_aligned_with_norm_index, word_piece_words_src_aligned_with_norm_index,  PAD_ID_LOSS_STANDART, bucket_length_in_bpe_noisy_aligned)

      n_masks_for_norm = fill_array(n_masks_for_norm, n_masks_to_append_src_to_norm, i, pad_id)
      to_norm_np = fill_array(to_norm_np, to_norm, i, pad_id)

      # word ids
      wid_inputs = fill_array(wid_inputs, wids, i, PAD_ID_WORD)

      if normalization:
        wid_norm_inputs = fill_array(wid_norm_inputs, wids_norm, i, PAD_ID_WORD)
      shift, shift_end = 0, 0

      if add_start_char:
        shift += 1
      if add_end_char:
        shift_end += 1

      if "char" in input_level_ls:
        for w, cids in enumerate(cid_seqs):
          if add_start_char:
            cid_inputs[i, w, 0] = CHAR_START_ID

          cid_inputs[i, w, shift:len(cids)+shift] = cids

          if add_end_char:
            cid_inputs[i, w, len(cids)+shift] = CHAR_END_ID
          cid_inputs[i, w, shift+len(cids)+shift_end:] = PAD_ID_CHAR

        cid_inputs[i, inst_size:, :] = PAD_ID_CHAR

      assert not normalization
      # pos ids
      pid_inputs = fill_array(pid_inputs, pids, i, PAD_ID_TAG)
      # xpos ids
      xpid_inputs = fill_array(xpid_inputs, xpids, i, PAD_ID_TAG)
      # type ids
      tid_inputs = fill_array(tid_inputs, tids, i, PAD_ID_TAG)
      # heads
      hid_inputs = fill_array(hid_inputs, hids, i, PAD_ID_HEADS)
      masks_inputs[i, :inst_size] = 1.0
      for j, wid in enumerate(wids):
        if word_dictionary.is_singleton(wid):
          single_inputs[i, j] = 1
      raw_lines.append(lines)

    words = Variable(torch.from_numpy(wid_inputs), requires_grad=False) if wid_inputs is not None else None
    chars = Variable(torch.from_numpy(cid_inputs), requires_grad=False) if cid_inputs is not None else None
    word_norm = Variable(torch.from_numpy(wid_norm_inputs), requires_grad=False) if normalization and word_decoder else None
    chars_norm = Variable(torch.from_numpy(cids_norm), requires_grad=False) if normalization else None
    word_norm_not_norm = Variable(torch.from_numpy(word_norm_not_norm), requires_grad=False) if "norm_not_norm" in tasks else None
    edit = Variable(torch.from_numpy(edit), requires_grad=False) if "edit_prediction" in tasks else None

    wordpieces_words = Variable(torch.from_numpy(wordpieces_words), requires_grad=False) if wordpieces_words is not None else None
    wordpieces_raw_aligned_with_words = Variable(torch.from_numpy(wordpieces_raw_aligned_with_words), requires_grad=False) if wordpieces_raw_aligned_with_words is not None else None

    is_mwe_label = Variable(torch.from_numpy(is_mwe_label), requires_grad=False) if is_mwe_label is not None else None
    n_masks_to_app_in_raw_label = Variable(torch.from_numpy(n_masks_to_app_in_raw_label), requires_grad=False) if n_masks_to_app_in_raw_label is not None else None
    wordpieces_inputs_raw_tokens = Variable(torch.from_numpy(wordpieces_inputs_raw_tokens), requires_grad=False) if wordpieces_inputs_raw_tokens is not None else None
    #
    wordpiece_normalization = Variable(torch.from_numpy(wordpiece_normalization), requires_grad=False) if wordpiece_normalization is not None else None
    wordpiece_normalization_target_aligned_with_word = Variable(torch.from_numpy(wordpiece_normalization_target_aligned_with_word), requires_grad=False) if wordpiece_normalization_target_aligned_with_word is not None else None
    wordpiece_words_src_aligned_with_norm = Variable(torch.from_numpy(wordpiece_words_src_aligned_with_norm), requires_grad=False) if wordpiece_words_src_aligned_with_norm is not None else None
    n_masks_for_norm = Variable(torch.from_numpy(n_masks_for_norm), requires_grad=False) if n_masks_for_norm is not None else None
    to_norm_np = Variable(torch.from_numpy(to_norm_np), requires_grad=False) if to_norm is not None else None

    ind_wordpieces_raw_aligned_alignement_index = np.array(ind_wordpieces_raw_aligned_alignement_index) if ind_wordpieces_raw_aligned_alignement_index is not None else None
    ind_wordpieces_inputs_raw_tokens_alignement_index = np.array(ind_wordpieces_inputs_raw_tokens_alignement_index) if ind_wordpieces_inputs_raw_tokens_alignement_index is not None else None
    ind_wordpieces_words_alignement_index = np.array(ind_wordpieces_words_alignement_index) if ind_wordpieces_words_alignement_index is not None else None
    ind_wordpiece_words_src_aligned_with_norm_index = np.array(ind_wordpiece_words_src_aligned_with_norm_index) if ind_wordpiece_words_src_aligned_with_norm_index is not None else None
    ind_wordpiece_normalization_target_aligned_with_word_index = np.array(ind_wordpiece_normalization_target_aligned_with_word_index) if ind_wordpiece_normalization_target_aligned_with_word_index is not None else None
    ind_wordpiece_normalization_alignement_index = np.array(ind_wordpiece_normalization_alignement_index) if ind_wordpiece_normalization_alignement_index is not None else None
    all_indexes = np.array(all_indexes)

    # we don't put as pytorch alignement indexes
    pos = Variable(torch.from_numpy(pid_inputs), requires_grad=False) if pid_inputs is not None else None
    xpos = Variable(torch.from_numpy(xpid_inputs), requires_grad=False) if xpid_inputs is not None else None
    heads = Variable(torch.from_numpy(hid_inputs), requires_grad=False) if hid_inputs is not None else None
    types = Variable(torch.from_numpy(tid_inputs), requires_grad=False) if tid_inputs is not None else None
    masks = Variable(torch.from_numpy(masks_inputs), requires_grad=False) if masks_inputs is not None else None
    single = Variable(torch.from_numpy(single_inputs), requires_grad=False) if single_inputs is not None else None
    lengths = torch.from_numpy(lengths_inputs)

    if n_total_word is not None:
      printing("INFO {} total token processed and loaded", var=[n_total_word], verbose_level=1,verbose=verbose)
    data_variable.append((all_indexes, words, word_norm,
                          wordpieces_words, wordpieces_raw_aligned_with_words, wordpieces_inputs_raw_tokens,
                          ind_wordpieces_words_alignement_index, ind_wordpieces_raw_aligned_alignement_index, ind_wordpieces_inputs_raw_tokens_alignement_index,
                          is_mwe_label, n_masks_to_app_in_raw_label,
                          wordpiece_normalization, ind_wordpiece_normalization_alignement_index,
                          wordpiece_normalization_target_aligned_with_word, ind_wordpiece_normalization_target_aligned_with_word_index,
                          wordpiece_words_src_aligned_with_norm, ind_wordpiece_words_src_aligned_with_norm_index,
                          n_masks_for_norm, to_norm_np,
                          chars, chars_norm, word_norm_not_norm, edit, pos, xpos, heads, types,
                          masks, single, lengths, order_inputs, raw_word_inputs, words_normalized_str, raw_lines))

  return data_variable, bucket_sizes, _buckets, max_char_length_dic["n_sent"], max_char_length_dic["mean_batch_size"]


def get_batch_variable(data,
                       batch_size,
                       max_token_per_batch=300,
                       lattice=None,
                       normalization=False,):
  """
  given read_data_to_variable() get a random batches in buck
  """
  data_variable, bucket_sizes, _buckets, _,_ = data
  total_size = float(sum(bucket_sizes))
  # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
  # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
  # the size if i-th training bucket, as used later.
  buckets_scale = [sum(bucket_sizes[:i + 1]) / total_size for i in range(len(bucket_sizes))]
  # Choose a bucket according to data distribution. We pick a random number
  # in [0, 1] and use the corresponding interval in train_buckets_scale.
  random_number = np.random.random_sample()
  bucket_id = min([i for i in range(len(buckets_scale)) if buckets_scale[i] > random_number])
  bucket_length = _buckets[bucket_id]

  MAX_BPE_ACCEPTED = 0

  if MAX_BPE_ACCEPTED:
    while bucket_length > MAX_BPE_ACCEPTED:
      random_number = np.random.random_sample()
      bucket_id = min([i for i in range(len(buckets_scale)) if buckets_scale[i] > random_number])
      bucket_length = _buckets[bucket_id]
    print("WARNING : ONLY TAKING BUCKETS THAT ARE MAXIMUM {} BPE ".format(MAX_BPE_ACCEPTED))
  all_indexes, words, word_norm, wordpieces_words, wordpieces_raw_aligned_with_words, wordpieces_inputs_raw_tokens, \
  ind_wordpieces_words_alignement_index, ind_wordpieces_raw_aligned_alignement_index, ind_wordpieces_inputs_raw_tokens_alignement_index,\
  is_mwe_label, n_masks_to_app_in_raw_label, \
  wordpiece_normalization, ind_wordpiece_normalization_alignement_index, wordpiece_normalization_target_aligned_with_word, ind_wordpiece_normalization_target_aligned_with_word_index, \
  wordpiece_words_src_aligned_with_norm, ind_wordpiece_words_src_aligned_with_norm_index, n_masks_for_norm, to_norm_np,\
  chars, chars_norm, word_norm_not_norm, edit, pos, xpos, heads, types, masks, single, lengths, order_inputs, raw, normalized_str, raw_lines = data_variable[bucket_id]
  bucket_len_max = _buckets[bucket_id]
  bucket_size = bucket_sizes[bucket_id]
  
  if max_token_per_batch is not None:
    assert max_token_per_batch > 0
    batch_size = max(max_token_per_batch//bucket_len_max, 1)
    if batch_size == 1:
      printing("WARNING : {} is above {} for bucket id {} out of {} buckets ", var=[bucket_len_max, max_token_per_batch, bucket_id, len(_buckets)],verbose=1, verbose_level=1)
    printing("INFO (max_token_per_batch not none {} bucket max len {})  setting batch_size to {}", var=[max_token_per_batch, bucket_len_max, batch_size], verbose=1, verbose_level=1)

  batch_size = min(bucket_size, batch_size)
  index = torch.randperm(bucket_size).long()[:batch_size]

  words = query_batch(index, words)
  #words = words[index]

  # syntax word
  wordpieces_words = query_batch(index, wordpieces_words)
  ind_wordpieces_words_alignement_index  = query_batch(index, ind_wordpieces_words_alignement_index )
  # syntax word aligned with raw token
  wordpieces_raw_aligned_with_words = query_batch(index, wordpieces_raw_aligned_with_words)
  ind_wordpieces_raw_aligned_alignement_index = query_batch(index, ind_wordpieces_raw_aligned_alignement_index)
  # raw token
  wordpieces_inputs_raw_tokens = query_batch(index, wordpieces_inputs_raw_tokens)
  ind_wordpieces_inputs_raw_tokens_alignement_index = query_batch(index, ind_wordpieces_inputs_raw_tokens_alignement_index)
  # word segmentaiton : is_mwe_label
  is_mwe_label = query_batch(index, is_mwe_label)
  # n_masks
  n_masks_to_app_in_raw_label = query_batch(index, n_masks_to_app_in_raw_label)
  all_indexes = query_batch(index, all_indexes)

  pos = query_batch(index, pos)
  xpos = query_batch(index, xpos)

  heads = query_batch(index, heads)
  types = query_batch(index, types)

  chars = query_batch(index, chars)
  chars_norm = query_batch(index, chars_norm)
  word_norm_not_norm = query_batch(index, word_norm_not_norm)
  word_norm = query_batch(index, word_norm)
  edit = query_batch(index, edit)
  raw = [raw[i.cpu().item()] for i in index]
  normalized_str = [normalized_str[i.cpu().item()] for i in index]

  wordpiece_normalization = query_batch(index, wordpiece_normalization)
  ind_wordpiece_normalization_alignement_index = query_batch(index, ind_wordpiece_normalization_alignement_index)

  wordpiece_normalization_target_aligned_with_word = query_batch(index, wordpiece_normalization_target_aligned_with_word)
  ind_wordpiece_normalization_target_aligned_with_word_index = query_batch(index, ind_wordpiece_normalization_target_aligned_with_word_index)
  wordpiece_words_src_aligned_with_norm = query_batch(index, wordpiece_words_src_aligned_with_norm)
  ind_wordpiece_words_src_aligned_with_norm_index = query_batch(index, ind_wordpiece_words_src_aligned_with_norm_index)
  n_masks_for_norm = query_batch(index, n_masks_for_norm)
  to_norm_np = query_batch(index, to_norm_np)

  return all_indexes, words, word_norm, wordpieces_words, wordpieces_raw_aligned_with_words, wordpieces_inputs_raw_tokens, \
         ind_wordpieces_words_alignement_index, ind_wordpieces_raw_aligned_alignement_index, ind_wordpieces_inputs_raw_tokens_alignement_index, \
         is_mwe_label, n_masks_to_app_in_raw_label, \
         wordpiece_normalization, ind_wordpiece_normalization_alignement_index, \
         wordpiece_normalization_target_aligned_with_word, ind_wordpiece_normalization_target_aligned_with_word_index, \
         wordpiece_words_src_aligned_with_norm, ind_wordpiece_words_src_aligned_with_norm_index, n_masks_for_norm, to_norm_np,\
         chars, chars_norm, word_norm_not_norm, edit, pos, xpos, heads, types,\
         masks[index], lengths[index], order_inputs[index.cpu()], raw, normalized_str, raw_lines


def iterate_batch_variable(data, batch_size, unk_replace=0.,
                           word_decoding=False,
                           lattice=None, normalization=False, verbose=1):
  """
  Iterate over the dataset based on read_data_to_variable() object (used for evaluation/strict prediction)
  """

  data_variable, bucket_sizes, _buckets, _,_ = data
  bucket_indices = np.arange(len(_buckets))

  for bucket_id in bucket_indices:
    bucket_size = bucket_sizes[bucket_id]
    bucket_length = _buckets[bucket_id]
    if bucket_size == 0:
      continue

    all_indexes, words, word_norm, wordpieces_words, wordpieces_raw_aligned_with_words, wordpieces_inputs_raw_tokens, \
    ind_wordpieces_words_alignement_index, ind_wordpieces_raw_aligned_alignement_index, ind_wordpieces_inputs_raw_tokens_alignement_index, \
    is_mwe_label, n_masks_to_app_in_raw_label, \
    wordpiece_normalization, ind_wordpiece_normalization_alignement_index, wordpiece_normalization_target_aligned_with_word, \
    ind_wordpiece_normalization_target_aligned_with_word_index, wordpiece_words_src_aligned_with_norm, ind_wordpiece_words_src_aligned_with_norm_index, \
    n_masks_for_norm, to_norm_np,\
    chars, chars_norm, word_norm_not_norm, edit, pos, xpos, heads, types, \
    masks, single, lengths, order_ids, raw_word_inputs, normalized_str, raw_lines = data_variable[bucket_id]

    if unk_replace:
      ones = Variable(single.data.new(bucket_size, bucket_length).fill_(1))
      noise = Variable(masks.data.new(bucket_size, bucket_length).bernoulli_(unk_replace).long())
      words = words * (ones - single * noise)
    _word_norm = None
    _edit = None
    for start_idx in range(0, bucket_size, batch_size):
      excerpt = slice(start_idx, start_idx + batch_size)
      _chars_norm = query_batch(excerpt, chars_norm)
      _word_norm_not_norm = query_batch(excerpt, word_norm_not_norm)
      _word_norm = query_batch(excerpt, word_norm)
      _wordpieces_words = wordpieces_words[excerpt] if wordpieces_words is not None else None
      _ind_wordpieces_words_alignement_index = ind_wordpieces_words_alignement_index[excerpt] if ind_wordpieces_words_alignement_index is not None else None
      _all_indexes = all_indexes[excerpt] if all_indexes is not None else None
      _wordpieces_raw_aligned_with_words = wordpieces_raw_aligned_with_words[excerpt] if wordpieces_raw_aligned_with_words is not None else None
      _ind_wordpieces_raw_aligned_alignement_index = ind_wordpieces_raw_aligned_alignement_index[excerpt] if ind_wordpieces_raw_aligned_alignement_index is not None else None

      _wordpieces_inputs_raw_tokens = wordpieces_inputs_raw_tokens[excerpt] if wordpieces_inputs_raw_tokens is not None else wordpieces_inputs_raw_tokens
      _ind_wordpieces_inputs_raw_tokens_alignement_index = ind_wordpieces_inputs_raw_tokens_alignement_index[excerpt] if ind_wordpieces_inputs_raw_tokens_alignement_index is not None else None
      _is_mwe_label = is_mwe_label[excerpt] if is_mwe_label is not None else None
      _n_masks_to_app_in_raw_label = n_masks_to_app_in_raw_label[excerpt] if n_masks_to_app_in_raw_label is not None else None

      _chars = query_batch(excerpt, chars)
      _pos = query_batch(excerpt, pos)
      _xpos = query_batch(excerpt, xpos)
      _heads = query_batch(excerpt, heads)
      _types = query_batch(excerpt, types)

      _words = query_batch(excerpt, words)

      _wordpiece_normalization = query_batch(excerpt, wordpiece_normalization)
      _ind_wordpiece_normalization_alignement_index = query_batch(excerpt, ind_wordpiece_normalization_alignement_index)

      _wordpiece_normalization_target_aligned_with_word = query_batch(excerpt, wordpiece_normalization_target_aligned_with_word)
      _ind_wordpiece_normalization_target_aligned_with_word_index = query_batch(excerpt, ind_wordpiece_normalization_target_aligned_with_word_index)

      _wordpiece_words_src_aligned_with_norm = query_batch(excerpt, wordpiece_words_src_aligned_with_norm)
      _ind_wordpiece_words_src_aligned_with_norm_index = query_batch(excerpt, ind_wordpiece_words_src_aligned_with_norm_index)

      _n_masks_for_norm = query_batch(excerpt, n_masks_for_norm)
      _to_norm_np = query_batch(excerpt, to_norm_np)

      yield _all_indexes, _words, _word_norm, _wordpieces_words, _wordpieces_raw_aligned_with_words, _wordpieces_inputs_raw_tokens, \
            _ind_wordpieces_words_alignement_index, _ind_wordpieces_raw_aligned_alignement_index, _ind_wordpieces_inputs_raw_tokens_alignement_index, \
            _is_mwe_label, _n_masks_to_app_in_raw_label, \
            _wordpiece_normalization, _ind_wordpiece_normalization_alignement_index,\
            _wordpiece_normalization_target_aligned_with_word, _ind_wordpiece_normalization_target_aligned_with_word_index, \
            _wordpiece_words_src_aligned_with_norm, _ind_wordpiece_words_src_aligned_with_norm_index, \
            _n_masks_for_norm, _to_norm_np, \
            _chars, _chars_norm , _word_norm_not_norm, _edit, \
            _pos, _xpos, _heads, _types,  \
            masks[excerpt], lengths[excerpt], order_ids[excerpt], \
            raw_word_inputs[excerpt], normalized_str[excerpt], raw_lines[excerpt]
