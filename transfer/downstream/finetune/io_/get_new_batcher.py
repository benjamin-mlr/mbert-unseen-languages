from transfer.downstream.finetune.env.imports import random, os, pdb, time
from transfer.downstream.finetune.io_.logger import printing
from transfer.downstream.finetune.io_.data_iterator import readers_load, data_gen_multi_task_sampling_batch



def get_new_shard(shard_path, n_shards, rand=True, verbose=1):
    # pick a new file randomly

    assert rand

    i_shard = random.choice(range(n_shards))

    path = os.path.join(shard_path, "train_{}.conll".format(i_shard))

    assert os.path.isfile(path), "ERROR {}".format(path)

    printing("INFO : picking shard {} ", var=[path], verbose=verbose, verbose_level=1)
    return [path]


def load_batcher_shard_data(args, args_load_batcher_shard_data,
                            shard_dir, verbose):

    word_dictionary, tokenizer, word_norm_dictionary, char_dictionary,\
    pos_dictionary, xpos_dictionary, type_dictionary, use_gpu,\
    norm_not_norm, word_decoder, add_start_char, add_end_char, symbolic_end,\
    symbolic_root, bucket, max_char_len, must_get_norm, bucketing_level,\
    use_gpu_hardcoded_readers, auxilliary_task_norm_not_norm, random_iterator_train = \
        args_load_batcher_shard_data["word_dictionary"],\
        args_load_batcher_shard_data["tokenizer"], args_load_batcher_shard_data["word_norm_dictionary"], \
        args_load_batcher_shard_data["char_dictionary"], args_load_batcher_shard_data["pos_dictionary"], \
        args_load_batcher_shard_data["xpos_dictionary"], args_load_batcher_shard_data["type_dictionary"], \
        args_load_batcher_shard_data["use_gpu"], args_load_batcher_shard_data["norm_not_norm"], \
        args_load_batcher_shard_data["word_decoder"], args_load_batcher_shard_data["add_start_char"], \
        args_load_batcher_shard_data["add_end_char"], args_load_batcher_shard_data["symbolic_end"], \
        args_load_batcher_shard_data["symbolic_root"], args_load_batcher_shard_data["bucket"], \
        args_load_batcher_shard_data["max_char_len"], args_load_batcher_shard_data["must_get_norm"], args_load_batcher_shard_data["bucketing_level"], \
        args_load_batcher_shard_data["use_gpu_hardcoded_readers"], args_load_batcher_shard_data["auxilliary_task_norm_not_norm"], args_load_batcher_shard_data["random_iterator_train"],

    printing("INFO ITERATOR LOADING new batcher based on {} ", var=[shard_dir], verbose=verbose, verbose_level=1)
    start = time.time()
    readers = readers_load(datasets=shard_dir,
                           tasks=args.tasks,
                           args=args,
                           word_dictionary=word_dictionary,
                           bert_tokenizer=tokenizer,
                           word_dictionary_norm=word_norm_dictionary, char_dictionary=char_dictionary,
                           pos_dictionary=pos_dictionary, xpos_dictionary=xpos_dictionary,
                           type_dictionary=type_dictionary, use_gpu=use_gpu_hardcoded_readers,
                           norm_not_norm=auxilliary_task_norm_not_norm,
                           word_decoder=True,
                           add_start_char=1, add_end_char=1, symbolic_end=1,
                           symbolic_root=1, bucket=True, max_char_len=20,
                           input_level_ls=args_load_batcher_shard_data["input_level_ls"],
                           must_get_norm=True,
                           verbose=verbose)

    batchIter = data_gen_multi_task_sampling_batch(tasks=args.tasks, readers=readers,
                                                   batch_size=args.batch_size,
                                                   word_dictionary=word_dictionary,
                                                   char_dictionary=char_dictionary,
                                                   pos_dictionary=pos_dictionary,
                                                   word_dictionary_norm=word_norm_dictionary,
                                                   get_batch_mode=random_iterator_train,
                                                   print_raw=False,
                                                   dropout_input=0.0,
                                                   verbose=verbose)
    end = time.time()-start
    printing("INFO ITERATOR LOADED  {:0.3f}min ", var=[end/60], verbose=verbose, verbose_level=1)

    return batchIter
