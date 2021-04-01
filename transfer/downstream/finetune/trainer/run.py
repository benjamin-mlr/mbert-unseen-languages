from transfer.downstream.finetune.env.imports import pdb, OrderedDict, json, torch, re, np, SummaryWriter, time, os, sys, git
from transfer.downstream.finetune.io_.logger import printing
from transfer.downstream.finetune.env.gpu_tools.gpu_info import use_gpu_, printout_allocated_gpu_memory

from transfer.downstream.finetune.env.flags import REPORT_FLAG_DIR_STR
from transfer.downstream.finetune.io_.manage_dirs.make_dirs import setup_repoting_location
from transfer.downstream.finetune.model.optimization.get_optmizers import apply_fine_tuning_strategy

try:
    from transfer.downstream.finetune.io_.runs_tracker.google_sheet_report import update_status
except:
    update_status = None

from transfer.downstream.finetune.io_.data_iterator import readers_load, data_gen_multi_task_sampling_batch
from transfer.downstream.finetune.io_.dat import conllu_data

from transfer.downstream.finetune.io_.build_files_shard import build_shard
from transfer.downstream.finetune.io_.get_new_batcher import get_new_shard

from transfer.downstream.finetune.trainer.tools.multi_task_tools import get_vocab_size_and_dictionary_per_task, update_batch_size_mean
from transfer.downstream.finetune.trainer.epoch_run import epoch_run
from transfer.downstream.finetune.model.architecture.get_model import get_model_multi_task_bert
from transfer.downstream.finetune.io_.report.report_tools import write_args, get_hyperparameters_dict, get_dataset_label, get_name_model_id_with_extra_name

from transfer.downstream.finetune.env.dir.project_directories import CHECKPOINT_BERT_DIR
from transfer.downstream.finetune.env.vars import N_SENT_MAX_CONLL_PER_SHARD
from transfer.downstream.finetune.model.settings import TASKS_PARAMETER
from transfer.downstream.finetune.transformers.transformers.tokenization_bert import BertTokenizer
from transfer.downstream.finetune.transformers.transformers.tokenization_xlm import XLMTokenizer


def run(args,
        n_observation_max_per_epoch_train,
        vocab_size, model_dir,
        voc_tokenizer, auxilliary_task_norm_not_norm,
        null_token_index, null_str, tokenizer,
        n_observation_max_per_epoch_dev_test=None,
        run_mode="train",
        dict_path=None, end_predictions=None,
        report=True,
        model_suffix="", description="",
        saving_every_epoch=10,
        model_location=None, model_id=None,
        report_full_path_shared=None, skip_1_t_n=False,
        heuristic_test_ls=None,
        remove_mask_str_prediction=False, inverse_writing=False,
        extra_label_for_prediction="",
        random_iterator_train=True, bucket_test=False, must_get_norm_test=True,
        early_stoppin_metric=None, subsample_early_stoping_metric_val=None,
        compute_intersection_score_test=True,
        threshold_edit=3,
        name_with_epoch=False,max_token_per_batch=200,
        encoder=None,
        debug=False, verbose=1):

    """
    Wrapper for training/prediction/evaluation

    2 modes : train (will train using train and dev iterators with test at the end on test_path)
              test : only test at the end : requires all directories to be created
    :return:
    """
    assert run_mode in ["train", "test"], "ERROR run mode {} corrupted ".format(run_mode)
    input_level_ls = ["wordpiece"]
    assert early_stoppin_metric is not None and subsample_early_stoping_metric_val is not None, "ERROR : assert early_stoppin_metric should be defined and subsample_early_stoping_metric_val "
    if n_observation_max_per_epoch_dev_test is None:
        n_observation_max_per_epoch_dev_test = n_observation_max_per_epoch_train
    printing("MODEL : RUNNING IN {} mode", var=[run_mode], verbose=verbose, verbose_level=1)
    printing("WARNING : casing was set to {} (this should be consistent at train and test)", var=[args.case], verbose=verbose, verbose_level=2)

    if len(args.tasks) == 1:
        printing("INFO : MODEL : 1 set of simultaneous tasks {}".format(args.tasks), verbose=verbose, verbose_level=1)

    if run_mode == "test":
        assert args.test_paths is not None and isinstance(args.test_paths, list)
    if run_mode == "train":
        printing("CHECKPOINTING info : "
                 "saving model every {}", var=saving_every_epoch, verbose=verbose, verbose_level=1)

    use_gpu = use_gpu_(use_gpu=None, verbose=verbose)

    def get_commit_id():
        repo = git.Repo(os.path.dirname(os.path.realpath(__file__)), search_parent_directories=True)
        git_commit_id = str(repo.head.commit)  # object.hexsha
        return git_commit_id
    if verbose>1:
        print(f"GIT ID : {get_commit_id()}")

    train_data_label = get_dataset_label(args.train_path, default="train")

    iter_train = 0
    iter_dev = 0
    row = None
    writer = None

    printout_allocated_gpu_memory(verbose, "{} starting all".format(model_id))

    if run_mode == "train":
        if os.path.isdir(args.train_path[0]) and len(args.train_path) == 1:
            data_sharded = args.train_path[0]
            printing("INFO args.train_path is directory so not rebuilding shards", verbose=verbose, verbose_level=1)
        elif os.path.isdir(args.train_path[0]):
            raise(Exception(" {} is a directory but len is more than one , not supported".format(args.train_path[0], len(args.train_path))))
        else:
            data_sharded = None
        assert model_location is None and model_id is None, "ERROR we are creating a new one "

        model_id, model_location, dict_path, tensorboard_log, end_predictions, data_sharded \
            = setup_repoting_location(model_suffix=model_suffix, data_sharded=data_sharded,
                                      root_dir_checkpoints=CHECKPOINT_BERT_DIR,
                                      shared_id=args.overall_label, verbose=verbose)
        hyperparameters = get_hyperparameters_dict(args, args.case, random_iterator_train, seed=args.seed, verbose=verbose,
                                                   dict_path=dict_path,
                                                   model_id=model_id, model_location=model_location)
        args_dir = write_args(model_location, model_id=model_id, hyperparameters=hyperparameters, verbose=verbose)

        if report:
            if report_full_path_shared is not None:
                tensorboard_log = os.path.join(report_full_path_shared, "tensorboard")
            printing("tensorboard --logdir={} --host=localhost --port=1234 ", var=[tensorboard_log], verbose_level=1,verbose=verbose)
            writer = SummaryWriter(log_dir=tensorboard_log)
            if writer is not None:
                writer.add_text("INFO-ARGUMENT-MODEL-{}".format(model_id), str(hyperparameters), 0)
    else:
        args_checkpoint = json.load(open(args.init_args_dir, "r"))
        dict_path = args_checkpoint["hyperparameters"]["dict_path"]
        assert dict_path is not None and os.path.isdir(dict_path), "ERROR {} ".format(dict_path)
        end_predictions = args.end_predictions
        assert end_predictions is not None and os.path.isdir(end_predictions), "ERROR end_predictions"
        model_location = args_checkpoint["hyperparameters"]["model_location"]
        model_id = args_checkpoint["hyperparameters"]["model_id"]
        assert model_location is not None and model_id is not None, "ERROR model_location model_id "
        args_dir = os.path.join(model_location, "{}-args.json".format(model_id))

        printing("CHECKPOINTING : starting writing log \ntensorboard --logdir={} --host=localhost --port=1234 ",
                 var=[os.path.join(model_id, "tensorboard")], verbose_level=1,
                 verbose=verbose)

    # build or make dictionaries
    _dev_path = args.dev_path if args.dev_path is not None else args.train_path
    word_dictionary, word_norm_dictionary, char_dictionary, pos_dictionary, \
    xpos_dictionary, type_dictionary = \
        conllu_data.load_dict(dict_path=dict_path,
                              train_path=args.train_path if run_mode == "train" else None,
                              dev_path=args.dev_path if run_mode == "train" else None,
                              test_path=None,
                              word_embed_dict={},
                              dry_run=False,
                              expand_vocab=False,
                              word_normalization=True,
                              force_new_dic=True if run_mode == "train" else False,
                              tasks=args.tasks,
                              pos_specific_data_set=args.train_path[1] if len(args.tasks) > 1 and len(args.train_path)>1 and "pos" in args.tasks else None,
                              case=args.case,
                              # if not normalize pos or parsing in tasks we don't need dictionary
                              do_not_fill_dictionaries=len(set(["normalize", "pos", "parsing"])&set([task for tasks in args.tasks for task in tasks])) == 0,
                              add_start_char=1 if run_mode == "train" else None,
                              verbose=verbose)
    # we flatten the taskssd
    printing("DICTIONARY CREATED/LOADED", verbose=verbose, verbose_level=1)
    num_labels_per_task, task_to_label_dictionary = get_vocab_size_and_dictionary_per_task([task for tasks in args.tasks for task in tasks],
                                                                                           vocab_bert_wordpieces_len=vocab_size,
                                                                                           pos_dictionary=pos_dictionary,
                                                                                           type_dictionary=type_dictionary,
                                                                                           task_parameters=TASKS_PARAMETER)
    voc_pos_size = num_labels_per_task["pos"] if "pos" in args.tasks else None
    if voc_pos_size is not None:
        printing("MODEL : voc_pos_size defined as {}", var=voc_pos_size,  verbose_level=1, verbose=verbose)
    printing("MODEL init...", verbose=verbose, verbose_level=1)
    if verbose>1:
        print("DEBUG : TOKENIZER :voc_tokenizer from_pretrained", voc_tokenizer)
    #pdb.set_trace()
    #voc_tokenizer = "bert-base-multilingual-cased"
    tokenizer = tokenizer.from_pretrained(voc_tokenizer, do_lower_case=args.case == "lower",shuffle_bpe_embedding=args.shuffle_bpe_embedding)
    mask_id = tokenizer.convert_tokens_to_ids(tokenizer.mask_token) #convert_tokens_to_ids([MASK_BERT])[0]
    printout_allocated_gpu_memory(verbose, "{} loading model ".format(model_id))
    model = get_model_multi_task_bert(args=args, model_dir=model_dir, encoder=encoder,
                                      num_labels_per_task=num_labels_per_task, mask_id=mask_id)

    def prune_heads(prune_heads):
        if prune_heads is not None:
            pune_heads_ls = prune_heads.split(",")[:-1]
            assert len(pune_heads_ls) > 0
            for layer in pune_heads_ls:
                parsed_layer_to_prune =layer.split("-")
                assert parsed_layer_to_prune[0] == "prune_heads"
                assert parsed_layer_to_prune[1] == "layer"
                assert parsed_layer_to_prune[3] == "heads"
                heads = parsed_layer_to_prune[4]
                head_index_ls = heads.split("_")
                heads_ls = [int(index) for index in head_index_ls]
                print(f"MODEL : pruning layer {parsed_layer_to_prune[2]} heads {heads_ls}")
                model.encoder.encoder.layer[int(parsed_layer_to_prune[2])].attention.prune_heads(heads_ls)
    if args.prune_heads is not None and args.prune_heads!="None":
        print(f"INFO : args.prune_heads {args.prune_heads}")
        prune_heads(args.prune_heads)

    if use_gpu:
        model.to("cuda")
        printing("MODEL TO CUDA", verbose=verbose, verbose_level=1)
    printing("MODEL model.config {} ", var=[model.config], verbose=verbose, verbose_level=1)
    printout_allocated_gpu_memory(verbose, "{} model loaded".format(model_id))
    model_origin = OrderedDict()
    pruning_mask = OrderedDict()
    printout_allocated_gpu_memory(verbose, "{} model cuda".format(model_id))
    for name, param in model.named_parameters():
        model_origin[name] = param.detach().clone()
        printout_allocated_gpu_memory(verbose, "{} param cloned ".format(name))
        if args.penalization_mode == "pruning":
            abs = torch.abs(param.detach().flatten())
            median_value = torch.median(abs)
            pruning_mask[name] = (abs > median_value).float()
        printout_allocated_gpu_memory(verbose, "{} pruning mask loaded".format(model_id))

    printout_allocated_gpu_memory(verbose, "{} model clone".format(model_id))

    inv_word_dic = word_dictionary.instance2index
    # load , mask, bucket and index data

    assert tokenizer is not None, "ERROR : tokenizer is None , voc_tokenizer failed to be loaded {}".format(voc_tokenizer)
    if run_mode == "train":
        time_load_readers_train_start = time.time()
        if not args.memory_efficient_iterator:

            data_sharded, n_shards, n_sent_dataset_total_train = None, None, None
            args_load_batcher_shard_data = None
            printing("INFO : starting loading readers", verbose=verbose, verbose_level=1)
            readers_train = readers_load(datasets=args.train_path,
                                         tasks=args.tasks,
                                         word_dictionary=word_dictionary,
                                         bert_tokenizer=tokenizer,
                                         word_dictionary_norm=word_norm_dictionary, char_dictionary=char_dictionary,
                                         pos_dictionary=pos_dictionary, xpos_dictionary=xpos_dictionary,
                                         type_dictionary=type_dictionary,
                                         word_decoder=True,
                                         run_mode=run_mode,
                                         add_start_char=1, add_end_char=1, symbolic_end=1,
                                         symbolic_root=1, bucket=True,
                                         must_get_norm=True, input_level_ls=input_level_ls,
                                         verbose=verbose)
            n_sent_dataset_total_train = readers_train[list(readers_train.keys())[0]][3]
            printing("INFO : done with sharding", verbose=verbose, verbose_level=1)
        else:
            printing("INFO : building/loading shards ", verbose=verbose, verbose_level=1)
            data_sharded, n_shards, n_sent_dataset_total_train = build_shard(data_sharded, args.train_path, n_sent_max_per_file=N_SENT_MAX_CONLL_PER_SHARD, verbose=verbose)

        time_load_readers_dev_start = time.time()
        time_load_readers_train = time.time()-time_load_readers_train_start
        readers_dev_ls = []
        dev_data_label_ls = []
        printing("INFO : g readers for dev", verbose=verbose, verbose_level=1)
        printout_allocated_gpu_memory(verbose, "{} reader train loaded".format(model_id))
        for dev_path in args.dev_path:
            dev_data_label = get_dataset_label(dev_path, default="dev")
            dev_data_label_ls.append(dev_data_label)
            readers_dev = readers_load(datasets=dev_path, tasks=args.tasks, word_dictionary=word_dictionary,
                                       word_dictionary_norm=word_norm_dictionary, char_dictionary=char_dictionary,
                                       pos_dictionary=pos_dictionary, xpos_dictionary=xpos_dictionary,
                                       bert_tokenizer=tokenizer,
                                       type_dictionary=type_dictionary,
                                       word_decoder=True, run_mode=run_mode,
                                       add_start_char=1, add_end_char=1,
                                       symbolic_end=1, symbolic_root=1, bucket=False,
                                       must_get_norm=True,input_level_ls=input_level_ls,
                                       verbose=verbose) if args.dev_path is not None else None
            readers_dev_ls.append(readers_dev)
        printout_allocated_gpu_memory(verbose, "{} reader dev loaded".format(model_id))

        time_load_readers_dev = time.time()-time_load_readers_dev_start
        # Load tokenizer
        printing("TIME : {} ", var=[OrderedDict([("time_load_readers_train", "{:0.4f} min".format(time_load_readers_train/60)), ("time_load_readers_dev",  "{:0.4f} min".format(time_load_readers_dev/60))])],
                 verbose=verbose, verbose_level=2)

        early_stoping_val_former = 1000
        # training starts when epoch is 1
        #args.epochs += 1
        #assert args.epochs >= 1, "ERROR need at least 2 epochs (1 eval , 1 train 1 eval"
        flexible_batch_size = False
        
        if args.optimizer == "AdamW":
            model, optimizer, scheduler = apply_fine_tuning_strategy(model=model,
                                                                     fine_tuning_strategy=args.fine_tuning_strategy,
                                                                     lr_init=args.lr, betas=(0.9, 0.99),epoch=0,
                                                                     weight_decay=args.weight_decay,
                                                                     optimizer_name=args.optimizer,
                                                                     t_total=n_sent_dataset_total_train / args.batch_update_train * args.epochs if n_sent_dataset_total_train / args.batch_update_train * args.epochs > 1 else 5,
                                                                     verbose=verbose)

        try:
            for epoch in range(args.epochs):
                if args.memory_efficient_iterator:
                    # we start epoch with a new shart everytime !
                    training_file = get_new_shard(data_sharded, n_shards)
                    printing("INFO Memory efficient iterator triggered (only build for train data , starting with {}",
                             var=[training_file], verbose=verbose, verbose_level=1)
                    args_load_batcher_shard_data = {"word_dictionary": word_dictionary, "tokenizer": tokenizer,
                                                    "word_norm_dictionary": word_norm_dictionary,
                                                    "char_dictionary": char_dictionary,
                                                    "pos_dictionary": pos_dictionary,
                                                    "xpos_dictionary": xpos_dictionary,
                                                    "type_dictionary": type_dictionary, "use_gpu": use_gpu,
                                                    "norm_not_norm": auxilliary_task_norm_not_norm,
                                                    "word_decoder": True,
                                                    "add_start_char": 1, "add_end_char": 1, "symbolic_end": 1,
                                                    "symbolic_root": 1,
                                                    "bucket": True, "max_char_len": 20, "must_get_norm": True,
                                                    "use_gpu_hardcoded_readers": False,
                                                    "bucketing_level": "bpe", "input_level_ls": ["wordpiece"],
                                                    "auxilliary_task_norm_not_norm": auxilliary_task_norm_not_norm,
                                                    "random_iterator_train": random_iterator_train
                                                    }

                    readers_train = readers_load(datasets=args.train_path if not args.memory_efficient_iterator else training_file,
                                                 tasks=args.tasks, word_dictionary=word_dictionary,
                                                 bert_tokenizer=tokenizer, word_dictionary_norm=word_norm_dictionary, char_dictionary=char_dictionary,
                                                 pos_dictionary=pos_dictionary, xpos_dictionary=xpos_dictionary,
                                                 type_dictionary=type_dictionary,
                                                 word_decoder=True, run_mode=run_mode,
                                                 add_start_char=1, add_end_char=1, symbolic_end=1,
                                                 symbolic_root=1, bucket=True,
                                                 must_get_norm=True, input_level_ls=input_level_ls, verbose=verbose)

                checkpointing_model_data = (epoch % saving_every_epoch == 0 or epoch == (args.epochs - 1))
                # build iterator on the loaded data
                printout_allocated_gpu_memory(verbose, "{} loading batcher".format(model_id))


                if args.batch_size == "flexible":
                    flexible_batch_size = True

                    printing("INFO : args.batch_size {} so updating it based on mean value {}",
                             var=[args.batch_size, update_batch_size_mean(readers_train)],
                             verbose=verbose, verbose_level=1)
                    args.batch_size = update_batch_size_mean(readers_train)

                    if args.batch_update_train == "flexible":
                        args.batch_update_train = args.batch_size
                    printing("TRAINING : backward pass every {} step of size {} in average",
                             var=[int(args.batch_update_train // args.batch_size), args.batch_size],
                             verbose=verbose, verbose_level=1)
                    try:
                        assert isinstance(args.batch_update_train // args.batch_size, int)\
                           and args.batch_update_train // args.batch_size > 0, \
                            "ERROR batch_size {} should be a multiple of {} ".format(args.batch_update_train, args.batch_size)
                    except Exception as e:
                        print("WARNING {}".format(e))
                batchIter_train = data_gen_multi_task_sampling_batch(tasks=args.tasks,
                                                                     readers=readers_train,
                                                                     batch_size=readers_train[list(readers_train.keys())[0]][4],
                                                                     max_token_per_batch=max_token_per_batch if flexible_batch_size else None,
                                                                     word_dictionary=word_dictionary,
                                                                     char_dictionary=char_dictionary,
                                                                     pos_dictionary=pos_dictionary,
                                                                     word_dictionary_norm=word_norm_dictionary,
                                                                     get_batch_mode=random_iterator_train,
                                                                     print_raw=False,
                                                                     dropout_input=0.0,
                                                                     verbose=verbose)

                # -|-|-
                printout_allocated_gpu_memory(verbose, "{} batcher train loaded".format(model_id))
                batchIter_dev_ls = []
                batch_size_DEV = 1

                if verbose > 1:
                    print("WARNING : batch_size for final eval was hardcoded and set to {}".format(batch_size_DEV))
                for readers_dev in readers_dev_ls:
                    batchIter_dev = data_gen_multi_task_sampling_batch(tasks=args.tasks, readers=readers_dev,
                                                                       batch_size=batch_size_DEV,
                                                                       word_dictionary=word_dictionary,
                                                                       char_dictionary=char_dictionary,
                                                                       pos_dictionary=pos_dictionary,
                                                                       word_dictionary_norm=word_norm_dictionary,
                                                                       get_batch_mode=False,
                                                                       print_raw=False,

                                                                       dropout_input=0.0,
                                                                       verbose=verbose) if args.dev_path is not None else None
                    batchIter_dev_ls.append(batchIter_dev)


                model.train()
                printout_allocated_gpu_memory(verbose, "{} batcher dev loaded".format(model_id))
                if args.optimizer != "AdamW":

                    model, optimizer, scheduler = apply_fine_tuning_strategy(model=model,
                                                                         fine_tuning_strategy=args.fine_tuning_strategy,
                                                                         lr_init=args.lr, betas=(0.9, 0.99),
                                                                         weight_decay=args.weight_decay,
                                                                         optimizer_name=args.optimizer,
                                                                         t_total=n_sent_dataset_total_train / args.batch_update_train * args.epochs if n_sent_dataset_total_train / args.batch_update_train*args.epochs > 1 else 5,
                                                                         epoch=epoch, verbose=verbose)
                printout_allocated_gpu_memory(verbose, "{} optimizer loaded".format(model_id))
                loss_train = None

                if epoch >= 0:
                    printing("TRAINING : training on GET_BATCH_MODE ", verbose=verbose, verbose_level=2)
                    printing("TRAINING {} training 1 'epoch' = {} observation size args.batch_"
                             "update_train (foward {} batch_size {} backward  "
                             "(every int(args.batch_update_train//args.batch_size) step if {})) ",
                             var=[model_id, n_observation_max_per_epoch_train, args.batch_size, args.batch_update_train,
                                  args.low_memory_foot_print_batch_mode],
                             verbose=verbose, verbose_level=1)
                    loss_train, iter_train, perf_report_train, _ = epoch_run(batchIter_train, tokenizer,
                                                                             args=args,
                                                                             model_origin=model_origin,
                                                                             pruning_mask=pruning_mask,
                                                                             task_to_label_dictionary=task_to_label_dictionary,
                                                                             data_label=train_data_label,
                                                                             model=model,
                                                                             dropout_input_bpe=args.dropout_input_bpe,
                                                                             writer=writer,
                                                                             iter=iter_train, epoch=epoch,
                                                                             writing_pred=epoch == (args.epochs - 1),
                                                                             dir_end_pred=end_predictions,
                                                                             optimizer=optimizer, use_gpu=use_gpu,
                                                                             scheduler=scheduler,
                                                                             predict_mode=(epoch-1)%5 == 0,
                                                                             skip_1_t_n=skip_1_t_n,
                                                                             model_id=model_id,
                                                                             reference_word_dic={"InV": inv_word_dic},
                                                                             null_token_index=null_token_index, null_str=null_str,
                                                                             norm_2_noise_eval=False,
                                                                             early_stoppin_metric=None,
                                                                             n_obs_max=n_observation_max_per_epoch_train,
                                                                             data_sharded_dir=data_sharded,
                                                                             n_shards=n_shards,
                                                                             n_sent_dataset_total=n_sent_dataset_total_train,
                                                                             args_load_batcher_shard_data=args_load_batcher_shard_data,
                                                                             memory_efficient_iterator=args.memory_efficient_iterator,
                                                                             verbose=verbose)

                else:
                    printing("TRAINING : skipping first epoch to start by evaluating on devs dataset0", verbose=verbose, verbose_level=1)
                printout_allocated_gpu_memory(verbose, "{} epoch train done".format(model_id))
                model.eval()

                if args.dev_path is not None and (epoch%3==0 or epoch<=6):
                    if verbose > 1:
                        print("RUNNING DEV on ITERATION MODE")
                    early_stoping_val_ls = []
                    loss_dev_ls = []
                    for i_dev, batchIter_dev in enumerate(batchIter_dev_ls):
                        loss_dev, iter_dev, perf_report_dev, early_stoping_val = epoch_run(batchIter_dev, tokenizer,
                                                                                           args=args,
                                                                                           epoch=epoch,
                                                                                           model_origin=model_origin,
                                                                                           pruning_mask=pruning_mask,
                                                                                           task_to_label_dictionary=task_to_label_dictionary,
                                                                                           iter=iter_dev, use_gpu=use_gpu,
                                                                                           model=model,
                                                                                           writer=writer,
                                                                                           optimizer=None,
                                                                                           writing_pred=True,#epoch == (args.epochs - 1),
                                                                                           dir_end_pred=end_predictions,
                                                                                           predict_mode=True,
                                                                                           data_label=dev_data_label_ls[i_dev],
                                                                                           null_token_index=null_token_index,
                                                                                           null_str=null_str,
                                                                                           model_id=model_id,
                                                                                           skip_1_t_n=skip_1_t_n,
                                                                                           dropout_input_bpe=0,
                                                                                           reference_word_dic={"InV": inv_word_dic},
                                                                                           norm_2_noise_eval=False,
                                                                                           early_stoppin_metric=early_stoppin_metric,
                                                                                           subsample_early_stoping_metric_val=subsample_early_stoping_metric_val,
                                                                                           #case=case,
                                                                                           n_obs_max=n_observation_max_per_epoch_dev_test,
                                                                                           verbose=verbose)

                        printing("TRAINING : loss train:{} dev {}:{} for epoch {}  out of {}",
                                 var=[loss_train, i_dev, loss_dev, epoch, args.epochs], verbose=1, verbose_level=1)
                        printing("PERFORMANCE {} DEV {} {} ", var=[epoch, i_dev+1, perf_report_dev], verbose=verbose,
                                 verbose_level=1)
                        early_stoping_val_ls.append(early_stoping_val)
                        loss_dev_ls.append(loss_dev)

                    else:
                        if verbose > 1:
                            print("NO DEV EVAL")
                        loss_dev, iter_dev, perf_report_dev = None, 0, None
                # NB : early_stoping_val is based on first dev set
                printout_allocated_gpu_memory(verbose, "{} epoch dev done".format(model_id))

                early_stoping_val = early_stoping_val_ls[0]
                if checkpointing_model_data or early_stoping_val < early_stoping_val_former:
                    if early_stoping_val is not None:
                        _epoch = "best" if early_stoping_val < early_stoping_val_former else epoch
                    else:
                        if verbose > 1:
                            print('WARNING early_stoping_val is None so saving based on checkpointing_model_data only')
                        _epoch = epoch
                    # model_id enriched possibly with some epoch informaiton if name_with_epoch
                    _model_id = get_name_model_id_with_extra_name(epoch=epoch, _epoch=_epoch,
                                                                 name_with_epoch=name_with_epoch, model_id=model_id)
                    checkpoint_dir = os.path.join(model_location, "{}-checkpoint.pt".format(_model_id))

                    if _epoch == "best":
                        printing("CHECKPOINT : SAVING BEST MODEL {} (epoch:{}) (new loss is {} former was {})".format(checkpoint_dir, epoch, early_stoping_val, early_stoping_val_former), verbose=verbose, verbose_level=1)
                        last_checkpoint_dir_best = checkpoint_dir
                        early_stoping_val_former = early_stoping_val
                        best_epoch = epoch
                        best_loss = early_stoping_val
                    else:
                        printing("CHECKPOINT : NOT SAVING BEST MODEL : new loss {} did not beat first loss {}".format(early_stoping_val , early_stoping_val_former), verbose_level=1, verbose=verbose)
                    last_model = ""
                    if epoch == (args.epochs - 1):
                        last_model = "last"
                    printing("CHECKPOINT : epoch {} saving {} model {} ", var=[epoch,last_model, checkpoint_dir], verbose=verbose,verbose_level=1)
                    torch.save(model.state_dict(), checkpoint_dir)

                    args_dir = write_args(dir=model_location, checkpoint_dir=checkpoint_dir,
                                          hyperparameters=hyperparameters if name_with_epoch else None,
                                          model_id=_model_id,
                                          info_checkpoint=OrderedDict([("epochs", epoch+1),
                                                                       ("batch_size", args.batch_size if not args.low_memory_foot_print_batch_mode else args.batch_update_train),
                                                                       ("train_path", train_data_label), ("dev_path", dev_data_label_ls), ("num_labels_per_task", num_labels_per_task)]),
                                          verbose=verbose)

            if row is not None and update_status is not None:
                update_status(row=row, value="training-done", verbose=1)
        except Exception as e:
            if row is not None and update_status is not None:
                update_status(row=row, value="ERROR", verbose=1)
            raise(e)

    # reloading last (best) checkpoint
    if run_mode in ["train", "test"] and args.test_paths is not None:
        report_all = []
        if run_mode == "train" and args.epochs>0:
            if use_gpu:
                model.load_state_dict(torch.load(last_checkpoint_dir_best))
                model = model.cuda()
                printout_allocated_gpu_memory(verbose, "{} after reloading model".format(model_id))
            else:
                model.load_state_dict(torch.load(last_checkpoint_dir_best, map_location=lambda storage, loc: storage))
            printing("MODEL : RELOADING best model of epoch {} with loss {} based on {}({}) metric (from checkpoint {})", var=[best_epoch, best_loss, early_stoppin_metric, subsample_early_stoping_metric_val, last_checkpoint_dir_best], verbose=verbose, verbose_level=1)

        model.eval()

        printout_allocated_gpu_memory(verbose, "{} starting test".format(model_id))
        for test_path in args.test_paths:
            assert len(test_path) == len(args.tasks), "ERROR test_path {} args.tasks {}".format(test_path, args.tasks)
            for test, task_to_eval in zip(test_path, args.tasks):
                label_data = get_dataset_label([test], default="test")
                if len(extra_label_for_prediction) > 0:
                    label_data += "-" + extra_label_for_prediction

                if args.shuffle_bpe_embedding and args.test_mode_no_shuffle_embedding:
                    printing("TOKENIZER: as args.shuffle_bpe_embedding {} and test_mode_no_shuffle {} : reloading tokenizer with no shuffle_embedding",
                             var=[args.shuffle_bpe_embedding, args.test_mode_no_shuffle_embedding], verbose=1, verbose_level=1)
                    tokenizer = tokenizer.from_pretrained(voc_tokenizer, do_lower_case=args.case == "lower", shuffle_bpe_embedding=False)
                readers_test = readers_load(datasets=[test],
                                            tasks=[task_to_eval],
                                            word_dictionary=word_dictionary,
                                            word_dictionary_norm=word_norm_dictionary,
                                            char_dictionary=char_dictionary,
                                            pos_dictionary=pos_dictionary,
                                            xpos_dictionary=xpos_dictionary,
                                            type_dictionary=type_dictionary,
                                            bert_tokenizer=tokenizer,
                                            word_decoder=True,
                                            run_mode=run_mode,
                                            add_start_char=1, add_end_char=1, symbolic_end=1,
                                            symbolic_root=1, bucket=bucket_test,
                                            input_level_ls=input_level_ls,
                                            must_get_norm=must_get_norm_test,
                                            verbose=verbose)

                heuritics_zip = [None]
                gold_error_or_not_zip = [False]
                norm2noise_zip = [False]

                if heuristic_test_ls is None:
                    assert len(gold_error_or_not_zip) == len(heuritics_zip) and len(heuritics_zip) == len(norm2noise_zip)

                batch_size_TEST = 1
                if verbose>1:
                    print("WARNING : batch_size for final eval was hardcoded and set to {}".format(batch_size_TEST))
                for (heuristic_test, gold_error, norm_2_noise_eval) in zip(heuritics_zip, gold_error_or_not_zip, norm2noise_zip):

                    assert heuristic_test is None and not gold_error and not norm_2_noise_eval

                    batchIter_test = data_gen_multi_task_sampling_batch(tasks=[task_to_eval], readers=readers_test,
                                                                        batch_size=batch_size_TEST,
                                                                        word_dictionary=word_dictionary,
                                                                        char_dictionary=char_dictionary,
                                                                        pos_dictionary=pos_dictionary,
                                                                        word_dictionary_norm=word_norm_dictionary,
                                                                        get_batch_mode=False,
                                                                        dropout_input=0.0,
                                                                        verbose=verbose)
                    try:
                        loss_test, iter_test, perf_report_test, _ = epoch_run(batchIter_test, tokenizer,
                                                                              args=args,
                                                                              iter=iter_dev, use_gpu=use_gpu,
                                                                              model=model,
                                                                              task_to_label_dictionary=task_to_label_dictionary,
                                                                              writer=None,
                                                                              writing_pred=True,
                                                                              optimizer=None,
                                                                              args_dir=args_dir, model_id=model_id,
                                                                              dir_end_pred=end_predictions,
                                                                              skip_1_t_n=skip_1_t_n,
                                                                              predict_mode=True, data_label=label_data,
                                                                              epoch="LAST", extra_label_for_prediction=label_data,
                                                                              null_token_index=null_token_index,
                                                                              null_str=null_str,
                                                                              log_perf=False,
                                                                              dropout_input_bpe=0,
                                                                              norm_2_noise_eval=norm_2_noise_eval,
                                                                              compute_intersection_score=compute_intersection_score_test,
                                                                              remove_mask_str_prediction=remove_mask_str_prediction,
                                                                              reference_word_dic={"InV": inv_word_dic},
                                                                              threshold_edit=threshold_edit,
                                                                              verbose=verbose,
                                                                              n_obs_max=n_observation_max_per_epoch_dev_test)
                        if verbose>1:
                            print("LOSS TEST", loss_test)
                    except Exception as e:
                        print("ERROR (epoch_run test) {} test_path {} , heuristic {} , gold error {} , norm2noise {} ".format(e, test, heuristic_test, gold_error, norm_2_noise_eval))
                        raise(e)
                    print("PERFORMANCE TEST on data  {} is {} ".format(label_data, perf_report_test))
                    print("DATA WRITTEN {}".format(end_predictions))
                    if writer is not None:
                        writer.add_text("Accuracy-{}-{}-{}".format(model_id, label_data, run_mode),
                                        "After {} epochs with {} : performance is \n {} ".format(args.epochs, description,
                                                                                                 str(perf_report_test)), 0)
                    else:
                        printing("WARNING : could not add accuracy to tensorboard cause writer was found None", verbose=verbose,
                                 verbose_level=2)
                    report_all.extend(perf_report_test)
                    printout_allocated_gpu_memory(verbose, "{} test done".format(model_id))
    else:
        printing("ERROR : EVALUATION none cause {} empty or run_mode {} ",
                 var=[args.test_paths, run_mode], verbose_level=1, verbose=verbose)

    if writer is not None:
        writer.close()
        printing("tensorboard --logdir={} --host=localhost --port=1234 ", var=[tensorboard_log], verbose_level=1,verbose=verbose)

    report_dir = os.path.join(model_location, model_id+"-report.json")
    if report_full_path_shared is not None:
        report_full_dir = os.path.join(report_full_path_shared, args.overall_label + "-report.json")
        if os.path.isfile(report_full_dir):
            report = json.load(open(report_full_dir, "r"))
        else:
            report = []
            printing("REPORT = creating overall report at {} ", var=[report_dir], verbose=verbose, verbose_level=1)
        report.extend(report_all)
        json.dump(report, open(report_full_dir, "w"))
        printing("{} {} ", var=[REPORT_FLAG_DIR_STR, report_full_dir], verbose=0, verbose_level=0)

    json.dump(report_all, open(report_dir, "w"))
    printing("REPORTING TO {}".format(report_dir), verbose=verbose, verbose_level=1)
    if report_full_path_shared is None:
        printing("WARNING ; report_full_path_shared is None", verbose=verbose, verbose_level=1)
        printing("{} {} ", var=[REPORT_FLAG_DIR_STR, report_dir], verbose=verbose, verbose_level=0)

    return model
