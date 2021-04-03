from transfer.downstream.finetune.io_.logger import printing
from transfer.downstream.finetune.env.imports import pdb, os, json, re, OrderedDict, np
from transfer.downstream.finetune.env.dir.project_directories import CHECKPOINT_BERT_DIR
from transfer.downstream.finetune.env.dir.data_dir import get_code_data

#from env.project_variables import REPO_DATASET
REPO_DATASET = {}

def get_score(scores, metric, info_score, task, data):
    for report in scores:
        if report["metric"] == metric and report["info_score"] == info_score and report["task"] == task and report["data"] == data:
            return report
    raise (Exception(
        "REPORT with {} metric {} info_score {} task and {} data not found in {} ".format(metric, info_score, task, data,
                                                                                         scores)))


def checkout_layer_name(name_param, model_parameters, info_epoch=""):
    for name, param in model_parameters:
        if param.requires_grad:
            if name == name_param:
                print("DEBUG END REPLICATION:epoch (checkout_layers_tools) {} ".format(info_epoch), "name", name, param.data)


def pred_word_to_list(pred_word, special_symb_ls):
    index_special_ls = []

    pred_word = [pred_word]
    ind_pred_word = 0
    counter = 0
    while True:
        counter += 1
        index_special_ls = []
        _pred_word = pred_word[ind_pred_word]
        # Looking for all special character (we only look at the first one found)
        for special_symb in special_symb_ls:
            index_special_ls.append(_pred_word.find(special_symb))
        indexes = np.argsort(index_special_ls)
        index_special_char=-1
        # Getting the index and the character of the first special character if nothing we get -1
        for ind, a in enumerate(indexes):
            if index_special_ls[a] >= 0:
                special_symb = special_symb_ls[a]
                index_special_char = index_special_ls[a]
                break
            if ind > len(indexes):
                index_special_char = -1
                special_symb = ""
                break
        # if found a special character
        if (index_special_char) >= 0:
            starting_seq = [_pred_word[:index_special_char]] if index_special_char> 0 else []
            middle = [_pred_word[index_special_char:index_special_char + len(special_symb) ]]
            end_seq = [_pred_word[index_special_char + len(special_symb):]]
            if len(end_seq[0].strip()) == 0:
                end_seq = []
            _pred_word_ls = starting_seq + middle +end_seq
            pred_word[ind_pred_word] = _pred_word_ls[0]
            if len(_pred_word_ls) > 0:
                pred_word.extend(_pred_word_ls[1:])
            ind_pred_word += 1
            pdb.set_trace()
            if len(starting_seq) > 0:
                ind_pred_word += 1
        else:
            ind_pred_word += 1
        pdb.set_trace()
        if ind_pred_word >= len(pred_word):
            break

    new_word = []
    # transform the way we splitted in list of characters (including special ones)
    for word in pred_word:
        if word in special_symb_ls:
            new_word.append(word)
        else:
            new_word.extend(list(word))

    return new_word


def get_init_args_dir(init_args_dir):
    """
    to simplify reporting we allow three ways of providing init_args_dir
    :param init_args_dir:
    :return:
    """
    if os.path.isfile(init_args_dir):  # , "ERROR {} not found to reload checkpoint".format(init_args_dir)
        _dir = init_args_dir
    elif os.path.isfile(os.path.join(CHECKPOINT_BERT_DIR, init_args_dir)):
        printing("MODEL init {} not found as directory so using second template ", var=[init_args_dir], verbose=1,
                 verbose_level=1)
        _dir = os.path.join(CHECKPOINT_BERT_DIR, init_args_dir)
    else:
        printing("MODEL init {} not found as directory and as subdirectory so using third template template ",
                 var=[init_args_dir], verbose=1, verbose_level=1)
        match = re.match("(.*-model_[0-9]+).*", init_args_dir)
        assert match is not None, "ERROR : template {} not found in {}".format("([.*]-model_[0-9]+).*", init_args_dir)
        _dir = os.path.join(CHECKPOINT_BERT_DIR, match.group(1), init_args_dir + "-args.json")
        assert os.path.isfile(_dir), "ERROR : {} does not exist (based on param {}) ".format(_dir, init_args_dir)
    return _dir


def get_name_model_id_with_extra_name(epoch, _epoch, name_with_epoch, model_id):
    """
    if name_with_epoch we enrich model_id with epoch information
    :param epoch:
    :param _epoch:
    :param name_with_epoch:
    :param model_id:
    :return:
    """
    if not name_with_epoch:
        extra_name = ""
    else:
        extra_name = str(epoch) + "_ep_best" if _epoch == "best" else str(epoch)+"_ep"
        extra_name = "-" + extra_name
    model_id = model_id + extra_name
    return model_id


def write_args(dir, model_id, checkpoint_dir=None,
               hyperparameters=None,
               info_checkpoint=None, verbose=1):

    args_dir = os.path.join(dir, "{}-args.json".format(model_id))
    if os.path.isfile(args_dir):
        info = "updated"
        args = json.load(open(args_dir, "r"))
        args["checkpoint_dir"] = checkpoint_dir
        args["info_checkpoint"] = info_checkpoint
        json.dump(args, open(args_dir, "w"))
    else:
        assert hyperparameters is not None, "REPORT : args.json created for the first time : hyperparameters dic required "
        #assert info_checkpoint is None, "REPORT : args. created for the first time : no checkpoint yet "
        info = "new"
        json.dump(OrderedDict([("checkpoint_dir", checkpoint_dir),
                               ("hyperparameters", hyperparameters),
                               ("info_checkpoint", info_checkpoint)]), open(args_dir, "w"))
    printing("MODEL args.json {} written {} ".format(info, args_dir), verbose_level=1, verbose=verbose)
    return args_dir


def get_dataset_label(dataset_dir_ls, default):
    if dataset_dir_ls is None:
        return None

    if REPO_DATASET.get(dataset_dir_ls[0], None) is None:
        try:
            label = "|".join([get_code_data(path) for _, path in enumerate(dataset_dir_ls)])
        except:
            printing("REPORT : dataset name of directory {} not found as UD so using default ", var=[dataset_dir_ls], verbose=0, verbose_level=1)
            label = "|".join([REPO_DATASET.get(path, "{}_{}".format(default, i)) for i, path in enumerate(dataset_dir_ls)])
    else:
        label = "|".join([REPO_DATASET.get(path, "{}_{}".format(default, i)) for i, path in enumerate(dataset_dir_ls)])

    return label


def get_hyperparameters_dict(args, case, random_iterator_train, seed, verbose, model_id,
                             model_location, dict_path=None):

    hyperparameters = OrderedDict([("bert_model", args.bert_model), ("lr", args.lr),
                                   ("epochs", args.epochs),
                                   ("initialize_bpe_layer", args.initialize_bpe_layer),
                                   ("fine_tuning_strategy", args.fine_tuning_strategy),
                                   ("dropout_input_bpe", args.dropout_input_bpe),
                                   ("heuristic_ls", args.heuristic_ls),
                                   ("gold_error_detection", args.gold_error_detection),
                                   ("dropout_classifier", args.dropout_classifier if args.dropout_classifier is not None else "UNK"),
                                   ("dropout_bert", args.dropout_bert if args.dropout_bert is not None else "UNK"),
                                   ("tasks", args.tasks),
                                   ("masking_strategy", args.masking_strategy),
                                   ("portion_mask", args.portion_mask),
                                   ("init_args_dir", args.init_args_dir),
                                   ("norm_2_noise_training", args.norm_2_noise_training),
                                   ("random_iterator_train", random_iterator_train),
                                   ("aggregating_bert_layer_mode", args.aggregating_bert_layer_mode),
                                   ("tokenize_and_bpe", args.tokenize_and_bpe),
                                   ("seed", seed), ("case", case), ("bert_module", args.bert_module),
                                   ("freeze_layer_prefix_ls", args.freeze_parameters),
                                   ("layer_wise_attention", args.layer_wise_attention),
                                   ("append_n_mask", args.append_n_mask),
                                   ("multi_task_loss_ponderation", args.multi_task_loss_ponderation),
                                   ("multitask", args.multitask),
                                   ("low_memory_foot_print_batch_mode", args.low_memory_foot_print_batch_mode),
                                   ("graph_head_hidden_size_mlp_rel", args.graph_head_hidden_size_mlp_rel),
                                   ("graph_head_hidden_size_mlp_arc", args.graph_head_hidden_size_mlp_arc),
                                   ("ponderation_per_layer", args.ponderation_per_layer),
                                   ("norm_order_per_layer", args.norm_order_per_layer),
                                   ("weight_decay", args.weight_decay),
                                   ("penalize", args.penalize),
                                   ("hidden_dropout_prob", args.hidden_dropout_prob),
                                   ("schedule_lr", args.schedule_lr),
                                   ("n_steps_warmup",args.n_steps_warmup),
                                   ("random_init", args.random_init),
                                   ("dict_path", dict_path),
                                   ("model_id", model_id),
                                   ("optimizer", args.optimizer),
                                   ("model_location", model_location),
                                   ("shuffle_bpe_embedding", args.shuffle_bpe_embedding),
                                   ("test_mode_no_shuffle_embedding", args.test_mode_no_shuffle_embedding),
                                   ("not_load_params_ls", args.not_load_params_ls),
                                   ("report_gradient", args.report_gradient),
                                   ("prune_heads", args.prune_heads),
                                   ("hard_skip_attention_layers", ",".join(args.hard_skip_attention_layers) if len(args.hard_skip_attention_layers)>0 else "None"),
                                   ("hard_skip_all_layers", ",".join(args.hard_skip_all_layers) if len(args.hard_skip_all_layers)>0 else "None"),
                                   ("hard_skip_dense_layers", ",".join(args.hard_skip_dense_layers) if len(args.hard_skip_dense_layers)>0 else "None")
                                   ])

    printing("HYPERPARAMETERS {} ", var=[hyperparameters], verbose=verbose, verbose_level=1)
    printing("HYPERPARAMETERS KEYS {} ", var=[hyperparameters.keys()], verbose=verbose, verbose_level=1)
    return hyperparameters

