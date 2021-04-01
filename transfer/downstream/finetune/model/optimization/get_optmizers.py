

from transfer.downstream.finetune.env.imports import pdb, torch, re
from transfer.downstream.finetune.io_.logger import printing
import transfer.downstream.finetune.model.optimization.deep_learning_toolbox as dptx
from transfer.downstream.finetune.model.settings import AVAILABLE_BERT_FINE_TUNING_STRATEGY
#from training.bert_normalize.optimizer import WarmupLinearSchedule
from transfer.downstream.finetune.transformers.transformers.optimization import WarmupLinearSchedule


def apply_fine_tuning_strategy(fine_tuning_strategy, model, epoch, lr_init,optimizer_name,
                               betas=None, weight_decay=None, t_total=None, verbose=1):
    """
    get optimizers based on fine tuning strategies that might involve having several optimizers, and freezing some layers
    """
    try:
        assert fine_tuning_strategy in AVAILABLE_BERT_FINE_TUNING_STRATEGY, "{} not in {}".format(fine_tuning_strategy,
                                                                                              AVAILABLE_BERT_FINE_TUNING_STRATEGY)
    except:

        in_availabitlity = 0
        for avail in AVAILABLE_BERT_FINE_TUNING_STRATEGY:
            if fine_tuning_strategy.startswith(avail):
                in_availabitlity = 1
                break
        assert in_availabitlity == 1, "ERROR {} not found in Avail".format(fine_tuning_strategy)
    scheduler = None
    if fine_tuning_strategy in ["standart", "bert_out_first", "only_first_and_last", "freeze",
                                "word_embeddings_freeze",
                                "pos_embeddings_freeze", "embeddings_freeze",
                                "dense_freeze_all"] or \
            fine_tuning_strategy.startswith("attention") or fine_tuning_strategy.startswith("dense") or \
            fine_tuning_strategy.startswith("encoder") or fine_tuning_strategy.startswith("layer_specific"):
        
        assert isinstance(lr_init, float), "{} lr : type {}".format(lr_init, type(lr_init))
        optimizer = [dptx.get_optimizer(model.parameters(), lr=lr_init, betas=betas, weight_decay=weight_decay, optimizer=optimizer_name)]

        if optimizer_name == "AdamW":
            assert t_total is not None
            assert len(optimizer) == 1, "ERROR scheduler only supported when 1 optimizer "
            printing("OPTIMIZING warmup_steps:{} t_total:{}", var=[t_total / 10, t_total], verbose=verbose,
                     verbose_level=1)
            scheduler = WarmupLinearSchedule(optimizer[0], warmup_steps=t_total / 10, t_total=t_total)  # PyTorch scheduler

        printing("TRAINING : fine tuning strategy {} : learning rate constant {} betas {} weight_decay {}", var=[fine_tuning_strategy, lr_init, betas,weight_decay],
                 verbose_level=2, verbose=verbose)

    elif fine_tuning_strategy == "flexible_lr":
        assert isinstance(lr_init, dict), "lr_init should be dict in {}".format(fine_tuning_strategy)
        # sanity check

        assert optimizer_name in ["adam"], "ERROR only adam supporte in flexible_lr"

        optimizer = []
        n_all_layers = len([a for a, _ in model.named_parameters()])
        n_optim_layer = 0
        for pref, lr in lr_init.items():
            param_group = [param for name, param in model.named_parameters() if name.startswith(pref)]
            n_optim_layer += len(param_group)
            optimizer.append(dptx.get_optimizer(param_group, lr=lr, betas=betas, optimizer=optimizer_name))
        assert n_all_layers == n_optim_layer, \
            "ERROR : You are missing some layers in the optimization n_all {} n_optim {} ".format(n_all_layers, n_optim_layer)

        printing("TRAINING : fine tuning strategy {} : learning rate constant : {} betas {}",
                 var=[fine_tuning_strategy, lr_init, betas], verbose_level=1, verbose=verbose)

    matching = re.match(".*-([0-9]+,)+", fine_tuning_strategy)

    if matching is not None:
        ls_layer = eval("[" + fine_tuning_strategy.split("-")[1] + "]")
    if fine_tuning_strategy in ["bert_out_first", "freeze"]:
        info_add = ""
        if (epoch <= 1 and fine_tuning_strategy == "bert_out_first") or fine_tuning_strategy == "freeze":
            info_add = "not"
            freeze_layer_prefix_ls = "encoder"
            model = dptx.freeze_param(model, freeze_layer_prefix_ls, verbose=verbose)

        printing("TRAINING : fine tuning strategy {} : {} freezing bert for epoch {}"\
                 .format(fine_tuning_strategy, info_add, epoch), verbose_level=1, verbose=verbose)
    elif fine_tuning_strategy == "only_first_and_last":
        #optimizer = [torch.optim.Adam(model.parameters(), lr=lr_init, betas=betas, eps=1e-9)]
        model = dptx.freeze_param(model, freeze_layer_prefix_ls=None, not_freeze_layer_prefix_ls=["embeddings", "classifier"], verbose=verbose)
    elif fine_tuning_strategy == "word_embeddings_freeze":
        model = dptx.freeze_param(model, freeze_layer_prefix_ls=["encoder.embeddings.word_embeddings"], not_freeze_layer_prefix_ls=None, verbose=verbose)
    elif fine_tuning_strategy == "pos_embeddings_freeze":
        model = dptx.freeze_param(model, freeze_layer_prefix_ls=["encoder.embeddings.position_embeddings"], not_freeze_layer_prefix_ls=None, verbose=verbose)
    elif fine_tuning_strategy == "embeddings_freeze":
        model = dptx.freeze_param(model, freeze_layer_prefix_ls=["encoder.embeddings"], not_freeze_layer_prefix_ls=None, verbose=verbose)

    elif fine_tuning_strategy.startswith("encoder_freeze"):
        if matching is None:
            #pdb.set_trace()
            model = dptx.freeze_param(model,
                                      freeze_layer_prefix_ls=["encoder.encoder"],
                                      not_freeze_layer_prefix_ls=None, verbose=verbose, mode_regex=False)
        else:
            model = dptx.freeze_param(model,
                                      freeze_layer_prefix_ls=["encoder.encoder.layer.{}.[a-z].*".format(layer) for layer in ls_layer],
                                      not_freeze_layer_prefix_ls=None, verbose=verbose, mode_regex=True)

        #??model = dptx.freeze_param(model, freeze_layer_prefix_ls=["encoder.encoder.layer.([0-9]+).attention.*"],
        #                          not_freeze_layer_prefix_ls=None, verbose=verbose, mode_regex=True)
    elif fine_tuning_strategy.startswith("attention_freeze_all"):
        if matching is None:
            model = dptx.freeze_param(model, freeze_layer_prefix_ls=["encoder.encoder.layer.([0-9]+).attention.*"],
                                      not_freeze_layer_prefix_ls=None, verbose=verbose, mode_regex=True)
        else:
            model = dptx.freeze_param(model, freeze_layer_prefix_ls=["encoder.encoder.layer.{}.attention.*".format(layer) for layer in ls_layer],
                                      not_freeze_layer_prefix_ls=None, verbose=verbose, mode_regex=True)

    elif fine_tuning_strategy.startswith("dense_freeze_all"):

        if matching is None:
            model = dptx.freeze_param(model, freeze_layer_prefix_ls=["encoder.encoder.layer.([0-9]+).intermediate.*", "encoder.encoder.layer.([0-9]+).output.*"], not_freeze_layer_prefix_ls=None, verbose=verbose, mode_regex=True)
        else:
            model = dptx.freeze_param(model,
                                      freeze_layer_prefix_ls=["encoder.encoder.layer.{}.intermediate.*".format(layer) for layer in ls_layer]+["encoder.encoder.layer.{}.output.*".format(layer) for layer in ls_layer],
                                      not_freeze_layer_prefix_ls=None, verbose=verbose, mode_regex=True)

    elif fine_tuning_strategy.startswith("attention_freeze_ponderation"):
            if matching is None:
                model = dptx.freeze_param(model,
                                          freeze_layer_prefix_ls=["encoder.encoder.layer.([0-9]+).attention.self.query.*",
                                                                  "encoder.encoder.layer.([0-9]+).attention.self.key.*"],
                                          not_freeze_layer_prefix_ls=None, verbose=verbose, mode_regex=True)
            else:
                model = dptx.freeze_param(model,
                                          freeze_layer_prefix_ls=["encoder.encoder.layer.{}.attention.self.query.*".format(layer)
                                                                  for layer in ls_layer]+["encoder.encoder.layer.{}.attention.self.key.*".format(layer) for layer in ls_layer],
                                          not_freeze_layer_prefix_ls=None, verbose=verbose, mode_regex=True)
    elif fine_tuning_strategy.startswith("attention_unfreeze_ponderation_only"):
            if matching is None:
                model = dptx.freeze_param(model,
                                          freeze_layer_prefix_ls=None,
                                          not_freeze_layer_prefix_ls=["encoder.encoder.layer.([0-9]+).attention.self.query.*",
                                                                      "encoder.encoder.layer.([0-9]+).attention.self.key.*", "head.*"],
                                          verbose=verbose, mode_regex=True)
            else:
                model = dptx.freeze_param(model,
                                          not_freeze_layer_prefix_ls=["encoder.encoder.layer.{}.attention.self.query.*".format(layer)
                                                                  for layer in ls_layer]+["encoder.encoder.layer.{}.attention.self.key.*".format(layer) for layer in ls_layer]+["head.*"],
                                          freeze_layer_prefix_ls=None, verbose=verbose, mode_regex=True)

    elif fine_tuning_strategy.startswith("layer_specific"):
        assert fine_tuning_strategy.count("-") == 2, "ERROR special char - should be 2 but is {} in {}".format(fine_tuning_strategy.count("-"), fine_tuning_strategy)
        assert fine_tuning_strategy[-1] == ",", "ERROR should end up with , but {}".format(fine_tuning_strategy)
        mode = fine_tuning_strategy.split("-")[1]
        to_match = fine_tuning_strategy.split("-")[2]
        assert len(to_match) > 0, "ERROR fine_tuning_strategy {} not correct pattern".format(fine_tuning_strategy)
        assert mode in ["not_freeze", "freeze"]
        regex_ls = to_match.split(",")[:-1]
        assert len(regex_ls)>0, "ERROR {}".format(regex_ls)
        printing("TRAINING : Un/Freezing with layer_specific mode:{} regex:{}",var=[mode,regex_ls], verbose=verbose, verbose_level=1)
        if mode == "not_freeze":
            not_freeze_layer_prefix_ls = regex_ls
            not_freeze_layer_prefix_ls = not_freeze_layer_prefix_ls + ["head.*"]
            freeze_layer_prefix_ls = None
        elif mode == "freeze":
            not_freeze_layer_prefix_ls = None
            freeze_layer_prefix_ls = regex_ls
        else:
            raise(Exception("mode {}".format(mode)))
        model = dptx.freeze_param(model,
                                  not_freeze_layer_prefix_ls=not_freeze_layer_prefix_ls,
                                  freeze_layer_prefix_ls=freeze_layer_prefix_ls, verbose=verbose, mode_regex=True)

        #
    return model, optimizer, scheduler
