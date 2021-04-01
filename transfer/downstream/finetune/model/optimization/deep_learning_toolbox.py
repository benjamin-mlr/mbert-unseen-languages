
from transfer.downstream.finetune.env.imports import torch, np, re, pdb
from transfer.downstream.finetune.model.settings import AVAILABLE_OPTIMIZER
from transfer.downstream.finetune.io_.logger import printing

#from transfer.downstream.finetune.transformers.transformers.optimization import AdamW
from transformers import AdamW


def get_optimizer(parameters, lr, optimizer="adam", betas=None, weight_decay=None, verbose=1):

    assert optimizer in AVAILABLE_OPTIMIZER, "ERROR optimizers supported are {} ".format(AVAILABLE_OPTIMIZER)

    if optimizer == "adam":
        if betas is None:
            # betas = (0.9, 0.9)
            print("DEFAULT betas:", betas)
        if weight_decay is None:
            weight_decay = 0
        opt = torch.optim.Adam(parameters, lr=lr, betas=betas, eps=1e-9, weight_decay=weight_decay)

    elif optimizer == "SGD":
        assert betas is None, "ERROR "
        opt = torch.optim.SGD(parameters, lr=lr)

    elif optimizer == "bahdanu-adadelta":
        assert betas is None, "ERROR betas not supported for optimizer {}".format(optimizer)
        opt = torch.optim.Adadelta(parameters, eps=10e-6, rho=0.95)

    elif optimizer == "AdamW":
        opt = AdamW(parameters, lr=lr, weight_decay=weight_decay)

    printing("TRAINING : optimizer {} has been reloaded with lr {} betas {} ", var=[optimizer, lr, betas], verbose=verbose, verbose_level=2)
    return opt


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_cumulated_list(sent_len):
    sent_len_cumulated = [0]
    cumu = 0
    for len_sent in sent_len:
        cumu += int(len_sent)
        sent_len_cumulated.append(cumu)
    return sent_len_cumulated


def freeze_param(model, freeze_layer_prefix_ls=None, not_freeze_layer_prefix_ls=None,verbose=1, mode_regex=False):
    freezing_layer = 0
    to_freeze_ls = []
    if not_freeze_layer_prefix_ls is None:
        not_freeze_layer_prefix_ls = []
    if freeze_layer_prefix_ls is None:
        freeze_layer_prefix_ls = []
    for name, param in model.named_parameters():
        if len(freeze_layer_prefix_ls)>0:
            for prefix in freeze_layer_prefix_ls:
                if name.startswith(prefix) and not mode_regex:
                    param.requires_grad = False
                    freezing_layer += 1
                    printing("TRAINING : freezing {} parameter ", var=[name], verbose=verbose, verbose_level=1)
                if mode_regex and re.match(prefix, name) is not None:
                    param.requires_grad = False
                    freezing_layer += 1
                    printing("TRAINING (regex match): freezing {} parameter ", var=[name], verbose=verbose, verbose_level=1)

        if len(not_freeze_layer_prefix_ls) > 0:
            to_freeze = 0
            for prefix in not_freeze_layer_prefix_ls:
                # if not name.startswith(prefix) and not mode_regex:
                #    to_freeze += 1
                # elif re.match(prefix, name) is  None and mode_regex:
                #    to_freeze += 1
                # if not to_freeze == len(not_freeze_layer_prefix_ls):
                #    pdb.set_trace()
                #    param.requires_grad = False
                #    freezing_layer += 1
                if (not mode_regex and name.startswith(prefix) is None) or (mode_regex and re.match(prefix, name) is None):
                    to_freeze += 1
            if to_freeze == len(not_freeze_layer_prefix_ls):
                to_freeze_ls.append(name)
                #elif mode_regex and re.match(prefix, name) is None:
                #    pdb.set_trace()
                #    to_freeze_ls.append(name)

    #printing("TRAINING : based on {} not to freeze found {} to freeze", var=[not_freeze_layer_prefix_ls, to_freeze_ls],
    #         verbose=verbose, verbose_level=1)

    if len(to_freeze_ls) > 0:
        for name, param in model.named_parameters():
            if name in to_freeze_ls:
                param.requires_grad = False
                freezing_layer += 1
                printing("TRAINING {}: freezing {} parameter ", var=["(regex match)" if mode_regex else "", name], verbose=verbose, verbose_level=1)
    printing("TRAINING : freezing {} layers : {} prefix , not freezing {} ",
             var=[freezing_layer, freeze_layer_prefix_ls, not_freeze_layer_prefix_ls],
             verbose=verbose, verbose_level=1)

    assert freezing_layer > 0, "ERROR : did not fine any layers starting with {}".format(prefix)

    return model


def dropout_input_tensor(input_tokens_tensor, mask_token_index, sep_token_index, dropout, cls_token_index=None, pad_index=None,
                         apply_dropout=None, applied_dropout_rate=None):
    if apply_dropout is None:
        assert applied_dropout_rate is not None
        apply_dropout = np.random.random() < applied_dropout_rate
    droping_multiplier_input_tokens_tensor = torch.zeros_like(input_tokens_tensor).bernoulli_(1 - dropout)
    droping_multiplier_input_tokens_tensor[input_tokens_tensor == sep_token_index] = 1
    if cls_token_index is not None:
        droping_multiplier_input_tokens_tensor[input_tokens_tensor == cls_token_index] = 1
        droping_multiplier_input_tokens_tensor[input_tokens_tensor == pad_index] = 1
    # we mask all the tokens which got droping_multiplier_input_tokens_tensor 0
    if apply_dropout:
        input_tokens_tensor[droping_multiplier_input_tokens_tensor == 0] = mask_token_index
    return input_tokens_tensor, droping_multiplier_input_tokens_tensor, apply_dropout
