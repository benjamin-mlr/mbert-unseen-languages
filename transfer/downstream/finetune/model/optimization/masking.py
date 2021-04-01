from transfer.downstream.finetune.env.imports import pdb, OrderedDict, torch,  np
from transfer.downstream.finetune.io_.logger import printing
from transfer.downstream.finetune.model.optimization.deep_learning_toolbox import dropout_input_tensor


def dropout_mlm(input_tokens_tensor, mask_token_index, sep_token_index, cls_token_index, pad_index, use_gpu,
                dropout_mask=0.15, dropout_random_bpe_of_masked=0.5, vocab_len=None):

    # dropout_mask% of the time we replace the bpe token by the MASK (except CLS,SEP and pad that will be kept untouched
    input_tokens_tensor, mask_dropout, apply_dropout = dropout_input_tensor(input_tokens_tensor, mask_token_index,
                                                                            sep_token_index=sep_token_index,
                                                                            cls_token_index=cls_token_index, pad_index=pad_index,
                                                                            dropout=dropout_mask,
                                                                            apply_dropout=None,
                                                                            applied_dropout_rate=0.8)

    # if dropout has not been aplied (20% the time) : then  dropout_random_bpe_of_masked we permute
    if not apply_dropout and dropout_random_bpe_of_masked > 0:

        assert vocab_len is not None

        random_bpe_instead = np.random.random() < 0.5

        if random_bpe_instead:

            permute = (torch.randperm(torch.tensor(vocab_len - 2))[:len(input_tokens_tensor[mask_dropout == 0])] + 1)
            # if we get sep, cls or pad we make sure we don't permute with them them
            permute[permute == sep_token_index] = sep_token_index + 10
            permute[permute == mask_token_index] = mask_token_index + 10
            permute[permute == pad_index] = 53

            #if use_gpu:
            #    permute = permute.cuda()

            input_tokens_tensor[mask_dropout == 0] = permute

    return input_tokens_tensor


def focused_masking(masking_strategy, input_tokens_tensor, output_tokens_tensor_aligned, dropout_input_bpe, mask_token_index, sep_token_index,
                    use_gpu, epoch, n_epoch, portion_mask, input_mask, tokenizer,
                    verbose):

    if masking_strategy in ["mlm", "mlm_need_norm"]:

        dropout = 0.15
        assert dropout_input_bpe == 0., "in args.masking_strategy mlm we hardcoded dropout to 0.2 {}".format(
            dropout)
        # standart standart_mlm means : standart MLM prediction
        standart_mlm = True
        # unmask_loss : bool do we unmask other loss than only the MASKed tokens
        unmask_loss = portion_mask
        if masking_strategy == "mlm_need_norm":
            # if mlm_need_norm strategy : in args.portion_mask% of the time we learn as a standart mlm the rest
            # of the time we do the same but only on need_norm tokens (masking them)
            standart_mlm = np.random.random() < portion_mask
            # we force unmask loss to 0
            unmask_loss = 0
        if standart_mlm:
            # standart mlm
            input_tokens_tensor, mask_dropout, dropout_applied = dropout_input_tensor(input_tokens_tensor, mask_token_index, sep_token_index=sep_token_index, applied_dropout_rate=0.8, dropout=dropout)
        elif masking_strategy == "mlm_need_norm" and not standart_mlm:
            # todo : factorize
            feeding_the_model_with_label = output_tokens_tensor_aligned.clone()
            # we only learn on tokens that are different from gold
            feeding_the_model_with_label[input_tokens_tensor == output_tokens_tensor_aligned] = -1
            if np.random.random() < 0.85:
                # 80% of the time we mask the tokens as standart mlm
                input_tokens_tensor[input_tokens_tensor != output_tokens_tensor_aligned] = mask_token_index
            else:
                # within the 15% rest : 50% of the time we replace by random 50% we keep
                if np.random.random() < 0.5:
                    permute = (torch.randperm(torch.tensor(len(tokenizer.vocab) -2))
                               [:len(input_tokens_tensor[input_tokens_tensor != output_tokens_tensor_aligned]) ] +1)
                    permute[permute == sep_token_index] = sep_token_index + 10
                    permute[permute == mask_token_index] = mask_token_index + 10
                    permute[permute == 0] = 53
                    if use_gpu:
                        permute = permute.cuda()
                    input_tokens_tensor[input_tokens_tensor != output_tokens_tensor_aligned] = permute
            mask_dropout = (input_tokens_tensor == output_tokens_tensor_aligned)

        if standart_mlm and not dropout_applied:
            random_bpe_instead = np.random.random() < 0.5
            if random_bpe_instead:
                permute = (torch.randperm(torch.tensor(len(tokenizer.vocab) -2))
                           [:len(input_tokens_tensor[mask_dropout == 0])] +1)
                permute[permute == sep_token_index] = sep_token_index +10
                permute[permute == mask_token_index] = mask_token_index + 10
                permute[permute == 0] = 53
                if use_gpu:
                    permute = permute.cuda()

                input_tokens_tensor[mask_dropout == 0] = permute

        if unmask_loss:
            print("WARNING : unmaskloss is {}  : 0 means only optimizing on the MASK  , > 0 means optimizes "
                  "also on some other sampled based on dropout_adapted)".format(unmask_loss))
            power = 3
            capped = 0.5
            dropout_adated = min(((epoch + 1) / n_epoch) ** power, capped)
            printing("LABEL NOT MASKING {}/1 of gold labels with power {} and capped {}".format(dropout_adated, power, capped), verbose=verbose, verbose_level=2)
            _, mask_losses = dropout_input_tensor(input_tokens_tensor, mask_token_index,
                                                  sep_token_index=sep_token_index,
                                                  apply_dropout=False,
                                                  dropout=dropout_adated)
            # we backpropagate only on tokens that receive a mask (MLM objective) +
            #  some extra ones tgat we control with dropout_adated
            mask_loss = mask_dropout *mask_losses
        else:
            mask_loss = mask_dropout
        feeding_the_model_with_label = output_tokens_tensor_aligned.clone()
        feeding_the_model_with_label[mask_loss != 0] = -1
        # hald the time we actually mask those tokens otherwise we predict
    elif masking_strategy in ["norm_mask", "norm_mask_variable"] :
        if masking_strategy == "norm_mask_variable":
            # args.portion_mask = min(((epoch + 1) / n_epoch), 0.6)
            portion_mask = 1 - (epoch + 1) / n_epoch  # , 0.6))
        mask_normed = np.random.random() < portion_mask
        feeding_the_model_with_label = output_tokens_tensor_aligned.clone()
        if mask_normed:
            print("MASKING NORMED in mode {} portion mask {}".format(masking_strategy,
                                                                     portion_mask))
            feeding_the_model_with_label[input_tokens_tensor == output_tokens_tensor_aligned] = -1
            if np.random.random() < 0.5:
                # half the time we mask not to make the model only normalizing
                input_tokens_tensor[input_tokens_tensor != output_tokens_tensor_aligned] = mask_token_index
    else:
        feeding_the_model_with_label = output_tokens_tensor_aligned.clone()
        # TODO -- handle loggin of output_tokens_tensor_aligned everywhere
        printing("MASK mask:{} \nMASK input:{} \nMASK output:{}",
                 var=[input_mask, input_tokens_tensor, feeding_the_model_with_label],
                 verbose_level="raw_data", verbose=verbose)

    return input_tokens_tensor, feeding_the_model_with_label

