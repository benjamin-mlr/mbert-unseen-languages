from transfer.downstream.finetune.env.imports import np, torch, time,  OrderedDict, Variable, pdb
from transfer.downstream.finetune.io_.logger import printing



def subsequent_mask(size):
    " Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


class MaskBatch(object):
    """
    """
    def __init__(self, input_seq, output_seq,
                 raw_input=None, raw_output=None, types=None, heads=None,
                 output_word=None, pos=None, input_word=None, edit=None,
                 wordpieces_words=None, wordpieces_raw_aligned_with_words=None, wordpieces_inputs_raw_tokens=None, is_mwe_label=None,
                 n_masks_to_app_in_raw_label=None,
                 all_indexes=None,
                 wordpiece_normalization=None, ind_wordpiece_normalization_alignement_index=None,
                 wordpiece_normalization_target_aligned_with_word=None,
                 ind_wordpiece_normalization_target_aligned_with_word_index=None,
                 wordpiece_words_src_aligned_with_norm=None, ind_wordpiece_words_src_aligned_with_norm_index=None,
                 n_masks_for_norm=None, to_norm_np=None,
                 ind_wordpieces_words_alignement_index=None, ind_wordpieces_raw_aligned_alignement_index=None, ind_wordpieces_inputs_raw_tokens_alignement_index=None,
                 ):

        self.raw_input = raw_input
        self.raw_output = raw_output

        self.input_seq = input_seq
        self.input_word = input_word

        #self.mwe_prediction = wordpieces_words
        self.wordpieces_inputs_words = wordpieces_words
        self.wordpieces_raw_aligned_with_words = wordpieces_raw_aligned_with_words
        self.wordpieces_inputs_raw_tokens = wordpieces_inputs_raw_tokens

        self.mwe_detection = is_mwe_label
        self.n_masks_mwe = n_masks_to_app_in_raw_label
        # NB : the convention is that the alignement tensor is named as the original tensor with _alignement
        # PB !!
        self.all_indexes = all_indexes
        if len(self.all_indexes.shape) == 1:
            self.all_indexes = np.expand_dims(self.all_indexes, axis=0)
        self.wordpieces_inputs_words_alignement = ind_wordpieces_words_alignement_index
        # resolving single sample batch
        if self.wordpieces_inputs_words_alignement is not None and len(self.wordpieces_inputs_words_alignement.shape) == 1:
            self.wordpieces_inputs_words_alignement = np.expand_dims(self.wordpieces_inputs_words_alignement,axis=0)
        self.wordpieces_raw_aligned_with_words_alignement = ind_wordpieces_raw_aligned_alignement_index
        if self.wordpieces_raw_aligned_with_words_alignement is not None and len(self.wordpieces_raw_aligned_with_words_alignement.shape) == 1:
            self.wordpieces_raw_aligned_with_words_alignement = np.expand_dims(self.wordpieces_raw_aligned_with_words_alignement, axis=0)
        self.wordpieces_inputs_raw_tokens_alignement = ind_wordpieces_inputs_raw_tokens_alignement_index
        if self.wordpieces_inputs_raw_tokens_alignement is not None and len(self.wordpieces_inputs_raw_tokens_alignement.shape) == 1:
            self.wordpieces_inputs_raw_tokens_alignement = np.expand_dims(self.wordpieces_inputs_raw_tokens_alignement,axis=0)

        # NB : the attributes should be aligned with the task_settings label field
        self.pos = pos
        self.types = types
        self.heads = heads

        self.wordpiece_normalization = wordpiece_normalization
        self.wordpiece_normalization_alignement = ind_wordpiece_normalization_alignement_index

        self.wordpiece_normalization_target_aligned_with_word = wordpiece_normalization_target_aligned_with_word
        self.wordpiece_normalization_target_aligned_with_word_alignement = ind_wordpiece_normalization_target_aligned_with_word_index

        self.wordpiece_words_src_aligned_with_norm = wordpiece_words_src_aligned_with_norm
        self.wordpiece_words_src_aligned_with_norm_alignement =ind_wordpiece_words_src_aligned_with_norm_index

        if self.wordpiece_words_src_aligned_with_norm_alignement is not None and len(self.wordpiece_words_src_aligned_with_norm_alignement.shape) == 1:
            self.wordpiece_words_src_aligned_with_norm_alignement = np.expand_dims(self.wordpiece_words_src_aligned_with_norm_alignement, axis=0)
        if self.wordpiece_normalization_target_aligned_with_word_alignement is not None and len(self.wordpiece_normalization_target_aligned_with_word_alignement.shape) == 1:
            self.wordpiece_normalization_target_aligned_with_word_alignement = np.expand_dims(self.wordpiece_normalization_target_aligned_with_word_alignement, axis=0)

        self.n_masks_for_norm = n_masks_for_norm
        self.to_norm_np = to_norm_np
        # unsqueeze add 1 dim between batch and word len ##- ?   ##- for commenting on context implementaiton
        start = time.time()



    @staticmethod
    def make_mask(output_seq, padding):
        "create a mask to hide paddding and future work"
        mask = (output_seq != padding).unsqueeze(-2)
        mask = mask & Variable(subsequent_mask(output_seq.size(-1)).type_as(mask.data))
        return mask


# test

if __name__=="__main__":

    plt.figure(figsize=(5, 5))
    plt.imshow(subsequent_mask(20)[0])
    #plt.show()
    data_out = torch.cat((torch.empty(1, 4, dtype=torch.long).random_(2, 5), torch.ones(1, 4, dtype=torch.long)),  dim=1)
    data_out = torch.cat((data_out, data_out), dim=0)
    data_in = torch.cat((torch.empty(1, 4, dtype=torch.long).random_(2, 4), torch.zeros(1, 3, dtype=torch.long)), dim=1)
    data_in[:, 0] = 2
    data_out[:, 0] = 2
    #data_in = torch.cat((data_in, data_in), dim=0)
    #data_out = data_out.unsqueeze(0)
    #data_in = data_in.unsqueeze(0)
    print("DATA IN {} {} ".format(data_in, data_in.size()))
    print("DATA OUT {} {} ".format(data_out, data_out.size()))
    batch = MaskBatch(data_in, data_out, pad=1, verbose=5)
    print("INPUT MASK {} , output mask {} ".format(batch.input_seq_mask, batch.output_mask))

    # NB : sequence cannot be padded on the middle (we'll cut to the first padded sequence )