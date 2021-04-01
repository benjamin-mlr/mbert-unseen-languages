
from transfer.downstream.finetune.env.imports import pdb, torch, OrderedDict


def get_batch_per_layer_head(attention, layer_head_att_batch=None, head=True):
    """

    :param attention:
    :param layer_head_att_batch:  if not None, will append to it as a list of tensor
    :return:
    """
    if layer_head_att_batch is None:
        layer_head_att_batch = OrderedDict()
    for i_layer in range(len(attention)):
        if head:
            for i_head in range(attention[0].size(1)):
                if f"layer_{i_layer}-head_{i_head}" not in layer_head_att_batch:
                    layer_head_att_batch[f"layer_{i_layer}-head_{i_head}"] = []
                #pdb.set_trace()
                layer_head_att_batch[f"layer_{i_layer}-head_{i_head}"].append(attention[i_layer][:, i_head].detach())
        else:
            if f"layer_{i_layer}" not in layer_head_att_batch:
                layer_head_att_batch[f"layer_{i_layer}"] = []
            layer_head_att_batch[f"layer_{i_layer}"].append(attention[i_layer][:].detach())
    return layer_head_att_batch


def get_hidden_representation(data, model, tokenizer, special_start="[CLS]",
                                  special_end="[SEP]", pad="[PAD]", max_len=100,
                                  pad_below_max_len=False, output_dic=True):
    """
    get hidden representation (ie contetualized vector at the word level : add it as list or padded tensor : output[attention|layer|layer_head]["layer_x"] list or tensor)
    :param data: list of raw text
    :param pad: will add padding below max_len
    :return: output a dictionary (if output_dic) or a tensor (if not output_dic) : of contextualized representation at the word level per layer/layer_head
    """
    model.eval()
    special_start = tokenizer.bos_token
    special_end = tokenizer.eos_token
    if special_start is None or special_end is None:
        special_start="[CLS]"
        special_end="[SEP]"

    layer_head_att_tensor_dic = OrderedDict()
    layer_hidden_state_tensor_dic = OrderedDict()
    layer_head_hidden_state_tensor_dic = OrderedDict()
    layer_head_att_batch_dic = OrderedDict()
    layer_head_hidden_state_dic = OrderedDict()
    layer_hidden_state_dic = OrderedDict()
    print(f"Getting hidden representation : adding special char start:{special_start} end:{special_end}")
    for seq in data:
        seq = special_start+" "+seq+" "+special_end
        tokenized = tokenizer.encode(seq)
        if len(tokenized) >= max_len:
            tokenized = tokenized[:max_len-1]
            tokenized += tokenizer.encode(special_end)
        mask = [1 for _ in range(len(tokenized))]
        real_len = len(tokenized)
        if pad_below_max_len:
            if len(tokenized) < max_len:
                for _ in range(max_len - len(tokenized) ):
                    tokenized += tokenizer.encode(pad)
                    mask.append(0)
            assert len(tokenized) == max_len
        assert len(tokenized) <= max_len+2
        

        encoded = torch.tensor(tokenized).unsqueeze(0)
        inputs = OrderedDict([("wordpieces_inputs_words", encoded)])
        attention_mask = OrderedDict([("wordpieces_inputs_words", torch.tensor(mask).unsqueeze(0))])
        assert real_len
        if torch.cuda.is_available():
            inputs["wordpieces_inputs_words"] = inputs["wordpieces_inputs_words"].cuda()
            
            attention_mask["wordpieces_inputs_words"] = attention_mask["wordpieces_inputs_words"].cuda()
        model_output = model(input_ids_dict=inputs, attention_mask=attention_mask)
        #pdb.set_trace()
        #logits = model_output[0]

        # getting the output index based on what we are asking the model
        hidden_state_per_layer_index = 2 if model.config.output_hidden_states else False
        attention_index_original_index = 3-int(not hidden_state_per_layer_index) if model.config.output_attentions else False
        hidden_state_per_layer_per_head_index = False#4-int(not attention_index_original_index) if model.config.output_hidden_states_per_head else False
        # getting the output
        hidden_state_per_layer = model_output[hidden_state_per_layer_index] if hidden_state_per_layer_index else None
        attention = model_output[attention_index_original_index] if attention_index_original_index else None
        hidden_state_per_layer_per_head = model_output[hidden_state_per_layer_per_head_index] if hidden_state_per_layer_per_head_index else None


        # checking that we got the correct output
        try:
            if attention is not None:
                assert len(attention) == 12, "ERROR attenttion"
                assert attention[0].size()[-1] == attention[0].size()[-2], "ERROR attenttion"
            if hidden_state_per_layer is not None:
                assert len(hidden_state_per_layer) == 12+1, "ERROR hidden state"
                assert hidden_state_per_layer[0].size()[-1] == 768, "ERROR hidden state"
            if hidden_state_per_layer_per_head is not None:
                assert len(hidden_state_per_layer_per_head) == 12, "ERROR hidden state per layer"
                assert hidden_state_per_layer_per_head[0].size()[1] == 12 and hidden_state_per_layer_per_head[0].size()[-1] == 64, "ERROR hidden state per layer"
        except Exception as e:
            raise(Exception(e))

        # concat as a batch per layer/layer_head
        if hidden_state_per_layer is not None:
            layer_hidden_state_dic = get_batch_per_layer_head(hidden_state_per_layer, layer_hidden_state_dic, head=False)
        if attention is not None:
            layer_head_att_batch_dic = get_batch_per_layer_head(attention, layer_head_att_batch_dic)
        if hidden_state_per_layer_per_head is not None:
            layer_head_hidden_state_dic = get_batch_per_layer_head(hidden_state_per_layer_per_head, layer_head_hidden_state_dic)

    output = ()
    if output_dic:
        if len(layer_hidden_state_dic) > 0:
            output = output + (layer_hidden_state_dic,)
        if len(layer_head_att_batch_dic) > 0:
            output = output + (layer_head_att_batch_dic,)
        if len(layer_head_hidden_state_dic)>0:
            output = output + (layer_head_hidden_state_dic, )
    else:
        # concatanate in a tensor
        # should have padding on !
        assert pad_below_max_len
        if len(layer_hidden_state_dic) > 0:
            for key in layer_hidden_state_dic:
                layer_hidden_state_tensor_dic[key] = torch.cat(layer_hidden_state_dic[key], 0)
            output = output + (layer_hidden_state_tensor_dic, )
        if len(layer_head_att_batch_dic) > 0:
            for key in layer_head_att_batch_dic:
                layer_head_att_tensor_dic[key] = torch.cat(layer_head_att_batch_dic[key], 0)
            output = output + (layer_head_att_tensor_dic,)
        if len(layer_head_hidden_state_dic) > 0:
            for key in layer_head_hidden_state_dic:
                layer_head_hidden_state_tensor_dic[key] = torch.cat(layer_head_hidden_state_dic[key], 0)
            output = output + (layer_head_hidden_state_tensor_dic,)

    return output

