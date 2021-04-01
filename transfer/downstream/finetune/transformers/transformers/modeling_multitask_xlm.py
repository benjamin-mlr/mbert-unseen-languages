
from transfer.downstream.finetune.env.imports import nn,  OrderedDict, re, pdb, CrossEntropyLoss
from transfer.downstream.finetune.model.settings import TASKS_PARAMETER, LABEL_PARAMETER
from transfer.downstream.finetune.model.constants import PAD_ID_LOSS_STANDART
from transfer.downstream.finetune.env.gpu_tools.gpu_info import printout_allocated_gpu_memory
from transfer.downstream.finetune.model.tools import get_key_name_num_label
from transfer.downstream.finetune.model.architecture.parser_modules.mlp import MLP
from transfer.downstream.finetune.model.architecture.parser_modules.biaffine import Biaffine
from .modeling_bert import BertPreTrainedModel, BertOnlyMLMHead
from .modeling_xlm import XLMPreTrainedModel, XLMModel
# only supported for XLM : exact copy of modeling_multitask (just renaming class and CLASS_PRETRAINED_MODEL set to XLMPreTrainedModel)

CLASS_PRETRAINED_MODEL = XLMPreTrainedModel


class BertTokenHead(nn.Module):
    def __init__(self, config, num_labels):
        super(BertTokenHead, self).__init__()
        dropout_classifier = config.dropout_classifier
        self.num_labels = num_labels
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.dropout = nn.Dropout(dropout_classifier) if dropout_classifier is not None else None
        #self.apply(self.init_bert_weights)

    def forward(self, x, head_mask=None):
        assert head_mask is None, "ERROR : not need of active logits only : handled in the loss for training " \
                                       "attention_mask"
        if self.dropout is not None:
            x = self.dropout(x)
        logits = self.classifier(x)
        # Only keep active parts of the loss
        # NB : the , is mandatory !
        return logits,


class BertGraphHead(nn.Module):
    # the MLP layers
    def __init__(self, config, num_labels=None):
        super(BertGraphHead, self).__init__()

        n_mlp_arc = config.graph_head_hidden_size_mlp_arc if config.graph_head_hidden_size_mlp_arc is not None else 100
        n_mlp_rel = config.graph_head_hidden_size_mlp_rel if config.graph_head_hidden_size_mlp_rel is not None else 100
        
        print("MODEL : n_mlp_arc {} n_mlp_rel {} ".format(n_mlp_arc, n_mlp_rel))
        n_rels = num_labels
        pad_index = 1
        unk_index = 0

        self.mlp_arc_h = MLP(n_in=config.hidden_size,
                             n_hidden=n_mlp_arc,
                             dropout=config.dropout_classifier if config.dropout_classifier is not None else 0.1)
        self.mlp_arc_d = MLP(n_in=config.hidden_size,
                             n_hidden=n_mlp_arc,
                             dropout=config.dropout_classifier if config.dropout_classifier is not None else 0.1)
        self.mlp_rel_h = MLP(n_in=config.hidden_size,
                             n_hidden=n_mlp_rel,
                             dropout=config.dropout_classifier if config.dropout_classifier is not None else 0.1)
        self.mlp_rel_d = MLP(n_in=config.hidden_size,
                             n_hidden=n_mlp_rel,
                             dropout=config.dropout_classifier if config.dropout_classifier is not None else 0.1)

        # the Biaffine layers
        self.arc_attn = Biaffine(n_in=n_mlp_arc,
                                 bias_x=True,
                                 bias_y=False)
        self.rel_attn = Biaffine(n_in=n_mlp_rel,
                                 n_out=n_rels,
                                 bias_x=True,
                                 bias_y=True)

        self.pad_index = pad_index
        self.unk_index = unk_index

    def forward(self, x, head_mask=None):
        # apply MLPs to the BiLSTM output states
        arc_h = self.mlp_arc_h(x)

        arc_d = self.mlp_arc_d(x)

        rel_h = self.mlp_rel_h(x)

        rel_d = self.mlp_rel_d(x)

        # get arc and rel scores from the bilinear attention
        # [batch_size, seq_len, seq_len]
        s_heads = self.arc_attn(arc_d, arc_h)
        # [batch_size, seq_len, seq_len, n_rels]
        s_labels = self.rel_attn(rel_d, rel_h).permute(0, 2, 3, 1)

        if head_mask is not None:
            # NB ? : is it necessary : as we only keep
            # set the scores that exceed the length of each sentence to -inf
            head_mask = head_mask.byte()
            s_heads.masked_fill_(~head_mask.unsqueeze(1), float('-inf'))

        return s_heads, s_labels



class BertMultiTaskXLM(CLASS_PRETRAINED_MODEL):
    """
    BERT model which can call any other modules
    """
    def __init__(self, config, **kwargs):# tasks, num_labels_per_task, mask_id, encoder_class):
        super(BertMultiTaskXLM, self).__init__(config)
        # encoder_class only BertModel or RobertaModel
        # some arguments specific to BertMultiTask could be passed in config or in kwargs
        encoder_class = kwargs["encoder"]
        tasks = kwargs["tasks"] if "tasks" in kwargs else config.tasks
        num_labels_per_task = kwargs["num_labels_per_task"] if "num_labels_per_task" in kwargs else config.tasks
        mask_id = kwargs["mask_id"] if "mask_id" in kwargs else config.tasks
        config.dropout_classifier = kwargs.get("dropout_classifier", 0.1)

        if "parsing" in tasks:
            config.graph_head_hidden_size_mlp_arc = kwargs.get("graph_head_hidden_size_mlp_arc", 500)
            config.graph_head_hidden_size_mlp_rel = kwargs.get("graph_head_hidden_size_mlp_rel", 200)

        self.encoder = encoder_class(config)
        print("BertMultitask instantiated with {} encoder".format(self.encoder.__class__.__name__))
        self.config = config
        assert isinstance(num_labels_per_task, dict)
        assert isinstance(tasks, list) and len(tasks) >= 1, "config.tasks should be a list of len >=1"
        self.head = nn.ModuleDict()
        self.mask_index_bert = mask_id
        self.tasks = tasks #
        self.tasks_available = tasks # all tasks available in the model (not only the one we want to use at a given run (self.tasks))
        self.task_parameters = TASKS_PARAMETER
        self.layer_wise_attention = None
        self.labels_supported = [label for task in tasks for label in self.task_parameters[task]["label"]]
        self.sanity_checking_num_labels_per_task(num_labels_per_task, tasks, self.task_parameters)
        self.num_labels_dic = num_labels_per_task

        for task in TASKS_PARAMETER:
            if task in tasks:
                num_label = get_key_name_num_label(task, self.task_parameters)
                if not self.task_parameters[task]["num_labels_mandatory"]:
                    # in this case we need to define and load MLM head of the model
                    self.head[task] = eval(self.task_parameters[task]["head"])(config)#, self.encoder.embeddings.word_embeddings.weight)
                else:
                    self.head[task] = eval(self.task_parameters[task]["head"])(config, num_labels=self.num_labels_dic[num_label])
            else:
                # we define empty heads for downstream use
                self.head[task] = None

    def forward(self, input_ids_dict, token_type_ids=None, attention_mask=None, labels=None, head_masks=None):
        if labels is None:
            labels = OrderedDict()
        if head_masks is None:
            head_masks = OrderedDict()
        sequence_output_dict = OrderedDict()
        logits_dict = OrderedDict()
        loss_dict = OrderedDict()
        # sanity check the labels : they should all be in
        for label, value in labels.items():
            assert label in self.labels_supported, "label {} in {} not supported".format(label, self.labels_supported)

        # task_wise layer attention
        printout_allocated_gpu_memory(1, " foward starting ")
        for input_name, input_tensors in input_ids_dict.items():
            # not able to output all layers anymore
            #print("INPUT {} {} ".format(input_name, input_tensors))

            sequence_output, _ = self.encoder(input_tensors, token_type_ids=None, attention_mask=attention_mask[input_name])
            sequence_output_dict[input_name] = sequence_output
            printout_allocated_gpu_memory(1, " forward pass bert")

        for task in self.tasks:
            # we don't use mask for parsing heads (cf. test performed below : the -1 already ignore the heads we don't want)
            # NB : head_masks for parsing only applies to heads not types
            head_masks_task = None # head_masks.get(task, None) if task != "parsing" else None
            # NB : head_mask means masks specific the the module heads (nothing related to parsing !! )
            assert self.task_parameters[task]["input"] in sequence_output_dict, \
                "ERROR input {} of task {} was not found in input_ids_dict {}" \
                " and therefore not in sequence_output_dict {} ".format(self.task_parameters[task]["input"],
                                                                        task, input_ids_dict.keys(),
                                                                        sequence_output_dict.keys())

            if not self.head[task].__class__.__name__ == BertOnlyMLMHead.__name__:#isinstance(self.head[task], BertOnlyMLMHead):

                logits_dict[task] = self.head[task](sequence_output_dict[self.task_parameters[task]["input"]], head_mask=head_masks_task)
            else:
                logits_dict[task] = self.head[task](sequence_output_dict[self.task_parameters[task]["input"]])
            # test performed : (logits_dict[task][0][1,2,:20]==float('-inf'))==(labels["parsing_heads"][1,:20]==-1)
            # handle several labels at output (e.g  parsing)

            printout_allocated_gpu_memory(1, " foward pass head {}".format(task))

            logits_dict = self.rename_multi_modal_task_logits(labels=self.task_parameters[task]["label"],  task=task,
                                                              logits_dict=logits_dict, task_parameters=self.task_parameters)

            printout_allocated_gpu_memory(1, "after renaming")

            for logit_label in logits_dict:

                label = re.match("(.*)-(.*)", logit_label)
                assert label is not None, "ERROR logit_label {}".format(logit_label)
                label = label.group(2)
                if label in self.task_parameters[task]["label"]:
                    _labels = None
                    if self.task_parameters[task]["input"] == "input_masked":
                        _labels = labels.get(label)
                        if _labels is not None:
                            _labels = _labels.clone()
                            _labels[input_ids_dict["input_masked"] != self.mask_index_bert] = PAD_ID_LOSS_STANDART
                    else:
                        _labels = labels.get(label)
                    printout_allocated_gpu_memory(1, " get label head {}".format(logit_label))
                    if _labels is not None:
                        #print("LABEL label {} {}".format(label, _labels))
                        loss_dict[logit_label] = self.get_loss(loss_func=self.task_parameters[task]["loss"],
                                                               label=label, num_label_dic=self.num_labels_dic,
                                                               labels=_labels, logits_dict=logits_dict, task=task,
                                                               logit_label=logit_label,
                                                               head_label=labels["heads"] if label == "types" else None)
                    printout_allocated_gpu_memory(1, " get loss {}".format(task))
                printout_allocated_gpu_memory(1, " puting to cpu {}".format(logit_label))
        # thrid output is for potential attention weights

        return logits_dict, loss_dict, None

    def append_extra_heads_model(self, downstream_tasks, num_labels_dic_new):

        self.labels_supported.extend([label for task in downstream_tasks for label in self.task_parameters[task]["label"]])
        self.sanity_check_new_num_labels_per_task(num_labels_new=num_labels_dic_new, num_labels_original=self.num_labels_dic)
        self.num_labels_dic.update(num_labels_dic_new)
        for new_task in downstream_tasks:
            if new_task not in self.tasks:
                num_label = get_key_name_num_label(new_task, self.task_parameters)
                self.head[new_task] = eval(self.task_parameters[new_task]["head"])(self.config, num_labels=num_labels_dic_new[num_label])

        # we update the tasks attributes
        self.tasks_available = list(set(self.tasks+downstream_tasks))
        self.tasks = downstream_tasks # tasks to be used at prediction time (+ possibly train)

    @staticmethod
    def get_loss(loss_func, label, num_label_dic, labels, logits_dict, task, logit_label, head_label=None):
        if label not in ["heads", "types"]:
            try:
                loss = loss_func(logits_dict[logit_label].view(-1, num_label_dic[logit_label]), labels.view(-1))
            except Exception as e:
                print(e)
                pdb.set_trace()
                print("ERROR task {} num_label {} , labels {} ".format(task, num_label_dic, labels.view(-1)))
                raise(e)

        elif label == "heads":
            # trying alternative way for loss
            loss = CrossEntropyLoss(ignore_index=LABEL_PARAMETER[label]["pad_value"],
                                    reduction="sum")(logits_dict[logit_label].view(-1, logits_dict[logit_label].size(2)), labels.view(-1))
            # other possibilities is to do log softmax then L1 loss (lead to other results)

        elif label == "types":
            assert head_label is not None, "ERROR head_label should be passed"
            # gold label after removing 0 gold
            gold = labels[head_label != LABEL_PARAMETER["heads"]["pad_value"]]
            # pred logits (after removing -1) on the gold heads
            pred = logits_dict["parsing-types"][(head_label != LABEL_PARAMETER["heads"]["pad_value"]).nonzero()[:, 0],
                                                (head_label != LABEL_PARAMETER["heads"]["pad_value"]).nonzero()[:, 1], head_label[head_label != LABEL_PARAMETER["heads"]["pad_value"]]]
            # remark : in the way it's coded for paring : the padding is already removed (so ignore index is null)
            loss = loss_func(pred, gold)

        return loss



    @staticmethod
    def rename_multi_modal_task_logits(labels, logits_dict, task, task_parameters):
        #if n_pred == 2:

        n_pred = len(list(logits_dict[task]))
        # try:
        assert n_pred == len(task_parameters[task]["label"]), \
            "ERROR : not as many labels as prediction for task {} : {} vs {} ".format(task, task_parameters[task][
                "label"], logits_dict[task])

        for i_label, label in enumerate(labels):
            # NB : the order of self.task_parameters[task]["label"] must be the same as the head output
            logits_dict[task+"-"+label] = logits_dict[task][i_label].clone()
        del logits_dict[task]

        return logits_dict



    @staticmethod
    def sanity_checking_num_labels_per_task(num_labels_per_task, tasks, task_parameters):
        for task in tasks:
            # for mwe_prediction no need of num_label we use the embedding matrix
            # do we need to check num_label for this task ? and is only 1 label assosiated to this task
            if task_parameters[task]["num_labels_mandatory"] and len(task_parameters[task]["label"]) == 1:
                if task != "parsing":
                    assert task+"-"+task_parameters[task]["label"][0] in num_labels_per_task,\
                        "ERROR : no num label for task+label {} ".format(task+"-"+task_parameters[task]["label"][0])
                else:
                    assert task in num_labels_per_task, "ERROR : no num label for task {} ".format(task)
            elif task_parameters[task]["num_labels_mandatory"] and len(task_parameters[task]["label"]) > 1:
                num_labels_mandatory_to_check = task_parameters[task].get("num_labels_mandatory_to_check")
                assert num_labels_mandatory_to_check is not None, "ERROR : task {} is related to at least 2 labels :" \
                                                                  " we need to know which one requires a num_label " \
                                                                  "to define the model head but field {} " \
                                                                  "not found in {}".format(task, "num_labels_mandatory_to_check", task_parameters[task])
                for label in num_labels_mandatory_to_check:
                    assert task+"-"+label in num_labels_per_task, "ERROR : task {} label {} not in num_labels_per_task {} dictionary".format(task, label, num_labels_per_task)
    @staticmethod
    def sanity_check_new_num_labels_per_task(num_labels_original, num_labels_new):
        for label in num_labels_new:
            if label in num_labels_original:
                assert num_labels_original[label] == num_labels_new[label], \
                    "ERROR new num label provided for existing task not the same as original original:{} new:{} ".format(num_labels_original[label], num_labels_new[label])







