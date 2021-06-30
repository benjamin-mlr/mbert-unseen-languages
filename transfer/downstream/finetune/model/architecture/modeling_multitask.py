
import logging
import re
import torch
import os
from io import open
from collections import OrderedDict
import six
from torch.nn import functional as F


logger = logging.getLogger(__name__)
#from .configuration_utils import PretrainedConfig
#from transformers.file_utils import cached_path, WEIGHTS_NAME, TF_WEIGHTS_NAME, TF2_WEIGHTS_NAME
from transformers.file_utils import (
    DUMMY_INPUTS,
    TF2_WEIGHTS_NAME,
    TF_WEIGHTS_NAME,
    WEIGHTS_NAME,
    cached_path,
    hf_bucket_url,
    is_remote_url,
)

from transfer.downstream.finetune.env.imports import nn,  OrderedDict, re, pdb, CrossEntropyLoss
from transfer.downstream.finetune.model.settings import TASKS_PARAMETER, LABEL_PARAMETER
from transfer.downstream.finetune.model.constants import PAD_ID_LOSS_STANDART
from transfer.downstream.finetune.env.gpu_tools.gpu_info import printout_allocated_gpu_memory
from transfer.downstream.finetune.model.tools import get_key_name_num_label
from transfer.downstream.finetune.model.architecture.parser_modules.mlp import MLP
from transfer.downstream.finetune.model.architecture.parser_modules.biaffine import Biaffine

# from transfer.downstream.finetune.transformers.transformers.modeling_bert import BertPreTrainedModel, BertOnlyMLMHead
# from .modeling_xlm import XLMPreTrainedModel, XLMModel
# only supported for Bert and Roberta so far

from transformers import BertPreTrainedModel, RobertaModel, BertForMaskedLM
from transformers.modeling_bert import BertOnlyMLMHead, BertPreTrainedModel, PreTrainedModel, BertModel
from transformers import AutoModel
CLASS_PRETRAINED_MODEL = BertPreTrainedModel#BertPreTrainedModel#XLMPreTrainedModel# BertPreTrainedModel#XLMPreTrainedModel#BertPreTrainedModel # XLMPreTrainedModel#


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


class BertMultiTask(CLASS_PRETRAINED_MODEL):
    """
    BERT model which can call any other modules
    """

    def __init__(self, config, **kwargs):# tasks, num_labels_per_task, mask_id, encoder_class):
        super(BertMultiTask, self).__init__(config)
        # encoder_class only BertModel or RobertaModel
        # some arguments specific to BertMultiTask could be passed in config or in kwargs
        #self.pretrained_model_archive_map = BERT_PRETRAINED_MODEL_ARCHIVE_MAP
        encoder_class = kwargs["encoder"]
        tasks = kwargs["tasks"] if "tasks" in kwargs else config.tasks
        num_labels_per_task = kwargs["num_labels_per_task"] if "num_labels_per_task" in kwargs else config.tasks
        mask_id = kwargs["mask_id"] if "mask_id" in kwargs else config.tasks
        config.dropout_classifier = kwargs.get("dropout_classifier", 0.1)

        if "parsing" in tasks:
            config.graph_head_hidden_size_mlp_arc = kwargs.get("graph_head_hidden_size_mlp_arc", 500)
            config.graph_head_hidden_size_mlp_rel = kwargs.get("graph_head_hidden_size_mlp_rel", 200)

        if "AutoModel" in str(encoder_class):
            self.encoder = encoder_class.from_config(config)
        else:
            self.encoder = encoder_class(config)

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
            sequence_output = self.encoder(input_tensors, token_type_ids=None, attention_mask=attention_mask[input_name])
            try:
                assert len(sequence_output) == 2+int(self.config.output_attentions)+int(self.config.output_hidden_states), f"ERROR should be {2+int(self.config.output_attentions)+int(self.config.output_hidden_states_per_head)+int(self.config.output_hidden_states)} : check that you're not outputing also all hidden states"
                # add to remove : int(self.config.output_hidden_states_per_head)
            except Exception as e:
                #print(f"Exception output sequence {e}")
                pdb.set_trace()
                assert len(sequence_output) == 2
            logits = sequence_output[0]

            sequence_output_dict[input_name] = logits
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
        output = (logits_dict, loss_dict, )

        output = output + sequence_output[2:]

        return output

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

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """Instantiate a pretrained pytorch model from a pre-trained model configuration.

        The model is set in evaluation mode by default using ``model.eval()`` (Dropout modules are deactivated)
        To train the model, you should first set it back in training mode with ``model.train()``

        The warning ``Weights from XXX not initialized from pretrained model`` means that the weights of XXX do not come pre-trained with the rest of the model.
        It is up to you to train those weights with a downstream fine-tuning task.

        The warning ``Weights from XXX not used in YYY`` means that the layer XXX is not used by YYY, therefore those weights are discarded.

        Parameters:
            pretrained_model_name_or_path: either:

                - a string with the `shortcut name` of a pre-trained model to load from cache or download, e.g.: ``bert-base-uncased``.
                - a path to a `directory` containing model weights saved using :func:`~transformers.PreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/``.
                - a path or url to a `tensorflow index checkpoint file` (e.g. `./tf_model/model.ckpt.index`). In this case, ``from_tf`` should be set to True and a configuration object should be provided as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
                - None if you are both providing the configuration and state dictionary (resp. with keyword arguments ``config`` and ``state_dict``)

            model_args: (`optional`) Sequence of positional arguments:
                All remaning positional arguments will be passed to the underlying model's ``__init__`` method

            config: (`optional`) instance of a class derived from :class:`~transformers.PretrainedConfig`:
                Configuration for the model to use instead of an automatically loaded configuation. Configuration can be automatically loaded when:

                - the model is a model provided by the library (loaded with the ``shortcut-name`` string of a pretrained model), or
                - the model was saved using :func:`~transformers.PreTrainedModel.save_pretrained` and is reloaded by suppling the save directory.
                - the model is loaded by suppling a local directory as ``pretrained_model_name_or_path`` and a configuration JSON file named `config.json` is found in the directory.

            state_dict: (`optional`) dict:
                an optional state dictionnary for the model to use instead of a state dictionary loaded from saved weights file.
                This option can be used if you want to create a model from a pretrained configuration but load your own weights.
                In this case though, you should check if using :func:`~transformers.PreTrainedModel.save_pretrained` and :func:`~transformers.PreTrainedModel.from_pretrained` is not a simpler option.

            cache_dir: (`optional`) string:
                Path to a directory in which a downloaded pre-trained model
                configuration should be cached if the standard cache should not be used.

            force_download: (`optional`) boolean, default False:
                Force to (re-)download the model weights and configuration files and override the cached versions if they exists.

            proxies: (`optional`) dict, default None:
                A dictionary of proxy servers to use by protocol or endpoint, e.g.: {'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.
                The proxies are used on each request.

            output_loading_info: (`optional`) boolean:
                Set to ``True`` to also return a dictionnary containing missing keys, unexpected keys and error messages.

            kwargs: (`optional`) Remaining dictionary of keyword arguments:
                Can be used to update the configuration object (after it being loaded) and initiate the model. (e.g. ``output_attention=True``). Behave differently depending on whether a `config` is provided or automatically loaded:

                - If a configuration is provided with ``config``, ``**kwargs`` will be directly passed to the underlying model's ``__init__`` method (we assume all relevant updates to the configuration have already been done)
                - If a configuration is not provided, ``kwargs`` will be first passed to the configuration class initialization function (:func:`~transformers.PretrainedConfig.from_pretrained`). Each key of ``kwargs`` that corresponds to a configuration attribute will be used to override said attribute with the supplied ``kwargs`` value. Remaining keys that do not correspond to any configuration attribute will be passed to the underlying model's ``__init__`` function.

        Examples::

            model = BertModel.from_pretrained('bert-base-uncased')    # Download model and configuration from S3 and cache.
            model = BertModel.from_pretrained('./test/saved_model/')  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
            model = BertModel.from_pretrained('bert-base-uncased', output_attention=True)  # Update configuration during loading
            assert model.config.output_attention == True
            # Loading from a TF checkpoint file instead of a PyTorch model (slower)
            config = BertConfig.from_json_file('./tf_model/my_tf_model_config.json')
            model = BertModel.from_pretrained('./tf_model/my_tf_checkpoint.ckpt.index', from_tf=True, config=config)

        """
        config = kwargs.pop('config', None)
        state_dict = kwargs.pop('state_dict', None)
        cache_dir = kwargs.pop('cache_dir', None)
        from_tf = kwargs.pop('from_tf', False)
        force_download = kwargs.pop('force_download', False)
        proxies = kwargs.pop('proxies', None)
        output_loading_info = kwargs.pop('output_loading_info', False)
        random_init = kwargs.pop("random_init", False)
        use_cdn = kwargs.pop("use_cdn", True)
        local_files_only = kwargs.pop("local_files_only", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        kwargs_config = kwargs.copy()

        mapping_keys_state_dic = kwargs.pop("mapping_keys_state_dic", None)
        kwargs_config.pop("mapping_keys_state_dic", None)

        if config is None:

            config, model_kwargs = cls.config_class.from_pretrained(pretrained_model_name_or_path, *model_args,
                                                                    cache_dir=cache_dir, return_unused_kwargs=True,
                                                                    force_download=force_download, **kwargs_config)
        else:
            model_kwargs = kwargs

        # Load model
        print('loading from ', pretrained_model_name_or_path)
        if pretrained_model_name_or_path is not None:
            if os.path.isdir(pretrained_model_name_or_path):
                if from_tf and os.path.isfile(os.path.join(pretrained_model_name_or_path, TF_WEIGHTS_NAME + ".index")):
                    # Load from a TF 1.0 checkpoint
                    archive_file = os.path.join(pretrained_model_name_or_path, TF_WEIGHTS_NAME + ".index")
                elif from_tf and os.path.isfile(os.path.join(pretrained_model_name_or_path, TF2_WEIGHTS_NAME)):
                    # Load from a TF 2.0 checkpoint
                    archive_file = os.path.join(pretrained_model_name_or_path, TF2_WEIGHTS_NAME)
                elif os.path.isfile(os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)):
                    # Load from a PyTorch checkpoint
                    archive_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
                else:
                    raise EnvironmentError(
                        "Error no file named {} found in directory {} or `from_tf` set to False".format(
                            [WEIGHTS_NAME, TF2_WEIGHTS_NAME, TF_WEIGHTS_NAME + ".index"],
                            pretrained_model_name_or_path,
                        )
                    )
            elif os.path.isfile(pretrained_model_name_or_path) or is_remote_url(pretrained_model_name_or_path):
                archive_file = pretrained_model_name_or_path
            elif os.path.isfile(pretrained_model_name_or_path + ".index"):
                assert (
                    from_tf
                ), "We found a TensorFlow checkpoint at {}, please set from_tf to True to load from this checkpoint".format(
                    pretrained_model_name_or_path + ".index"
                )
                archive_file = pretrained_model_name_or_path + ".index"
            else:
                archive_file = hf_bucket_url(
                    pretrained_model_name_or_path,
                    filename=(TF2_WEIGHTS_NAME if from_tf else WEIGHTS_NAME),
                    use_cdn=use_cdn,
                )

            try:
                # Load from URL or cache if already cached
                resolved_archive_file = cached_path(
                    archive_file,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    local_files_only=local_files_only,
                )
                if resolved_archive_file is None:
                    raise EnvironmentError
            except EnvironmentError:
                msg = (
                    f"Can't load weights for '{pretrained_model_name_or_path}'. Make sure that:\n\n"
                    f"- '{pretrained_model_name_or_path}' is a correct model identifier listed on 'https://huggingface.co/models'\n\n"
                    f"- or '{pretrained_model_name_or_path}' is the correct path to a directory containing a file named one of {WEIGHTS_NAME}, {TF2_WEIGHTS_NAME}, {TF_WEIGHTS_NAME}.\n\n"
                )
                raise EnvironmentError(msg)

            if resolved_archive_file == archive_file:
                logger.info("loading weights file {}".format(archive_file))
            else:
                logger.info("loading weights file {} from cache at {}".format(archive_file, resolved_archive_file))
        else:
            resolved_archive_file = None

        # Instantiate model.

        model = cls(config, *model_args, **model_kwargs)

        if state_dict is None and not from_tf:
            state_dict = torch.load(resolved_archive_file, map_location='cpu')
        
        missing_keys = []
        unexpected_keys = []
        error_msgs = []

        if from_tf:
            if resolved_archive_file.endswith('.index'):
                # Load from a TensorFlow 1.X checkpoint - provided by original authors
                model = cls.load_tf_weights(model, config, resolved_archive_file[:-6])  # Remove the '.index'
            else:
                # Load from our TensorFlow 2.0 checkpoints
                try:
                    from transformers import load_tf2_checkpoint_in_pytorch_model
                    model = load_tf2_checkpoint_in_pytorch_model(model, resolved_archive_file, allow_missing_keys=True)
                except ImportError as e:
                    logger.error(
                        "Loading a TensorFlow model in PyTorch, requires both PyTorch and TensorFlow to be installed. Please see "
                        "https://pytorch.org/ and https://www.tensorflow.org/install/ for installation instructions.")
                    raise e
        else:
            # Convert old format to new format if needed from a PyTorch state_dict
            old_keys = []
            new_keys = []

            for key in state_dict.keys():
                new_key = None
                if 'gamma' in key:
                    new_key = key.replace('gamma', 'weight')
                if 'beta' in key:
                    new_key = key.replace('beta', 'bias')
                if new_key:
                    old_keys.append(key)
                    new_keys.append(new_key)
            for old_key, new_key in zip(old_keys, new_keys):
                state_dict[new_key] = state_dict.pop(old_key)

            # copy state_dict so _load_from_state_dict can modify it
            metadata = getattr(state_dict, '_metadata', None)
            state_dict = state_dict.copy()
            if metadata is not None:
                state_dict._metadata = metadata
            # assert mapping_keys_state_dic is not None, "ERROR did not found mapping dicts for {} ".format(pretrained_model_name_or_path)
            # mapping_keys_state_dic = {"roberta": "encoder", "lm_head": "head.mlm"}
            if mapping_keys_state_dic is not None:
                assert isinstance(mapping_keys_state_dic, dict), "ERROR "
                print("INFO : from loading from pretrained method (assuming loading original google model : "
                      "need to rename some keys {})".format(mapping_keys_state_dic))
                state_dict = cls.adapt_state_dic_to_multitask(state_dict, keys_mapping=mapping_keys_state_dic, add_prefix=pretrained_model_name_or_path=="asafaya/bert-base-arabic")
                #pdb.set_trace()
            
            def load(module, prefix=''):

                local_metadata = {"version": 1}

                if not prefix.startswith("head") or prefix.startswith("head.mlm"):
                    assert len(missing_keys) == 0, "ERROR {} missing keys in state_dict {}".format(prefix, missing_keys)
                else:
                    if len(missing_keys) == 0:
                        print("Warning {} missing keys in state_dict {} (warning expected for task-specific fine-tuning)".format(prefix, missing_keys))

                module._load_from_state_dict(state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
                for name, child in module._modules.items():
                    
                    # load_params_only_ls = kwargs.get("load_params_only_ls ")
                    not_load_params_ls = kwargs.get("not_load_params_ls") if kwargs.get(
                        "not_load_params_ls") is not None else []
                    assert isinstance(not_load_params_ls, list), f"Argument error not_load_params_ls should be a list but is {not_load_params_ls}"
                    matching_not_load = []
                    # RANDOM-INIT
                    for pattern in not_load_params_ls:
                        matching = re.match(pattern, prefix + name)
                        if matching is not None:
                            matching_not_load.append(matching)
                    if len(matching_not_load) > 0:
                        # means there is at least one patter in not load pattern that matched --> so should load
                        print("MATCH not loading : {} parameters {} ".format(prefix + name, not_load_params_ls))
                    if child is not None and len(matching_not_load) == 0:
                        #print("MODEL loading : child {} full {} ".format(name, prefix + name + '.'))
                        load(child, prefix + name + '.')
                    else:
                        print("MODEL not loading : child {} matching_not_load {} ".format(child, matching_not_load))

            # Make sure we are able to load base models as well as derived models (with heads)
            start_prefix = ''
            model_to_load = model
            if not hasattr(model, cls.base_model_prefix) and any(
                    s.startswith(cls.base_model_prefix) for s in state_dict.keys()):
                start_prefix = cls.base_model_prefix + '.'
            if hasattr(model, cls.base_model_prefix) and not any(
                    s.startswith(cls.base_model_prefix) for s in state_dict.keys()):
                model_to_load = getattr(model, cls.base_model_prefix)
            if not random_init:
                load(model_to_load, prefix=start_prefix)
            else:
                print("WARNING : RANDOM INTIALIZATION OF BERTMULTITASK")

            if len(missing_keys) > 0:
                logger.info("Weights of {} not initialized from pretrained model: {}".format(
                    model.__class__.__name__, missing_keys))
            if len(unexpected_keys) > 0:
                logger.info("Weights from pretrained model not used in {}: {}".format(
                    model.__class__.__name__, unexpected_keys))
            if len(error_msgs) > 0:
                raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                    model.__class__.__name__, "\n\t".join(error_msgs)))

        if hasattr(model, 'tie_weights'):
            model.tie_weights()  # make sure word embedding weights are still tied

        # Set model in evaluation mode to desactivate DropOut modules by default
        model.eval()

        if output_loading_info:
            loading_info = {"missing_keys": missing_keys, "unexpected_keys": unexpected_keys, "error_msgs": error_msgs}
            return model, loading_info

        return model

    @staticmethod
    def adapt_state_dic_to_multitask(state_dict, keys_mapping, add_prefix=False):
        state_dict_new = OrderedDict()
        for key, _ in state_dict.items():
            state_dict_new[key] = state_dict[key].clone()
            if add_prefix:
                print(f"keys_mapping {keys_mapping} will be ignored")
                state_dict_new["encoder." + key] = state_dict[key].clone()
                del state_dict_new[key]
            else:
                for mapping_key, mapping_value in keys_mapping.items():

                    if key.startswith(mapping_key):
                        state_dict_new[mapping_value + key[len(mapping_key):]] = state_dict[mapping_key + key[len(mapping_key):]].clone()

                        del state_dict_new[mapping_key + key[len(mapping_key):]]
        return state_dict_new


