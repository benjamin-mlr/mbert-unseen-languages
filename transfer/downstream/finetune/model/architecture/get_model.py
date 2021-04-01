from transfer.downstream.finetune.env.imports import logging, tarfile, tempfile, torch, json, pdb, nn, OrderedDict, os

from transfer.downstream.finetune.env.dir.pretrained_model_dir import BERT_MODEL_DIC
from transfer.downstream.finetune.args.args_parse import get_config_param_to_modify

from transfer.downstream.finetune.io_.logger import printing
from transfer.downstream.finetune.io_.report.report_tools import get_init_args_dir

#from transfer.downstream.finetune.transformers.transformers.modeling_xlm import XLMModel
from transfer.downstream.finetune.env.dir.pretrained_model_dir import DIR_2_STAT_MAPPING

#from transfer.downstream.finetune.transformers.transformers.modeling_bert import BertConfig, BertModel
#from transfer.downstream.finetune.transformers.transformers.modeling_roberta import RobertaModel, RobertaConfig
from transformers import BertConfig, BertModel, AutoModel, RobertaModel

#from transfer.downstream.finetune.transformers.transformers.modeling_multitask import BertMultiTask
#from transfer.downstream.finetune.transformers.transformers.modeling_multitask_xlm import BertMultiTaskXLM
from transfer.downstream.finetune.model.architecture.modeling_multitask import BertMultiTask


def make_bert_multitask(pretrained_model_dir, tasks, num_labels_per_task, init_args_dir, mask_id, encoder=None, args=None):
    assert num_labels_per_task is not None and isinstance(num_labels_per_task, dict), \
        "ERROR : num_labels_per_task {} should be a dictionary".format(num_labels_per_task)
    assert isinstance(tasks, list) and len(tasks) >= 1, "ERROR tasks {} should be a list of len >=1".format(tasks)

    if init_args_dir is None:
        if pretrained_model_dir is None:

            pretrained_model_dir = args.bert_model
        # assert args.output_attentions is None or not args.output_attentions, "ERROR not supported "

        multitask_wrapper = BertMultiTask

        def get_state_dict_mapping(model):
            if model.startswith("xlm") or model.startswith("rob") or model.startswith("camembert"):
                return {"roberta": "encoder",  # "lm_head":,
                        "lm_head.decoder": "head.mlm.predictions.decoder",
                        "lm_head.dense": "head.mlm.predictions.transform.dense",
                        "lm_head.bias": "head.mlm.predictions.bias",
                        "lm_head.layer_norm": "head.mlm.predictions.transform.LayerNorm"}
            elif model.startswith("bert") or model.startswith("cahya") or model.startswith("KB"):
                return {"bert": "encoder", "cls": "head.mlm"}
            elif model.startswith("asafaya"):
                return {"bert": "encoder", "cls": "head.mlm"}
            else:
                raise(Exception(f"not supported by {multitask_wrapper} needs to define a "))

        state_dict_mapping = get_state_dict_mapping(args.bert_model)

        model = multitask_wrapper.from_pretrained(pretrained_model_dir,
                                                  tasks=tasks,
                                                  mask_id=mask_id,
                                                  output_attentions=args.output_attentions,
                                                  output_hidden_states=args.output_all_encoded_layers,
                                                  output_hidden_states_per_head=args.output_hidden_states_per_head,
                                                  hard_skip_attention_layers= args.hard_skip_attention_layers,
                                                  hard_skip_all_layers= args.hard_skip_all_layers,
                                                  hard_skip_dense_layers=args.hard_skip_dense_layers,
                                                  num_labels_per_task=num_labels_per_task,
                                                  mapping_keys_state_dic=state_dict_mapping,#DIR_2_STAT_MAPPING[pretrained_model_dir],
                                                  encoder=eval(encoder) if encoder is not None else BertModel,
                                                  dropout_classifier=args.dropout_classifier,
                                                  hidden_dropout_prob=args.hidden_dropout_prob,
                                                  random_init=args.random_init, load_params_only_ls=None, not_load_params_ls=args.not_load_params_ls)

    elif init_args_dir is not None:
        assert pretrained_model_dir is not None, "ERROR model_dir is needed here for reloading"
        init_args_dir = get_init_args_dir(init_args_dir)
        args_checkpoint = json.load(open(init_args_dir, "r"))
        assert "checkpoint_dir" in args_checkpoint, "ERROR checkpoint_dir not in {} ".format(args_checkpoint)

        checkpoint_dir = args_checkpoint["checkpoint_dir"]
        assert os.path.isfile(checkpoint_dir), "ERROR checkpoint {} not found ".format(checkpoint_dir)
        # redefining model and reloading
        def get_config_bert(bert_model, config_file_name="bert_config.json"):
            model_dir = BERT_MODEL_DIC[bert_model]["model"]
            #tempdir = tempfile.mkdtemp()
            #print("extracting archive file {} to temp dir {}".format(model_dir, tempdir))
            #with tarfile.open(model_dir, 'r:gz') as archive:
            #    archive.extractall(tempdir)
            #serialization_dir = tempdir
            serialization_dir = None
            config_file = os.path.join(model_dir, config_file_name)
            try:
                assert os.path.isfile(config_file), "ERROR {} not a file , extracted from {} : dir includes {} ".format(config_file, model_dir, [x[0] for x in os.walk(serialization_dir)])
            except Exception as e:
                config_file = os.path.join(model_dir, "config.json")
                assert os.path.join(config_file)
            return config_file

        config_file = get_config_bert(args_checkpoint["hyperparameters"]["bert_model"])
        encoder = eval(BERT_MODEL_DIC[args_checkpoint["hyperparameters"]["bert_model"]]["encoder"])
        config = BertConfig(config_file, output_attentions=args.output_attentions,
                            output_hidden_states=args.output_all_encoded_layers,
                            output_hidden_states_per_head=args.output_hidden_states_per_head)
        #
        config.vocab_size = 119547
        
        model = BertMultiTask(config=config, tasks=[task for tasks in args_checkpoint["hyperparameters"]["tasks"] for task in tasks], num_labels_per_task=args_checkpoint["info_checkpoint"]["num_labels_per_task"],
                              encoder=encoder, mask_id=mask_id)
        printing("MODEL : loading model from checkpoint {}", var=[checkpoint_dir], verbose=1, verbose_level=1)
        model.load_state_dict(torch.load(checkpoint_dir, map_location=lambda storage, loc: storage))
        model.append_extra_heads_model(downstream_tasks=tasks, num_labels_dic_new=num_labels_per_task)
    else:
        raise(Exception("only one of pretrained_model_dir checkpoint_dir can be defined "))

    return model


def get_model_multi_task_bert(args, model_dir, mask_id, encoder, num_labels_per_task=None):
    # we flatten the tasks to make the model (we don't need to know if tasks are simulateneaous or not )
    model = make_bert_multitask(args=args, pretrained_model_dir=model_dir, init_args_dir=args.init_args_dir,
                                tasks=[task for tasks in args.tasks for task in tasks],
                                mask_id=mask_id,encoder=encoder,
                                num_labels_per_task=num_labels_per_task)
   
    return model
