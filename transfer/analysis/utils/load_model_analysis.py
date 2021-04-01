
import sys
import os
import random

# sys.path.append("/Users/bemuller/Documents/Work/INRIA/dev/transfer/")
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", ".."))
from transfer.downstream.finetune.env.flags import REPORT_FLAG_DIR_STR

from transfer.downstream.finetune.env.imports import pdb, torch, OrderedDict, np, linear_model, json, uuid4
from transfer.downstream.finetune.model.architecture.get_model import make_bert_multitask
from transfer.analysis.attention_analysis.attention_extraction_utils import get_dirs, load_lang_ls
from transfer.downstream.finetune.args.args_parse import args_train, args_preprocessing, args_attention_analysis, args_preprocess_attention_analysis
from transfer.downstream.finetune.io_.dat import conllu_data
from transfer.downstream.finetune.trainer.tools.multi_task_tools import get_vocab_size_and_dictionary_per_task, update_batch_size_mean
from transfer.downstream.finetune.env.dir.pretrained_model_dir import BERT_MODEL_DIC
from transfer.downstream.finetune.env.dir.data_dir import DATA_UD_RAW, DATA_UD
from transfer.downstream.finetune.model.settings import TASKS_PARAMETER, LABEL_PARAMETER, SAMPLES_PER_TASK_TO_REPORT
from transfer.downstream.finetune.env.dir.project_directories import PROJECT_PATH


#from transfer.downstream.finetune.transformers.transformers.tokenization_bert import BertTokenizer, XLMRobertaTokenizer
from transformers import BertTokenizer, XLMRobertaTokenizer



try:
    sys.path.append(os.environ.get("EXPERIENCE"))
    from transfer.write_to_performance_repo import report_template
except:
    from transfer.downstream.finetune.evaluate.score.report_template import report_template
    print("REPORTING modules downloaded from local project ")


def load_all_analysis(args, dict_path, model_dir):

    encoder = BERT_MODEL_DIC[args.bert_model]["encoder"]
    vocab_size = BERT_MODEL_DIC[args.bert_model]["vocab_size"]
    voc_tokenizer = BERT_MODEL_DIC[args.bert_model]["vocab"]

    tokenizer = eval(BERT_MODEL_DIC[args.bert_model]["tokenizer"])
    random.seed(args.seed)

    if args.model_id_pref is None:
        run_id = str(uuid4())[:4]
    else:
        run_id = args.model_id_pref +"1"

    if args.init_args_dir is None:
        dict_path += "/" + run_id
        os.mkdir(dict_path)

    tokenizer = tokenizer.from_pretrained(voc_tokenizer, do_lower_case=args.case == "lower",
                                          shuffle_bpe_embedding=False)
    mask_id = tokenizer.encode(["[MASK]"])[0] if args.bert_model == "bert_base_multilingual_cased" else None

    _dev_path = args.dev_path if args.dev_path is not None else args.train_path
    word_dictionary, word_norm_dictionary, char_dictionary, pos_dictionary, \
    xpos_dictionary, type_dictionary = \
        conllu_data.load_dict(dict_path=dict_path,
                              train_path=args.train_path if args.init_args_dir is None else None,
                              dev_path=args.dev_path if args.init_args_dir is None else None,
                              test_path=args.test_paths if args.init_args_dir is None else None,
                              word_embed_dict={},
                              dry_run=False,
                              expand_vocab=False,
                              word_normalization=True,
                              force_new_dic=False,
                              tasks=args.tasks,
                              pos_specific_data_set=None,
                              # pos_specific_data_set=args.train_path[1] if len(args.tasks) > 1 and len(
                              #    args.train_path) > 1 and "pos" in args.tasks else None,
                              case=args.case,
                              # if not normalize pos or parsing in tasks we don't need dictionary
                              do_not_fill_dictionaries=len(set(["normalize", "pos", "parsing"]) & set(
                                  [task for tasks in args.tasks for task in tasks])) == 0,
                              add_start_char=True if args.init_args_dir is None else None,
                              verbose=1)

    num_labels_per_task, task_to_label_dictionary = get_vocab_size_and_dictionary_per_task(
        [task for tasks in args.tasks for task in tasks],
        vocab_bert_wordpieces_len=vocab_size,
        pos_dictionary=pos_dictionary,
        type_dictionary=type_dictionary,
        task_parameters=TASKS_PARAMETER)

    model = make_bert_multitask(args=args, pretrained_model_dir=model_dir, init_args_dir=args.init_args_dir,
                                tasks=[task for tasks in args.tasks for task in tasks],
                                mask_id=mask_id, encoder=encoder,
                                num_labels_per_task=num_labels_per_task)

    if torch.cuda.is_available():
        print("CUDA is available ")
        model.cuda()

    return model, tokenizer, run_id

