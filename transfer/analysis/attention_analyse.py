import sys
import os
import random


#sys.path.append("/Users/bemuller/Documents/Work/INRIA/dev/transfer/")
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
from transfer.downstream.finetune.transformers.transformers.tokenization_bert import BertTokenizer
from transfer.analysis.attention_analysis.attention_extraction import get_hidden_representation


try:
    sys.path.append(os.environ.get("EXPERIENCE"))
    from transfer.write_to_performance_repo import report_template
except:
    from transfer.downstream.finetune.evaluate.score.report_template import report_template
    print("REPORTING modules downloaded from local project ")



def main(args, dict_path, model_dir):

    encoder = BERT_MODEL_DIC[args.bert_model]["encoder"]
    vocab_size = BERT_MODEL_DIC[args.bert_model]["vocab_size"]
    voc_tokenizer = BERT_MODEL_DIC[args.bert_model]["vocab"]

    tokenizer = eval(BERT_MODEL_DIC[args.bert_model]["tokenizer"])
    random.seed(args.seed)

    if args.model_id_pref is None:
        run_id = str(uuid4())[:4]
    else:
        run_id = args.model_id_pref+"1"

    if args.init_args_dir is None:
        dict_path+="/"+run_id
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
                              #pos_specific_data_set=args.train_path[1] if len(args.tasks) > 1 and len(
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

    def get_n_params(model):
        pp = 0
        for p in list(model.parameters()):
            nn = 1
            for s in list(p.size()):

                nn = nn * s
            pp += nn
        return pp
    param = get_n_params(model)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])

    pdb.set_trace()

    data = ["I am here", "How are you"]
    model.eval()
    n_obs = args.n_sent
    max_len = args.max_seq_len
    lang_ls = args.raw_text_code
    data_all, y_all = load_lang_ls(DATA_UD_RAW, lang_ls=lang_ls)

    reg = linear_model.LogisticRegression()
    X_train = OrderedDict()
    X_test = OrderedDict()
    y_train = OrderedDict()
    y_test = OrderedDict()
    # just to get the keyw
    layer_head_att = get_hidden_representation(data, model, tokenizer, max_len=max_len, output_dic=False, pad_below_max_len=True)
    layer_head_att = layer_head_att[0]
    report_ls = []
    accuracy_dic = OrderedDict()
    sampling = args.sampling
    for ind, layer_head in enumerate(list(layer_head_att.keys())):
        report = OrderedDict()
        accuracy_ls = []
        layer_head = list(layer_head_att.keys())[len(list(layer_head_att.keys()))-ind-1]
        for _ in range(sampling):
            sample_ind = random.sample(population=range(len(data_all)), k=n_obs)
            sample_ind_test = random.sample(population=range(len(data_all)), k=n_obs)

            data = data_all[sample_ind]
            y = y_all[sample_ind]

            all = get_hidden_representation(data, model, tokenizer, max_len=max_len, output_dic=False, pad_below_max_len=True)

            layer_head_att = all[0]

            #pdb.set_trace()
            def reshape_x(z):
                return np.array(z.view(z.size(0)*z.size(1), -1))
            def reshape_y(z, n_seq):
                'multiply each element n_seq times'
                new_z = []
                for _z in z:
                    #for _ in range(n_seq):
                    new_z.extend([_z for _ in range(n_seq)])
                return np.array(new_z)
                #return np.array(z.view(z.size(0), -1).transpose(1, 0))
            #X_train[layer_head] = np.array(layer_head_att[layer_head].view(layer_head_att[layer_head].size(0), -1).transpose(1,0))
            X_train[layer_head] = reshape_x(layer_head_att[layer_head])
            y_train[layer_head] = reshape_y(y, max_len)
            #db.set_trace()
            #y_train[layer_head] = y

            reg.fit(X=X_train[layer_head], y=y_train[layer_head])

            # test
            data_test = data_all[sample_ind_test]
            layer_head_att_test = get_hidden_representation(data_test, model, tokenizer, max_len=max_len, output_dic=False, pad_below_max_len=True)
            X_test[layer_head] = reshape_x(layer_head_att_test[layer_head])
            y_test[layer_head] = reshape_y(y_all[sample_ind_test], max_len)

            y_pred = reg.predict(X_test[layer_head])

            Accuracy = np.sum((y_test[layer_head] == y_pred))/len(y_test[layer_head])
            accuracy_ls.append(Accuracy)

        accuracy_dic[layer_head] = np.mean(accuracy_ls)
        layer = layer_head.split("-")[0]
        if layer not in accuracy_dic:
            accuracy_dic[layer] = []
        accuracy_dic[layer].append(np.mean(accuracy_ls))

        print(f"Regression {layer_head} Accuracy test {np.mean(accuracy_ls)} on {n_obs * max_len}"
              f" word sample from {len(lang_ls)} languages task {args.tasks} args {'/'.join(args.init_args_dir.split('/')[-2:]) if args.init_args_dir is not None else None} "
              f"bert {args.bert_model} random init {args.random_init} std {np.std(accuracy_ls)} sampling {len(accuracy_ls)}=={sampling}")


        #report["model_type"] = args.bert_model if args.init_args_dir is None else args.tasks[0][0]+"-tune"
        #report["accuracy"] = np.mean(accuracy_ls)
        #report["sampling"] = len(accuracy_ls)
        #report["std"] = np.std(accuracy_ls)
        #report["n_sent"] = n_obs
        #report["n_obs"] = n_obs*max_len

        report = report_template(metric_val="accuracy", subsample=",".join(lang_ls),
                                 info_score_val=sampling,
                                 score_val=np.mean(accuracy_ls),
                                 n_sents=n_obs,
                                 avg_per_sent=np.std(accuracy_ls),
                                 n_tokens_score=n_obs*max_len,
                                 model_full_name_val=run_id, task="attention_analysis",
                                 evaluation_script_val="exact_match",
                                 model_args_dir=args.init_args_dir if args.init_args_dir is not None else args.random_init,
                                 token_type="word",
                                 report_path_val=None,
                                 data_val=layer_head)
        report_ls.append(report)
        
        # break

    for key in accuracy_dic:
        print(f"Summary {key} {np.mean(accuracy_dic[key])} model word sample from {len(lang_ls)} languages task {args.tasks} args {'/'.join(args.init_args_dir.split('/')[-2:]) if args.init_args_dir is not None else None} "
              f"bert {args.bert_model} random init {args.random_init} std {np.std(accuracy_ls)} sampling {len(accuracy_ls)}=={sampling}")
        
    if args.report_dir is None:
        report_dir = PROJECT_PATH+f"/../../analysis/attention_analysis/report/{run_id}-report"
        os.mkdir(report_dir)
    else:
        report_dir = args.report_dir
    assert os.path.isdir(report_dir)
    with open(report_dir+"/report.json", "w") as f:
        json.dump(report_ls, f)
    overall_report = args.overall_report_dir+"/"+args.overall_label+"-grid-report.json"
    with open(overall_report,"r") as g:
        report_all = json.load(g)
        report_all.extend(report_ls)
    with open(overall_report,"w") as file:
        json.dump(report_all, file)

    print("{} {} ".format(REPORT_FLAG_DIR_STR,overall_report))
    #printing("{} {} ", var=[REPORT_FLAG_DIR_STR, report_dir], verbose=verbose, verbose_level=0)


if __name__ == "__main__":
    # args.init_args_dir = "/Users/bemuller/Documents/Work/INRIA/dev/transfer/transfer/downstream/finetune/env/dir/../.././checkpoints/bert/dd1d5-2178d-dd1d5_job-3c621_model/dd1d5-2178d-dd1d5_job-3c621_model-args.json"
    args = args_attention_analysis()
    args = args_preprocess_attention_analysis(args)
    args, dict_path, model_dir = get_dirs(args)
    main(args, dict_path, model_dir)

# run on pretrained model / random model / fine-tuned model
# make it in neff

# compute F1 average and F1 per class
# real split train test
# same with the task
# report results in a dictionary with std-eviation


