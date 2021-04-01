
import sys
import os
import random

# sys.path.append("/Users/bemuller/Documents/Work/INRIA/dev/transfer/")
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", ".."))
from transfer.downstream.finetune.env.flags import REPORT_FLAG_DIR_STR
from transfer.downstream.finetune.io_.report.report_tools import write_args, get_hyperparameters_dict, get_dataset_label, get_name_model_id_with_extra_name

from transfer.downstream.finetune.env.imports import pdb, torch, OrderedDict, np, linear_model, json, uuid4, nn
from transfer.downstream.finetune.model.architecture.get_model import make_bert_multitask
from transfer.analysis.attention_analysis.attention_extraction_utils import get_dirs, load_lang_ls, load_data
from transfer.downstream.finetune.args.args_parse import args_train, args_preprocessing, args_attention_analysis, args_preprocess_attention_analysis
from transfer.downstream.finetune.env.dir.data_dir import DATA_UD_RAW, DATA_UD
from transfer.downstream.finetune.env.dir.project_directories import PROJECT_PATH
from transfer.analysis.attention_analysis.attention_extraction import get_hidden_representation
from transfer.analysis.utils.cka import kernel_CKA

try:
    sys.path.append(os.environ.get("EXPERIENCE"))
    from transfer.write_to_performance_repo import report_template
except:
    from transfer.downstream.finetune.evaluate.score.report_template import report_template
    print("REPORTING modules downloaded from local project ")

from transfer.analysis.utils.load_model_analysis import load_all_analysis


def main(args, dict_path, model_dir):

    model, tokenizer, run_id = load_all_analysis(args, dict_path, model_dir)

    if args.compare_to_pretrained:
        print("Loading Pretrained model also for comparison with pretrained")
        args_origin = args_attention_analysis()
        args_origin = args_preprocess_attention_analysis(args_origin)
        args_origin.init_args_dir = None
        args_origin, dict_path_0, model_dir_0 = get_dirs(args_origin)
        args_origin.model_id_pref += "again"
        model_origin, tokenizer_0, _ = load_all_analysis(args_origin, dict_path_0, model_dir_0)
        model_origin.eval()
        print("seco,")
    # only allow output of the model to be hidden states here
    print("Checkpoint loaded")
    assert not args.output_attentions
    assert args.output_all_encoded_layers and args.output_hidden_states_per_head

    data = ["I am here", "How are you"]
    model.eval()
    n_obs = args.n_sent
    max_len = args.max_seq_len
    lang_ls = args.raw_text_code

    lang = ["fr_pud", "de_pud", "ru_pud", "tr_pud", "id_pud", "ar_pud", "pt_pud",  "es_pud", "fi_pud",
         "it_pud", "sv_pud", "cs_pud", "pl_pud", "hi_pud", "zh_pud", "ko_pud", "ja_pud","th_pud"
        ]
    #lang = ["fr_pud", "fr_gsd"]
    src_lang_ls = ["en_pud"]#, "fr_pud", "ru_pud", "ar_pud"]
    print("Loading data...")
    data_target_ls = [load_data(DATA_UD+f"/{target}-ud-test.conllu",  line_filter="# text = ") for target in lang]
    data_target_dic = OrderedDict([(lang, data) for lang, data in zip(lang, data_target_ls)])
    #pdb.set_trace()
    src = src_lang_ls[0]#"en_pud"

    data_en = data_target_ls[0]#load_data(DATA_UD+f"/{src}-ud-test.conllu", line_filter="# text = ")

    for _data_target in data_target_dic:
        try:
            assert len(data_target_dic[_data_target]) == len(data_en), f"Should have as much sentences on both sides en:{len(data_en)} target:{len(data_target_dic[_data_target])}"
        except:
            data_en = data_en[:len(data_target_dic[_data_target])]
            print(f"Cutting {src} dataset based on target")
        assert len(data_target_dic[_data_target]) == len(data_en), f"Should have as much sentences on both sides en:{len(data_en)} target:{len(data_target_dic[_data_target])}"
    #reg = linear_model.LogisticRegression()
    # just to get the keyw
    layer_all = get_hidden_representation(data, model, tokenizer, max_len=max_len)
    # removed hidden_per_layer
    #pdb.set_trace()
    assert len(layer_all) == 1
    #assert len(layer_all) == 2, "ERROR should only have hidden_per_layer and hidden_per_head_layer"

    report_ls = []
    accuracy_dic = OrderedDict()
    sampling = args.sampling
    metric = args.similarity_metric
    if metric == "cka":
        pad_below_max_len, output_dic = False, True
    else:
        pad_below_max_len, output_dic = False, True
    assert metric in ["cos", "cka"]
    if metric == "cos":
        batch_size = 1
    else:
        batch_size = len(data_en)//4

    task_tuned = "No"



    if args.init_args_dir is None:
        #args.init_args_dir =
        id_model = f"{args.bert_model}-init-{args.random_init}"

        hyperparameters = OrderedDict([("bert_model", args.bert_model),
                                   ("random_init", args.random_init),
                                    ("not_load_params_ls", args.not_load_params_ls),
                                   ("dict_path", dict_path),
                                   ("model_id", id_model),])
        info_checkpoint = OrderedDict([("epochs", 0), ("batch_size", batch_size),
                     ("train_path", 0), ("dev_path", 0), ("num_labels_per_task", 0)])

        args.init_args_dir = write_args(os.environ.get("MT_NORM_PARSE", "./"), model_id=id_model,
                                        info_checkpoint=info_checkpoint,
                                        hyperparameters=hyperparameters, verbose=1)
        print("args_dir checkout ", args.init_args_dir)
        model_full_name_val = task_tuned+"-"+id_model
    else:
        argument = json.load(open(args.init_args_dir,'r'))
        task_tuned = argument["hyperparameters"]["tasks"][0][0] if not "wiki" in argument["info_checkpoint"]["train_path"] else "ner"
        model_full_name_val = task_tuned + "-" + args.init_args_dir.split("/")[-1]

    if args.analysis_mode == "layer":
        studied_ind = 0
    elif args.analysis_mode == "layer_head":
        studied_ind = 1
    else:
        raise(Exception(f"args.analysis_mode : {args.analysis_mode} corrupted"))
    layer_analysed = layer_all[studied_ind]

    #for ind, layer_head in enumerate(list(layer_analysed.keys())):
    report = OrderedDict()
    accuracy_ls = []
    src_lang = src
    
    cosine_sent_to_src = OrderedDict([(src_lang+"-"+lang, OrderedDict()) for src_lang in src_lang_ls for lang in data_target_dic.keys()])
    cosine_sent_to_origin = OrderedDict([(lang, OrderedDict()) for lang in data_target_dic.keys()])
    cosine_sent_to_origin_src = OrderedDict([(lang, OrderedDict()) for lang in src_lang_ls])
    cosine_sent_to_former_layer_src = OrderedDict([(lang, OrderedDict()) for lang in src_lang_ls])
    cosine_sent_to_former_layer = OrderedDict([(lang, OrderedDict()) for lang in data_target_dic.keys()])
    cosine_sent_to_first_layer = OrderedDict([(lang, OrderedDict()) for lang in data_target_dic.keys()])
    #layer_head = list(layer_analysed.keys())[len(list(layer_analysed.keys())) - ind -1]

    cosinus = nn.CosineSimilarity(dim=1)
    info_model = f" task {args.tasks} args {'/'.join(args.init_args_dir.split('/')[-2:]) if args.init_args_dir is not None else None} bert {args.bert_model} random init {args.random_init} "
    #"cka"
    output_dic = True
    pad_below_max_len = False
    max_len = 200

    n_batch = len(data_en)//batch_size

    for i_data in range(n_batch):

        for src_lang in src_lang_ls:
            print(f"Starting src", {src_lang})
            data_en = load_data(DATA_UD + f"/{src_lang}-ud-test.conllu", line_filter="# text = ")
            en_batch = data_en[i_data:i_data+batch_size]
            all = get_hidden_representation(en_batch, model, tokenizer, pad_below_max_len=pad_below_max_len, max_len=max_len, output_dic=output_dic)
            analysed_batch_dic_en = all[studied_ind]
            i_lang = 0

            if args.compare_to_pretrained:
                all_origin = get_hidden_representation(en_batch, model_origin, tokenizer_0,
                                                       pad_below_max_len=pad_below_max_len,
                                                       max_len=max_len, output_dic=output_dic)
                analysed_batch_dic_src_origin = all_origin[studied_ind]

            for lang, target in data_target_dic.items():
                print(f"Starting target", {lang})
                i_lang += 1
                target_batch = target[i_data:i_data+batch_size]

                all = get_hidden_representation(target_batch, model, tokenizer, pad_below_max_len=pad_below_max_len, max_len=max_len, output_dic=output_dic)

                if args.compare_to_pretrained:
                    all_origin = get_hidden_representation(target_batch, model_origin, tokenizer_0, pad_below_max_len=pad_below_max_len,
                                                           max_len=max_len, output_dic=output_dic)
                    analysed_batch_dic_target_origin = all_origin[studied_ind]

                analysed_batch_dic_target = all[studied_ind]

                former_layer, former_mean_target, former_mean_src = None, None, None
                for layer in analysed_batch_dic_target:
                    print(f"Starting layer", {layer})
                    # get average for sentence removing first and last special tokens
                    if output_dic:
                        mean_over_sent_src = []
                        mean_over_sent_target = []
                        mean_over_sent_target_origin = []
                        mean_over_sent_src_origin = []
                        for i_sent in range(len(analysed_batch_dic_en[layer])):
                            # removing special characters first and last and
                            mean_over_sent_src.append(np.array(analysed_batch_dic_en[layer][i_sent][0, 1:-1, :].mean(dim=0).cpu()))
                            mean_over_sent_target.append(np.array(analysed_batch_dic_target[layer][i_sent][0, 1:-1, :].mean(dim=0).cpu()))

                            if args.compare_to_pretrained:
                                mean_over_sent_target_origin.append(np.array(analysed_batch_dic_target_origin[layer][i_sent][0, 1:-1, :].mean(dim=0).cpu()))
                                if i_lang == 1:
                                    mean_over_sent_src_origin.append(np.array(analysed_batch_dic_src_origin[layer][i_sent][0, 1:-1, :].mean(dim=0).cpu()))

                        if args.compare_to_pretrained:
                            mean_over_sent_target_origin = np.array(mean_over_sent_target_origin)
                            if i_lang == 1:
                                mean_over_sent_src_origin = np.array(mean_over_sent_src_origin)
                        mean_over_sent_src = np.array(mean_over_sent_src)
                        mean_over_sent_target = np.array(mean_over_sent_target)

                    else:
                        mean_over_sent_src = analysed_batch_dic_en[layer][:, 1:-1, :].mean(dim=1).cpu()
                        mean_over_sent_target = analysed_batch_dic_target[layer][:, 1:-1, :].mean(dim=1).cpu()

                    if layer not in cosine_sent_to_src[src_lang+"-"+lang]:
                        cosine_sent_to_src[src_lang+"-"+lang][layer] = []
                    if layer not in cosine_sent_to_origin[lang]:
                        cosine_sent_to_origin[lang][layer] = []
                    if layer not in cosine_sent_to_origin_src[src_lang]:
                        cosine_sent_to_origin_src[src_lang][layer] = []

                    if metric == "cka":
                        mean_over_sent_src = np.array(mean_over_sent_src)
                        mean_over_sent_target = np.array(mean_over_sent_target)

                        cosine_sent_to_src[src_lang+"-"+lang][layer].append(kernel_CKA(mean_over_sent_src, mean_over_sent_target))
                        if args.compare_to_pretrained:
                            cosine_sent_to_origin[lang][layer].append(kernel_CKA(mean_over_sent_target, mean_over_sent_target_origin))
                            if i_lang == 1:
                                cosine_sent_to_origin_src[src_lang][layer].append(kernel_CKA(mean_over_sent_src_origin, mean_over_sent_src))
                                print(f"Measured EN TO ORIGIN {metric} {layer} {cosine_sent_to_origin_src[src_lang][layer][-1]} " + info_model)
                            print(f"Measured LANG {lang} TO ORIGIN {metric} {layer} {cosine_sent_to_origin[lang][layer][-1]} " + info_model)

                        print(f"Measured {metric} {layer} {kernel_CKA(mean_over_sent_src,mean_over_sent_target)} "+info_model)
                    else:
                        cosine_sent_to_src[src_lang+"-"+lang][layer].append(cosinus(mean_over_sent_src, mean_over_sent_target).item())

                    if former_layer is not None:
                        if layer not in cosine_sent_to_former_layer[lang]:
                            cosine_sent_to_former_layer[lang][layer] = []
                        if layer not in cosine_sent_to_former_layer_src[src_lang]:
                            cosine_sent_to_former_layer_src[src_lang][layer] = []
                        if metric == "cka":
                            cosine_sent_to_former_layer[lang][layer].append(kernel_CKA(former_mean_target, mean_over_sent_target))
                            if i_lang == 1:
                                cosine_sent_to_former_layer_src[src_lang][layer].append(kernel_CKA(former_mean_src, mean_over_sent_src))
                        else:
                            cosine_sent_to_former_layer[lang][layer].append(cosinus(former_mean_target, mean_over_sent_target).item())
                            if i_lang == 1:
                                cosine_sent_to_former_layer_src[src_lang][layer].append(cosinus(former_mean_target, mean_over_sent_target).item())

                    former_layer = layer
                    former_mean_target = mean_over_sent_target
                    former_mean_src = mean_over_sent_src



    # summary
    print_all = True
    lang_i = 0
    src_lang_i = 0
    #for lang, cosine_per_layer in cosine_sent_to_src.items():
    for lang, cosine_per_layer in cosine_sent_to_former_layer.items():
        layer_i = 0
        src_lang_i += 1
        for src_lang in src_lang_ls:
            lang_i += 1
            for layer, cosine_ls in cosine_per_layer.items():
                print(f"Mean {metric} between {src_lang} and {lang} for {layer} is {np.mean(cosine_sent_to_src[src_lang+'-'+lang][layer])} std:{np.std(cosine_sent_to_src[src_lang+'-'+lang][layer])} measured on {len(cosine_sent_to_src[src_lang+'-'+lang][layer])} model  "+info_model)
                if layer_i > 0 and print_all:

                    print(f"Mean {metric} for {lang} beween {layer} and former is {np.mean(cosine_sent_to_former_layer[lang][layer])} std:{np.std(cosine_sent_to_former_layer[lang][layer])} measured on {len(cosine_sent_to_former_layer[lang][layer])} model "+info_model)

                    report = report_template(metric_val=metric, subsample=lang + "_to_former_layer",
                                             info_score_val=None,
                                             score_val=np.mean(cosine_sent_to_former_layer[lang][layer]),
                                             n_sents=n_obs,
                                             avg_per_sent=np.std(cosine_sent_to_former_layer[lang][layer]),
                                             n_tokens_score=n_obs * max_len,
                                             model_full_name_val=model_full_name_val, task="hidden_state_analysis",
                                             evaluation_script_val="exact_match",
                                             model_args_dir=args.init_args_dir,
                                             token_type="word",
                                             report_path_val=None,
                                             data_val=layer,
                                             )
                    report_ls.append(report)

                    if lang_i == 1:
                        print(f"Mean {metric} for {lang} beween {layer} and former is {np.mean(cosine_sent_to_former_layer_src[src_lang][layer])} std:{np.std(cosine_sent_to_former_layer_src[src_lang][layer])} measured on {len(cosine_sent_to_former_layer_src[src_lang][layer])} model "+info_model)

                        report = report_template(metric_val=metric, subsample=src_lang + "_to_former_layer",
                                                 info_score_val=None,
                                                 score_val=np.mean(cosine_sent_to_former_layer_src[src_lang][layer]),
                                                 n_sents=n_obs,
                                                 avg_per_sent=np.std(cosine_sent_to_former_layer_src[src_lang][layer]),
                                                 n_tokens_score=n_obs * max_len,
                                                 model_full_name_val=model_full_name_val, task="hidden_state_analysis",
                                                 evaluation_script_val="exact_match",
                                                 model_args_dir=args.init_args_dir,
                                                 token_type="word",
                                                 report_path_val=None,
                                                 data_val=layer,
                                                 )
                        report_ls.append(report)

                layer_i += 1

                report = report_template(metric_val=metric, subsample=lang+"_to_"+src_lang,
                                         info_score_val=None,
                                         score_val=np.mean(cosine_sent_to_src[src_lang+'-'+lang][layer]),
                                         n_sents=n_obs,
                                         #avg_per_sent=np.std(cosine_ls),
                                         avg_per_sent=np.std(cosine_sent_to_src[src_lang+'-'+lang][layer]),
                                         n_tokens_score=n_obs*max_len,
                                         model_full_name_val=model_full_name_val, task="hidden_state_analysis",
                                         evaluation_script_val="exact_match",
                                         model_args_dir=args.init_args_dir,
                                         token_type="word",
                                         report_path_val=None,
                                         data_val=layer,
                                        )

                report_ls.append(report)

                #
                if args.compare_to_pretrained:

                    print(f"Mean {metric} for {lang} beween {layer} and origin model is {np.mean(cosine_sent_to_origin[lang][layer])} std:{np.std(cosine_sent_to_origin[lang][layer])} measured on {len(cosine_sent_to_origin[lang][layer])} model " + info_model)
                    report = report_template(metric_val=metric, subsample=lang + "_to_origin",
                                             info_score_val=None,
                                             score_val=np.mean(cosine_sent_to_origin[lang][layer]),
                                             n_sents=n_obs,
                                             avg_per_sent=np.std(cosine_sent_to_origin[lang][layer]),
                                             n_tokens_score=n_obs * max_len,
                                             model_full_name_val=model_full_name_val, task="hidden_state_analysis",
                                             evaluation_script_val="exact_match",
                                             model_args_dir=args.init_args_dir,
                                             token_type="word",
                                             report_path_val=None,
                                             data_val=layer)
                    report_ls.append(report)

                    if lang_i == 1:

                        print(f"Mean {metric} for en beween {layer} and origin model is {np.mean(cosine_sent_to_origin_src[src_lang][layer])} std:{np.std(cosine_sent_to_origin_src[src_lang][layer])} measured on {len(cosine_sent_to_origin_src[src_lang][layer])} model " + info_model)
                        report = report_template(metric_val=metric, subsample=src_lang + "_to_origin",
                                                 info_score_val=None,
                                                 score_val=np.mean(cosine_sent_to_origin_src[src_lang][layer]),
                                                 n_sents=n_obs,
                                                 avg_per_sent=np.std(cosine_sent_to_origin_src[src_lang][layer]),
                                                 n_tokens_score=n_obs * max_len,
                                                 model_full_name_val=model_full_name_val, task="hidden_state_analysis",
                                                 evaluation_script_val="exact_match",
                                                 model_args_dir=args.init_args_dir,
                                                 token_type="word",
                                                 report_path_val=None,
                                                 data_val=layer)
                        report_ls.append(report)

        # break

    if args.report_dir is None:
        report_dir = PROJECT_PATH +f"/../../analysis/attention_analysis/report/{run_id}-report"
        os.mkdir(report_dir)
    else:
        report_dir = args.report_dir
    assert os.path.isdir(report_dir)
    with open(report_dir + "/report.json", "w") as f:
        json.dump(report_ls, f)

    overall_report = args.overall_report_dir +"/" +args.overall_label +"-grid-report.json"
    with open(overall_report, "r") as g:
        report_all = json.load(g)
        report_all.extend(report_ls)
    with open(overall_report, "w") as file:
        json.dump(report_all, file)

    print("{} {} ".format(REPORT_FLAG_DIR_STR , overall_report))
    # printing("{} {} ", var=[REPORT_FLAG_DIR_STR, report_dir], verbose=verbose, verbose_level=0)


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


