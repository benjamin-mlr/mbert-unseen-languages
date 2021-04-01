import sys
import os
from tqdm import tqdm
import random

# sys.path.append("/Users/bemuller/Documents/Work/INRIA/dev/transfer/")
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", ".."))
from transfer.downstream.finetune.env.flags import REPORT_FLAG_DIR_STR
from transfer.downstream.finetune.io_.report.report_tools import write_args, get_hyperparameters_dict, \
    get_dataset_label, get_name_model_id_with_extra_name

from transfer.downstream.finetune.env.imports import pdb, torch, OrderedDict, np, linear_model, json, uuid4, nn
from transfer.downstream.finetune.model.architecture.get_model import make_bert_multitask
from transfer.analysis.attention_analysis.attention_extraction_utils import get_dirs, load_lang_ls, load_data
from transfer.downstream.finetune.args.args_parse import args_train, args_preprocessing, args_attention_analysis, \
    args_preprocess_attention_analysis
from transfer.downstream.finetune.env.dir.data_dir import DATA_UD_RAW, DATA_UD
from transfer.downstream.finetune.env.dir.project_directories import PROJECT_PATH
from transfer.analysis.attention_analysis.attention_extraction import get_hidden_representation
from transfer.analysis.utils.cka import kernel_CKA
from transfer.analysis.utils.utils_distance import split_train_test, get_closest_centroid, get_stat_distance, get_centroid, concat_all_lang_space_split_train_test, get_closest_n_sent, get_iou_inter, write_down_selected

from sklearn import mixture, cluster
from sklearn.metrics import classification_report, v_measure_score

sys.path.insert(0, os.environ.get("EXPERIENCE","."))
from meta_code.data_dir import get_dir_data


try:
    sys.path.append(os.environ.get("EXPERIENCE"))
    from transfer.write_to_performance_repo import report_template
except:
    from transfer.downstream.finetune.evaluate.score.report_template import report_template

    print("REPORTING modules downloaded from local project ")

from transfer.analysis.utils.load_model_analysis import load_all_analysis

OSCAR = os.environ.get("OSCAR")

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

    lang = ["fr_pud",  # "de_pud", "ru_pud", "tr_pud", "id_pud", "ar_pud", "pt_pud",  "es_pud", "fi_pud",
            # "it_pud", "sv_pud", "cs_pud", "pl_pud", "hi_pud", "zh_pud", "ko_pud", "ja_pud","th_pud"
            ]
    src_lang_ls = ["tr_imst", "en_ewt", #"ja_gsd", "ar_padt",  #"en_pud", "tr_pud", "ru_pud",# "ar_pud", #"de_pud", "ko_pud",
                   "ug_udt"
                    ]  # , "fr_pud", "ru_pud", "ar_pud"]
    src_lang_ls = ["tr_dedup", "az_100k_shuff",
                    "en_100k", "kk_100k_shuff", #"hu_dedup", #"ar_padt",
                   # "en_pud", "tr_pud", "ru_pud",# "ar_pud", #"de_pud", "ko_pud",
                   #"ckb_dedup",# "ja_dedup_200k",
                   #"ar_dedup_200k", "fa_dedup_200k", 
                   "ug_udt",
                   ]
    src_lang_ls = [#"ar_oscar", "tr_dedup", "az_100k_shuff", "fa_dedup_200k",
                   # "it_oscar", "en_oscar", #"hu_dedup", #"ar_padt",
                   "ar_oscar","de_oscar","en_oscar","fa_oscar" ,"fi_oscar" ,"fr_oscar", "he_oscar", "hi_oscar","hu_oscar","it_oscar","ja_oscar", "ko_oscar", "ru_oscar","tr_oscar", 
                   ]
    src_lang_ls.append(args.target_lang)

    def add_demo(src_lang_ls):
        for i in range(len(src_lang_ls)):
            if src_lang_ls[i]!="mt_mudt":
                src_lang_ls[i] += "_demo"
        return src_lang_ls

    #add_demo(src_lang_ls)
    

    # target is last
    target_class_ind = len(src_lang_ls)-1
    target_lang = src_lang_ls[target_class_ind]
    #to_class = [""]
    set_ = "test"
    #set_ = "test-demo"
    #print("Loading data...")

    #data_en = load_data(DATA_UD + f"/{src_lang_ls[0]}-ud-{set_}.conllu", line_filter="# text = ")

    #id_start_start_class, id_end_target_class = get_id_sent_target(target_class_ind, data_target_dic)

    # reg = linear_model.LogisticRegression()
    # just to get the keyw
    layer_all = get_hidden_representation(data, model, tokenizer, max_len=max_len)
    # removed hidden_per_layer

    assert len(layer_all) == 1
    # assert len(layer_all) == 2, "ERROR should only have hidden_per_layer and hidden_per_head_layer"

    report_ls = []
    accuracy_dic = OrderedDict()
    sampling = args.sampling
    metric = args.similarity_metric
    if metric == "cka":
        pad_below_max_len, output_dic = False, True
    else:
        pad_below_max_len, output_dic = False, True
    assert metric in ["cos", "cka"]

    batch_size = args.batch_size #len(data_en) // 4

    task_tuned = "No"

    if args.init_args_dir is None:
        # args.init_args_dir =
        id_model = f"{args.bert_model}-init-{args.random_init}"

        hyperparameters = OrderedDict([("bert_model", args.bert_model),
                                       ("random_init", args.random_init),
                                       ("not_load_params_ls", args.not_load_params_ls),
                                       ("dict_path", dict_path),
                                       ("model_id", id_model), ])
        info_checkpoint = OrderedDict([("epochs", 0), ("batch_size", batch_size),
                                       ("train_path", 0), ("dev_path", 0), ("num_labels_per_task", 0)])

        args.init_args_dir = write_args(os.environ.get("MT_NORM_PARSE", "./"), model_id=id_model,
                                        info_checkpoint=info_checkpoint,
                                        hyperparameters=hyperparameters, verbose=1)
        print("args_dir checkout ", args.init_args_dir)
        model_full_name_val = task_tuned + "-" + id_model
    else:
        argument = json.load(open(args.init_args_dir, 'r'))
        task_tuned = argument["hyperparameters"]["tasks"][0][0] if not "wiki" in argument["info_checkpoint"][
            "train_path"] else "ner"
        model_full_name_val = task_tuned + "-" + args.init_args_dir.split("/")[-1]

    if args.analysis_mode == "layer":
        studied_ind = 0
    elif args.analysis_mode == "layer_head":
        studied_ind = 1
    else:
        raise (Exception(f"args.analysis_mode : {args.analysis_mode} corrupted"))


    output_dic = True
    pad_below_max_len = False
    max_len = 500

    sent_embeddings_per_lang = OrderedDict()
    sent_text_per_lang = OrderedDict()
    pick_layer = ["layer_6"]
    n_batch = args.n_batch
    #assert n_batch==1, "ERROR not working otherwise ! "
    demo = 0
    assert args.n_sent_extract <= args.batch_size * args.n_batch * (len(src_lang_ls) - 1), "ERROR not enough data provided for the selection"
    
    print(f"Starting processing : {n_batch} batch of size {batch_size}")

    def sanity_len_check(src_lang_ls, n_sent_per_lang):
        for src_lang in src_lang_ls:
            
            dir_data = OSCAR + f"/{src_lang}-train.txt"
            num_lines = sum(1 for line in open(dir_data))
            print(f"Sanity checking {src_lang} should have more than {n_sent_per_lang} sentences, it has {num_lines}")
            assert num_lines>=n_sent_per_lang, f"ERROR {src_lang} {num_lines} < {n_sent_per_lang} n_sent_per_lang"
    
    sanity_len_check(src_lang_ls[:-1], n_sent_per_lang=args.batch_size * args.n_batch)



    for i_data in tqdm(range(n_batch)):
        if demo:
            batch_size = 50
            n_batch = 1
            if i_data > 0:
                break
        for src_lang in tqdm(src_lang_ls):
            print(f"Loading lang {src_lang} batch size {batch_size}")
            
            #data_en = load_data(DATA_UD + f"/{src_lang}-ud-{set_}.conllu", line_filter="# text = ")
            #en_batch =  # data[i_data:i_data + batch_size]
            try:
                dir_data = get_dir_data(set="train", data_code=src_lang)
                filter_row = "# text = "
            except Exception as e:
                dir_data = OSCAR + f"/{src_lang}-train.txt"
                filter_row = ""
                print(f"{src_lang} not supported or missing : data defined as {dir_data} filter empty")
            try:
                en_batch = load_data(dir_data, line_filter=filter_row, id_start=i_data*batch_size, id_end=(i_data+1)*batch_size)
            except Exception as e:
                print(f"ERROR: cannot load data {dir_data} skipping")
                if i_data==0:
                    raise(Exception(e))
                continue
            if en_batch is None:
                print(f"lang {src_lang} reading {i_data*batch_size} seems empty so skipping")
                continue

            if src_lang not in sent_text_per_lang:
                sent_text_per_lang[src_lang] = []
            sent_text_per_lang[src_lang].extend(en_batch)

            all = get_hidden_representation(en_batch, model, tokenizer, pad_below_max_len=pad_below_max_len,
                                            max_len=max_len, output_dic=output_dic)

            analysed_batch_dic_en = all[studied_ind]
            i_lang = 0

            if args.compare_to_pretrained:
                all_origin = get_hidden_representation(en_batch, model_origin, tokenizer_0,
                                                       pad_below_max_len=pad_below_max_len,
                                                       max_len=max_len, output_dic=output_dic)
                analysed_batch_dic_src_origin = all_origin[studied_ind]

            for layer in analysed_batch_dic_en:
                if layer not in pick_layer:
                    continue
                else:
                    print(f"Picking {pick_layer} layer")
                print(f"Starting layer", {layer})
                # get average for sentence removing first and last special tokens
                if layer not in sent_embeddings_per_lang:
                    sent_embeddings_per_lang[layer] = OrderedDict()
                if src_lang not in sent_embeddings_per_lang[layer]:
                    sent_embeddings_per_lang[layer][src_lang] = []
                if output_dic:
                    mean_over_sent_src = []
                    #mean_over_sent_target = []
                    #mean_over_sent_target_origin = []
                    mean_over_sent_src_origin = []
                    for i_sent in range(len(analysed_batch_dic_en[layer])):
                        # removing special characters first and last and
                        mean_over_sent_src.append(
                            np.array(analysed_batch_dic_en[layer][i_sent][0, 1:-1, :].cpu().mean(dim=0)))
                        #mean_over_sent_target.append(
                        #    np.array(analysed_batch_dic_target[layer][i_sent][0, 1:-1, :].mean(dim=0)))
                        if args.compare_to_pretrained:
                        #    mean_over_sent_target_origin.append(
                        #        np.array(analysed_batch_dic_target_origin[layer][i_sent][0, 1:-1, :].mean(dim=0)))
                            if i_lang == 1:
                                mean_over_sent_src_origin.append(
                                    np.array(analysed_batch_dic_src_origin[layer][i_sent][0, 1:-1, :].mean(dim=0)))
                    if args.compare_to_pretrained:
                    #    mean_over_sent_target_origin = np.array(mean_over_sent_target_origin)
                        if i_lang == 1:
                            mean_over_sent_src_origin = np.array(mean_over_sent_src_origin)
                    mean_over_sent_src = np.array(mean_over_sent_src)
                    #mean_over_sent_target = np.array(mean_over_sent_target)
                else:
                    mean_over_sent_src = analysed_batch_dic_en[layer][:, 1:-1, :].mean(dim=1)
                    #mean_over_sent_target = analysed_batch_dic_target[layer][:, 1:-1, :].mean(dim=1)

                sent_embeddings_per_lang[layer][src_lang].append(mean_over_sent_src)

    def get_id_sent_target(target_class_ind, data_target_dic):
        n_sent_total = 0

        assert target_class_ind <= len(data_target_dic)
        for ind_class, lang in enumerate(src_lang_ls):
            n_sent_total += len(data_target_dic[lang])
            if ind_class == target_class_ind:
                n_sent_class = len(data_target_dic[lang])
                id_start_start_class = n_sent_total
                id_end_target_class = n_sent_total + n_sent_class
        return id_start_start_class, id_end_target_class

    clustering = "distance"

    if clustering in ["gmm", "spectral"]:
        concat_train,  concat_test, y_train, y_test, lang2id = concat_all_lang_space_split_train_test(sent_embeddings_per_lang, src_lang_ls, pick_layer)
        #X = np.array(concat).squeeze(1)
        X_train = np.array(concat_train)
        X_test = np.array(concat_test)

        if len(X_train.shape) > 2:

            X_train = X_train.reshape(X_train.shape[0]*X_train.shape[1],-1)
            X_test = X_test.reshape(X_test.shape[0]*X_test.shape[1],-1)
        if clustering == "gmm":
            model = mixture.GaussianMixture(n_components=len(src_lang_ls)-1, covariance_type='full')
            model.fit(X_train)
            model_based_clustering = True
        elif clustering == "spectral":
            model = cluster.spectral_clustering(n_clusters=len(src_lang_ls))
            model.fit(X_train)
            model_based_clustering = True

    elif clustering == "distance":
        # concat batch_size

        for layer in sent_embeddings_per_lang:
            assert len(sent_embeddings_per_lang[layer])>1, "ERRO you're doing distance measure ! "
            for lang in sent_embeddings_per_lang[layer]:
                arr = np.array(sent_embeddings_per_lang[layer][lang])
                if arr.shape[0]!=n_batch:
                    print(f"WARNNIG: shape: {lang}  {np.array(sent_embeddings_per_lang[layer][lang]).shape} reshaping to {arr.shape[0]*arr.shape[1]}")
                sent_embeddings_per_lang[layer][lang] = arr.reshape(arr.shape[0] * arr.shape[1], -1)
            assert sent_embeddings_per_lang[layer][lang].shape[0] == len(sent_text_per_lang[lang]), f"ERROR lang {lang} layer {layer}  {sent_embeddings_per_lang[layer][lang].shape}[0]<>{len(sent_text_per_lang[lang])}"

        sent_embeddings_per_lang_train, sent_embeddings_per_lang_test, sent_text_per_lang = \
            split_train_test(sent_embeddings_per_lang, sent_text_per_lang,
                             keep_text_test=True, target_lang=target_lang,
                             target_lang_no_test=True,
                             prop_train=1 / 20)

        centroid_train, ls_lang = get_centroid(sent_embeddings_per_lang_train, target_lang=target_lang, only_target_centoid=False)
        # outputing for each sentence (with layer x lang information)
        print("ls_lang", ls_lang)
        closest_lang, score_to_target_test = get_closest_centroid(sent_embeddings_per_lang_test, centroid_train, ls_lang, ind_lang_target=target_class_ind)

        get_stat_distance(closest_lang, ls_lang, target_lang)

        count_n_extracted_sent = 0
        for layer in score_to_target_test:
            for lang in score_to_target_test[layer]:
                count_n_extracted_sent += len(score_to_target_test[layer][lang])
        print(f"Cosine extracted sent {count_n_extracted_sent}")
        test_sent_extracted, index_test_extraxted, info_per_layer_select = get_closest_n_sent(n_sent=args.n_sent_extract, score_to_target=score_to_target_test, sent_text_per_lang=sent_text_per_lang, lang_ls=src_lang_ls,
                                                                                              target_lang=target_lang)
        get_iou_inter(index_test_extraxted)


        dir_file = os.path.join(os.environ.get("OSCAR", "/Users/bemuller/Documents/Work/INRIA/dev/data"),"data_selected")
        #dir_file = "/Users/bemuller/Documents/Work/INRIA/dev/data/data_selected"
        write_down_selected(test_sent_extracted, info_per_layer_select, dir_file, id=f"select-{args.overall_label}-{args.bert_model}-{target_lang}-n_sent-{args.n_sent_extract}")


    if clustering in ["gmm", "spectral"]:
        target_class_ind = X_train
        predict_proba_train = model.predict_proba(X_train)
        predict_train = model.predict(X_train)
        predict_proba = model.predict_proba(X_test)
        predict_test = model.predict(X_test)

        def get_most_common_per_class(predict, lang2id):
            " for each class : finding the clustering predicting using majority vote "
            id_class_start = 0
            id_class_end = 0
            pred_label_to_real_label = {}
            for lang in lang2id:
                id_class_end += lang2id[lang]["n_sent_train"]

                pred_class = predict[id_class_start:id_class_end]

                assert len(pred_class)>0
                id_class_start = id_class_end
                from collections import Counter
                pred_class_counter = Counter(pred_class)
                lang2id[lang]["pred_label"] = pred_class_counter.most_common()[0][0]
                if pred_class_counter.most_common()[0][0] in pred_label_to_real_label:
                    print(f"WARNING: {pred_class_counter.most_common()[0][0]} pred label as mot_common in a class is predicted in two classes")
                pred_label_to_real_label[pred_class_counter.most_common()[0][0]] = lang2id[lang]["id"]
            return lang2id, pred_label_to_real_label

        lang2id, pred_label_to_real_label = get_most_common_per_class(predict_train, lang2id)
        print(f"V metric train {v_measure_score(predict_train, y_train)}")
        print(f"V metric test {v_measure_score(predict_test, y_test)}")

        def adapt_label(pred_label_to_real_label, pred):
            " based on majority bvote prediction : adapt prediction set to real label set"
            pred_new = []
            for label_pred in pred:
                if label_pred not in pred_label_to_real_label:
                    print("Warning : pred label not associated to any true label")
                pred_new.append(pred_label_to_real_label.get(label_pred, label_pred))
            return pred_new

        def print_report(report, src_lang_ls, lang2id):
            for lang in src_lang_ls:
                id_label = lang2id[lang]["id"]
                print(f"Lang {lang} summary {report[str(id_label)]}")

            print(f"Macro Avg {lang} summary {report['macro avg']}")

        pred_new_train = adapt_label(pred_label_to_real_label, predict_train)
        report = classification_report(y_pred=pred_new_train, y_true=y_train, output_dict=True)
        print_report(report, src_lang_ls, lang2id)

        pred_new_test = adapt_label(pred_label_to_real_label, predict_test)
        report = classification_report(y_pred=pred_new_test, y_true=y_test, output_dict=True)

        print_report(report, src_lang_ls, lang2id)

        #print(predict_proba_train, predict_proba)


    #print(gmm.predict(X_len(-train),
    #gmm.predict_proba(X[:1, :]))

    # based on this --> for a given source set of sentences (ex : uyghur sentences)
    # 1 - find the cluster id of Uyghur sentences
    # 2 - get the x top sentences that have high proba for
    # 3 - print it to see if that makes sense
    # do it for 1000 uy , 1000k for 10 other languages
    # then same
    # then compare overlap per layers


    # summary
    print_all = True
    lang_i = 0
    src_lang_i = 0
    # for lang, cosine_per_layer in cosine_sent_to_src.items():

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


