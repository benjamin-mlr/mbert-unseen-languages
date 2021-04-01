from collections import OrderedDict
import numpy as np
from scipy import spatial
import pdb


def split_train_test(sent_embeddings_per_lang, sent_text_per_lang, target_lang, target_lang_no_test=True, keep_text_test=False, prop_train=9/10):
    sent_embeddings_per_lang_train = OrderedDict()
    sent_embeddings_per_lang_test = OrderedDict()
    text_test = None
    print(f"Splitting train and test of each representation set (lang x layer) {prop_train}% training sentences")
    for ind, layer in enumerate(sent_embeddings_per_lang):
        for lang in sent_embeddings_per_lang[layer]:
            # sent_embeddings_per_lang["layer_6"]["tr_imst"]
            if layer not in sent_embeddings_per_lang_train:
                sent_embeddings_per_lang_train[layer] = OrderedDict()
                sent_embeddings_per_lang_test[layer] = OrderedDict()
            if lang not in sent_embeddings_per_lang_train[layer]:
                sent_embeddings_per_lang_train[layer][lang] = OrderedDict()
                sent_embeddings_per_lang_test[layer][lang] = OrderedDict()
            n_sent = len(sent_embeddings_per_lang[layer][lang])
            
            if n_sent * (1 - prop_train) ==0:
                prop_train = 1 / 2
                print(f"Splitting rate updated to {prop_train}")
                assert n_sent * (1 - prop_train) > 0, "ERROR"
            if lang != target_lang or not target_lang_no_test:
                sent_embeddings_per_lang_train[layer][lang] = np.array(sent_embeddings_per_lang[layer][lang])[0:int(n_sent * prop_train), :]
                sent_embeddings_per_lang_test[layer][lang] = np.array(sent_embeddings_per_lang[layer][lang])[int(n_sent * prop_train):, :]

            else:
                sent_embeddings_per_lang_train[layer][lang] = np.array(sent_embeddings_per_lang[layer][lang])
                sent_embeddings_per_lang_test[layer][lang] = np.array(sent_embeddings_per_lang[layer][lang])[:1, :]
            print(f"TRAIN {lang} split to {sent_embeddings_per_lang_train[layer][lang].shape} ")
            print(f"TEST {lang} split to {sent_embeddings_per_lang_test[layer][lang].shape} ")
            if ind == 0 and keep_text_test:
                sent_text_per_lang[lang] = sent_text_per_lang[lang][int(n_sent * prop_train):]

            if lang != target_lang or not target_lang_no_test:
                assert len(sent_embeddings_per_lang_train[layer][lang].shape) > 0
            assert len(sent_embeddings_per_lang_test[layer][lang].shape) > 0
        text_test = sent_text_per_lang
        if target_lang_no_test:
            assert sent_embeddings_per_lang_test[layer][target_lang].shape[0]==1, "ERROR : test ug should be 1 : only keep "

    return sent_embeddings_per_lang_train, sent_embeddings_per_lang_test, text_test


def get_closest_centroid(sent_embeddings_per_lang, centroid, ls_lang, ind_lang_target):

    closest_lang = OrderedDict()
    dist_to_target = OrderedDict()
    print(f"Lang target is {ls_lang[ind_lang_target]}")
    for layer in sent_embeddings_per_lang:
        if layer not in closest_lang:
            closest_lang[layer] = OrderedDict()
            dist_to_target[layer] = OrderedDict()
        for lang in sent_embeddings_per_lang[layer]:
            if lang not in closest_lang[layer]:
                closest_lang[layer][lang] = []
                dist_to_target[layer][lang] = []
            for sent in sent_embeddings_per_lang[layer][lang]:
                # i index of the languag on which the centroid is computed
                # lang language of the sentnece
                cosine_mat = np.array([spatial.distance.cosine(sent, centroid[layer][i, :]) for i in range(centroid[layer].shape[0])])
                print(f"selecting cosine shape {cosine_mat.shape}")
                lang_closest = ls_lang[np.argmin(cosine_mat)]
                score_to_target = cosine_mat[ind_lang_target]
                if lang_closest == lang:
                    lang_closest = ls_lang[np.argsort(cosine_mat)[1]]
                #else:
                #    #print(f"sentence closest to lang {lang_closest} than to its own lang {lang}")
                closest_lang[layer][lang].append((lang_closest, cosine_mat))
                dist_to_target[layer][lang].append(score_to_target)

    return closest_lang, dist_to_target


def get_centroid(sent_embeddings_per_lang,target_lang, output_as_array=True, n_dim_representation=768, only_target_centoid=False):
    centroid_vec = OrderedDict()
    for layer in sent_embeddings_per_lang:
        for lang in sent_embeddings_per_lang[layer]:
            # sent_embeddings_per_lang["layer_6"]["tr_imst"]
            if layer not in centroid_vec:
                centroid_vec[layer] = OrderedDict()
            if lang not in centroid_vec[layer]:
                if only_target_centoid and lang!=target_lang:
                    continue
                centroid_vec[layer][lang] = OrderedDict()
            try:
                if lang == target_lang:
                    print(f"Centroid on target {target_lang} computed on {np.array(sent_embeddings_per_lang[layer][lang]).shape[0]} sentences")
                centroid_vec[layer][lang] = np.array(sent_embeddings_per_lang[layer][lang]).mean(axis=0, keepdims=1)

            except:
                assert np.array(sent_embeddings_per_lang[layer][lang]).shape[0] == 1
                centroid_vec[layer][lang] = np.array(sent_embeddings_per_lang[layer][lang])
            try:
                assert centroid_vec[layer][lang].shape[1] == n_dim_representation, "ERROR dim centroid"
            except Exception as e:
                print(f"-->  centroid_vec[layer][lang].shape[1] {e} {n_dim_representation} ")
            assert len(centroid_vec[layer][lang].shape) == 2

    if output_as_array:
        #assert len(centroid_vec) == 1, "ERROR 1 layer representation supported here "
        array_centroid = OrderedDict()
        for layer in centroid_vec:
            ls_lang = list(centroid_vec[layer].keys())
            array_centroid[layer] = np.array(list(centroid_vec[layer].values())).squeeze(1)
        assert len(ls_lang) == array_centroid[layer].shape[0]
        assert n_dim_representation == array_centroid[layer].shape[1]
        return array_centroid, ls_lang

    return centroid_vec, None


def get_stat_distance(closest_lang, ls_lang, target_lang, verbose=1):
    stat = OrderedDict()
    for layer in closest_lang:
        for lang in closest_lang[layer]:
            for sent_info in closest_lang[layer][lang]:
                # pdb.set_trace()
                for i_lang, lang_measure in enumerate(ls_lang):
                    key = f"{layer}-{lang}-to-centroid_{lang_measure}"
                    if key not in stat:
                        stat[key] = []
                    # pdb.set_trace()
                    stat[key].append(sent_info[1][i_lang])
    # pdb.set_trace()
    stat_new = OrderedDict()
    for key, val in stat.items():
        mean = np.mean(np.array(val))
        var = np.var(np.array(val))
        assert key not in stat_new
        stat_new[key] = OrderedDict()
        stat_new[key]["mean"] = mean
        stat_new[key]["var"] = var
        stat_new[key]["max"] = np.max(np.array(val))
        stat_new[key]["min"] = np.min(np.array(val))
        stat_new[key]["count"] = len(val)

        if verbose >= 2:
            print(f"Distance {key} mean:{mean} var:{var} count:{len(val)}")
        if verbose == 1 and f"centroid_{target_lang}" in key:
            print(f"Distance {key} mean:{mean} var:{var} count:{len(val)} max {stat_new[key]['max']} min {stat_new[key]['min']}")

    return stat_new


def concat_all_lang_space_split_train_test(sent_embeddings_per_lang, src_lang_ls, pick_layer):
    concat_train = []
    concat_test = []
    y_train = []
    y_test = []
    lang2id = {}
    for class_id, src_lang in enumerate(src_lang_ls):
        n_sent = len(sent_embeddings_per_lang[pick_layer][src_lang][0])
        if n_sent > 5:
            n_train = int(n_sent * 9 / 10)
            concat_train.extend(sent_embeddings_per_lang[pick_layer][src_lang][0][:n_train])
            concat_test.extend(sent_embeddings_per_lang[pick_layer][src_lang][0][n_train:])

            y_train.extend([class_id for _ in sent_embeddings_per_lang[pick_layer][src_lang][0][:n_train]])
            n_test = n_sent-n_train
            y_test.extend([class_id for _ in sent_embeddings_per_lang[pick_layer][src_lang][0][n_train:]])
        else:
            print("No diff train test")
            concat_train.extend(sent_embeddings_per_lang[pick_layer][src_lang])
            concat_test.extend(sent_embeddings_per_lang[pick_layer][src_lang])
            y_train.extend(
                [class_id for _ in sent_embeddings_per_lang[pick_layer][src_lang]])
            y_test.extend(
                [class_id for _ in sent_embeddings_per_lang[pick_layer][src_lang]])
            n_train = n_sent
            n_test = n_sent

        if src_lang not in lang2id:
            lang2id[src_lang] = {}
        lang2id[src_lang]["id"] = class_id
        lang2id[src_lang]["n_sent_test"] = n_test
        lang2id[src_lang]["n_sent_train"] = n_train
    return concat_train,  concat_test, y_train, y_test, lang2id


def get_closest_n_sent(n_sent, score_to_target, lang_ls, target_lang, sent_text_per_lang, select_target=False, verbose=2):
    print(f"Select {n_sent} sentences closest to {target_lang} representations")
    all_score = OrderedDict()
    n_sent_per_lang = OrderedDict()
    index_dic_per_layer = OrderedDict()
    info_per_layer_select = OrderedDict()
    sent_text_per_lang_out = OrderedDict()
    former_len = -1
    
    for layer_select in score_to_target:
        scores = score_to_target[layer_select]
        if layer_select not in all_score:
            all_score[layer_select] = []
        for lang, lang_score_ls in scores.items():
            n_sent_per_lang[lang] = len(lang_score_ls)
            if former_len != -1:
                # if select target false :
                # we allow the number of sentences for the target
                # to be different as they are not included in the scores list
                if select_target or lang != target_lang:
                    assert former_len == len(lang_score_ls), "we're doing modulo so we need the same sent "
                    n_sent_per_lang_bucket = len(lang_score_ls)
            former_len = len(lang_score_ls)
            if lang != target_lang or select_target:
                all_score[layer_select].extend(lang_score_ls)
        # concatanate all scores in the order of lang_ls
        all_score[layer_select] = np.array(all_score[layer_select])
        sorted_score = np.argsort(all_score[layer_select])
        index_dic = OrderedDict()
        for ind, score_ind in enumerate(sorted_score):
            index_sent = score_ind % n_sent_per_lang_bucket
            lang_sent = score_ind // n_sent_per_lang_bucket

            if lang_ls[lang_sent] not in index_dic:
                index_dic[lang_ls[lang_sent]] = []
            index_dic[lang_ls[lang_sent]].append(index_sent)
            if ind >= n_sent:
                break # return index_dic
        if layer_select not in sent_text_per_lang_out:
            sent_text_per_lang_out[layer_select] = OrderedDict()
        for lang in sent_text_per_lang:
            #pdb.set_trace()
            sent_text_per_lang_out[layer_select][lang] = list(np.array(sent_text_per_lang[lang])[index_dic[lang]]) if lang in index_dic else []
            if len(sent_text_per_lang_out[layer_select][lang]) == 0:
                print(f"No sentences extracted from {lang}")
            else:
                info = f"{len(sent_text_per_lang_out[layer_select][lang])} extracted from lang {lang} layer {layer_select}"
                #print(info)
                if layer_select not in info_per_layer_select:
                    info_per_layer_select[layer_select] = []
                info_per_layer_select[layer_select].append(info)

        index_dic_per_layer[layer_select] = index_dic

    return sent_text_per_lang_out, index_dic_per_layer, info_per_layer_select


def get_iou_inter(index_test_extraxted):
    for layer in index_test_extraxted:
        for layer2 in index_test_extraxted:
            for lang in index_test_extraxted[layer]:
                int = len(set(index_test_extraxted[layer].get(lang, [])) & set(index_test_extraxted[layer2].get(lang, [])))
                union = len(set(index_test_extraxted[layer].get(lang, [])).union(set(index_test_extraxted[layer2].get(lang, []))))
                print(f"Inter {layer} {layer2} for lang {lang} is {int / union} ")


def write_down_selected(test_sent_extracted, info_per_layer_select, dir_file, id):
    for layer, sent_dic in test_sent_extracted.items():
        with open(f"{dir_file}/{id}-select_mix-{layer}-info.txt", "w") as readme:
            for info in info_per_layer_select[layer]:
                readme.write(info+"\n")
        file = f"{dir_file}/{id}-select_mix-{layer}.txt"
        with open(file,"w") as f:
            for lang, sents in sent_dic.items():
                for sent in sents:
                    f.write(sent+"\n")
                print(f"{len(sents)} added to {file} from lang {lang} ")
        print(f"{file} created")
