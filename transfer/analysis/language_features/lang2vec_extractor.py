from collections import OrderedDict
import numpy as np
import pdb
import lang2vec.lang2vec as l2v

def get_feature(src_lang="ug",lang=["eng"], prop="syntax_wals", printout=False, missing_accounting=False):
    readable = OrderedDict()
    if lang=="all":
        lang = list(l2v.available_languages())
    #elif src_lang is not None:
    lang.append(src_lang)

    dic = l2v.get_features(lang, prop,header=True)
    code = dic["CODE"]

    for key in dic:
        if key != "CODE":
            readable[key] = [(_code, val) for _code, val in zip(code,dic[key]) if val == 1]
            dic[key] = np.array(dic[key])
    if printout:
        print(readable)
    
    if src_lang is not None:
        intersect_with_src_rate= OrderedDict()
        intersect_with_src_ls = OrderedDict()
        disjoin_in_target_rate = OrderedDict()
        disjoin_with_lang = OrderedDict()
        disjoin_in_target = OrderedDict()
        intersect_over_union = OrderedDict()
        for _lang in lang:

            if missing_accounting:
                intersect_with_src_rate[_lang+"-"+src_lang] = np.round(np.sum(dic[src_lang][dic[src_lang]==dic[_lang]]=="1.0")/len(dic[src_lang]),decimals=3)
                intersect_with_src_ls[_lang+"-"+src_lang] = [(_code, val) for _code, val, val_2 in zip(code,dic[src_lang], dic[_lang]) if val == 1 and val==val_2]
            else:
                intersect_over_union[_lang+"-"+src_lang] = np.round(np.sum(dic[src_lang][dic[src_lang]==dic[_lang]]=="1.0")/len(dic[src_lang]),decimals=3)
                intersect_with_src_rate[_lang+"-"+src_lang] = np.round(np.sum(dic[src_lang][dic[src_lang]==dic[_lang]]=="1.0")/np.sum(dic[src_lang]=="1.0"),decimals=3)
                intersect_with_src_ls[_lang+"-"+src_lang] = [_code for _code, val, val_2 in zip(code,dic[src_lang], dic[_lang]) if val_2==val and val=="1.0"]
                # hard disjoint : if target is 1 and src is UNK --> considered disjoint
                disjoin_in_target[_lang+"-"+src_lang] = [_code for _code, val, val_2 in zip(code,dic[src_lang], dic[_lang]) if val_2!=val and val_2=="1.0"]
                disjoin_in_target_rate[_lang+"-"+src_lang]  = np.round(np.sum(dic[_lang][dic[src_lang]!=dic[_lang]]=="1.0")/np.sum(dic[_lang]=="1.0"),decimals=3)
                #pdb.set_trace()
                #np.round(np.sum(dic[src_lang][dic[src_lang]!=dic[_lang]]=="1.0")/np.sum(dic[src_lang]=="1.0"),decimals=3)
                # hard disjoint : if src is 1 and target is UNK --> considered disjoint
                disjoin_with_lang[_lang+"-"+src_lang] = [_code for _code, val, val_2 in zip(code,dic[src_lang], dic[_lang]) if val_2!=val and val=="1.0"]
                #npreadable[lang]
    return readable, dic, intersect_with_src_rate, intersect_with_src_ls, disjoin_in_target, disjoin_in_target_rate, disjoin_with_lang, intersect_over_union

def get_top_rate(intersect_with_src_rate,intersect_with_src_ls, disjoint_with_src_ls, disjoin_with_lang,
	iso2lang, topk=10, printout_prop=False):
    top_10 = np.argsort(np.array(list(intersect_with_src_rate.values())))[::-1][:topk]
    langs_top = np.array(list(intersect_with_src_rate.keys()))[top_10]

    if printout_prop:
        for rank, (lang, prop) in enumerate(zip(langs_top,top_10)):
            inter=intersect_with_src_rate[lang]
            target, src = lang.split("-")[0],lang.split("-")[1]
            label = iso2lang.get(target,target)
            print("\\")
            print("Intersection with {} ranked {}  ranked {} intersection rate {} ".format( label,lang, rank+1 , inter))
            print("INTER", intersect_with_src_ls[lang],len(intersect_with_src_ls[lang]), src, target)
            print("DISJOINT ({} as 1)".format(lang),disjoint_with_src_ls[lang])
            print("DISJOINT (lang)".format(lang),disjoin_with_lang[lang])
    return top_10, langs_top
