import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", ".."))

import json
import numpy as np
import string
import pdb
from tqdm import tqdm
from collections import OrderedDict
from lang_trans.arabic import buckwalter
from transliterate import translit
import pykakasi
from uuid import uuid4
import random
from transfer.transliteration.permutation import get_permute
#from transformers
import os
#import pinyin


def translit_string(script_src, script_trg, line, perm):
    
    if script_src == "latin" and script_trg == "ar":
        new_line = buckwalter.untransliterate(line)
    elif script_trg == "latin" and script_src == "ar":
        new_line = buckwalter.transliterate(line)
    # ka is Georgian
    elif script_src == "ka" and script_trg == "latin":
        new_line = translit(line, script_src, reversed=True)
    elif script_src == "ru" and script_trg == "latin":
        new_line = translit(line, script_src, reversed=True)
    elif script_trg == "latin" and script_src == "el":
        new_line = translit(line, script_src, reversed=True)
    elif script_trg == "ru" and script_src == "latin":
        new_line = translit(line, script_trg)
    elif script_trg == "latin" and script_src == "ja":
        kks = pykakasi.kakasi()
        
        new_line = " ".join([word["hepburn"] for word in kks.convert(line)])
    elif script_trg == "permute":
        #assert script_src == "latin"
        assert perm is not None
        line = list(line)
        new_line = "".join([perm.get(c, c) for c in line])
    else:
        raise (Exception(f"not supported {script_src}-->{script_trg}"))
    return new_line


def transliterate(src, target, script_src, script_trg, perm=None,
                  break_n_sent=None, type="raw"):
    with open(src, encoding='utf8', mode="r") as f:
        with open(target, encoding='utf8', mode="w") as tr:
            len_line = []
            count_short_sent = 0
            count_sent = 0
            n_line = 0
            for line in tqdm(f):
                line = line.strip()
                if type == "conll":
                    if len(line) == 0:
                        new_sent = 1
                        #token_count = 0
                        count_sent += 1
                        tr.write("\n")
                        n_line += 1

                        # target.write("# sent {} \n".format(count_sent))
                    elif line.startswith("#"):
                        tr.write(line+"\n")
                        n_line += 1
                    else:
                        #token_count += 1
                        # print(line)
                        line = line.strip()
                        
                        # print(line)
                        # pdb.set_trace()
                        if "-" in line[0]:
                            tr.write(line+"\n")
                            continue
                        line = line.split("\t")
                        tr.write("{id}\t{new_form}\t{lemma}\t{POS}\t{XPOS}\t_\t{head}\t{type}\t_\toriginal_form={original_form}\n".format(
                            id=line[0], new_form=translit_string(script_src, script_trg, line[1], perm),  # line[1],
                            POS=line[3],XPOS=line[4], head=line[6], type=line[7], lemma=line[2],original_form=line[1]))

                elif type == "raw":
                    new_line = translit_string(script_src, script_trg, line, perm)
                    tr.write(new_line + "\n")
                else:
                    raise(Exception("not supported"))

                count_sent += 1
                if break_n_sent and count_sent > break_n_sent:
                    print("BREAKING early ")
                    break

    print(f"{target} written from {src} with transformation {script_src}-->{script_trg}")
    n_tgt = sum(1 for _ in open(target))
    n_src = sum(1 for _ in open(src))
    try:
        assert n_tgt == n_src, f"src: {n_src} <> {n_tgt}:tgt"
        print("As many target lines than src lines")
    except Exception as e:
        print("WARNING: some lines got broken ",e)



if __name__ == "__main__":

    #perm = get_permute()
    #print(perm)
    dir = os.environ.get("OSCAR")
    dir = os.environ.get("WIKIPEDIA")

    lang = "mt_mudt"

    for set in []:
        file = f"{dir}/{lang}_oscar-{set}.txt"
        trg = f"{dir}/{lang}_perm_1_oscar-{set}.txt"
        perm = get_permute()

        transliterate(file, target=trg, script_src="latin", script_trg="permute", perm=perm)#, break_n_sent=1000)
        perm = get_permute()
        trg = f"{dir}/{lang}_perm_2_oscar-{set}.txt"
        transliterate(file, target=trg, script_src="latin", script_trg="permute", perm=perm)#, break_n_sent=1000)
        trg = f"{dir}/{lang}_ar_script_oscar-{set}.txt"
        transliterate(file, target=trg, script_src="latin", script_trg="ar")#, break_n_sent=1000)

    ## creating permutation and dumping 
    #char_set = json.load(open("./char_set_xlmr.json","r"))
    #perm= get_permute(alphanumerical_char=char_set)
    #uid=str(uuid4())[:5]
    #dir_perm=f"./char_set_xlmr_perm_{uid}.json"
    #json.dump(perm,open(dir_perm,"w",encoding="utf-8"), ensure_ascii=False, indent=4)

    ## loading 
    uid = "63d0b"
    #dir_perm=f"./char_set_xlmr_perm_{uid}.json"
    #perm = json.load(open(dir_perm,"r",encoding="utf-8"))
    #print(f"{dir_perm} written")
    #perm=None
    #target = "permute"
    label = "latin"

    #for lang, script in zip(["ru_gsd", "ar_padt", "el_gdt", "ja_gsd"], ["ru","ar", "el", "ja"]):
    #for lang, script in zip(["ja_wiki","ru_wiki"], ["ja","ru"]):
    for lang, script in zip(['ja_wiki'], ['ja']):
        for set in ["train", "test", "dev"]:
            corpus = "wikiner"
            #corpus = "ud"
            #perm = get_permute ()
            dir = os.environ.get("DATA_UD")+"/../wikiner-new"
            file = f"{dir}/{lang}-{corpus}-{set}.conll"
            if not os.path.isfile(file):
                print(f"File does not exist {file}")
                continue
            trg = f"{dir}/{lang}_{label}-{corpus}-{set}.conll"
            print(trg)
            #transliterate(file, target=trg, script_src="ar", script_trg="latin", type="conll")#, perm=perm)  # , break_n_sent=1000)
            print("trg is", trg)
            transliterate(file, target=trg, script_src=script, script_trg="latin", type="conll")
            #transliterate(file, target=trg, script_src=script, script_trg="permute", type="conll", perm=perm)

        print(f"{lang}_{label} done in {corpus} project transliterated ")

    lang = "bx"
    
    for lang in []:
        suff="wiki"
        for set in ["train", "test"]:
            file = f"{dir}/{lang}_{suff}-{set}.txt"
            trg = f"{dir}/{lang}_latin_{suff}-{set}.txt"
            #perm = get_permute()
            transliterate(file, target=trg, script_src="ka", script_trg="latin")#, break_n_sent=50000)


