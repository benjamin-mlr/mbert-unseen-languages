import os
from shutil import copyfile


def wiki_ann_to_tsv(src_dir, target_dir):

    for type in ["train", "dev", "test"]:
        src_file = src_dir + "/" +type
        target_file = target_dir +"/" +type +".bio"
        with open(src_file) as src:
            assert not os.path.isfile(target_file), target_file
            with open(target_file, "w") as trg:

                for line in src:
                    line = line.strip()
                    if line:
                        line = line.split("\t")
                        try:
                            word = line[0].split(":")[1]
                        except:
                            import pdb
                            pdb.set_trace()
                        tag = line[1]
                        trg.write(f"{word}\t{tag}\n")
                    else:
                        trg.write("\n")
        print(f"{target_file} created fron {src_file}")


def copy_tsv_to_new_loc(src_loc, trg_loc):
    for type in ["train", "dev", "test"]:
        src_file = src_loc + "/" + type+".bio"
        #target_file = trg_loc + "/" + type + ".bio"
        copyfile(src_file, trg_loc+"/"+type+".bio")
        print(f"{src_file} copied to {trg_loc+'/'+type+'.bio'}")

if __name__ == "__main__":
    ls_code = []
    #for lang in ["en", "fr", "de", "id", "tr", "ar", "ru"]:
    #for lang in ["am", "ug", "sd", "ceb", "ku", "tl", "mt", "fo"]:
    dic = {"mt": "Maltese", "ckb": "Sorani","fo":"Faroese", "sd" :"Sindhi",
           "ug": "Uyghur",  "xmf": "Mingrelian", "mhr":"Mari"}

    for lang in ["xmf", "fo", "ckb", "sd", "ug", "mhr"]:
    #for lang in ["ckb"]:
        target = f"/Users/bemuller/Documents/Work/INRIA/dev/data/wikiann-rahini/{lang}"
        src = f"/Users/bemuller/Documents/Work/INRIA/dev/data/wikiann-rahini/{lang}"
        if False:


            #target =
            #end_target = "/Users/bemuller/Documents/Work/INRIA/dev/parsing/data/wikiner/{}_wiki-wikiner-{}.conll".format(lang, set_)
            ls_code.append("{}_wiki".format(lang))
            n_line = wiki_ann_to_tsv(src, target)
            #cmd = "head -{} {} > {}".format(n_line-1, target, end_target)
            #os.system(cmd)
            #print("COPIED TO {}".format(end_target))
            #print("LANG DONE", lang)
        copy_tsv_to_new_loc(src, f"/Users/bemuller/Documents/Work/INRIA/dev/stanza-train/data/nerbase/{dic[lang]}-TEST")
    #print(set(ls_code))



# need to put them in the name I want ! (word2vec point to any kind)
