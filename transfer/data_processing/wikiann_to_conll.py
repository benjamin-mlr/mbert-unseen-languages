import os


def process(dir_src, dir_target):

    with open(dir_src,"r") as src:

        sent_id = 0
        id = 0
        n_line = 0
        try:
            assert not os.path.isfile(dir_target)
        except:
            os.remove(dir_target)
            assert not os.path.isfile(dir_target)
        with open(dir_target, 'w') as trg:
            for line in src:
                n_line += 1
                line = line.strip()
                if len(line) == 0 or sent_id == 0:
                    id = 1
                    sent_id += 1
                    if sent_id > 1:
                        trg.write("\n".format(sent_id))
                    trg.write("# new sent id : {} \n".format(sent_id))
                    if sent_id > 1:
                        continue
                line = line.split("\t")
                word = line[0].split(":")[1]
                ner = line[1]
                trg.write("{id}\t{word}\t{lemma}\t{ner}\t_\t_\t_\t_\t_\t_\n".format(id=id, word=word,lemma="_", ner=ner))
                id += 1
    print("File {} processed to {}".format(dir_src, dir_target))

    return n_line


if __name__ == "__main__":
    ls_code = []
    #for lang in ["en", "fr", "de", "id", "tr", "ar", "ru"]:
    #for lang in ["am", "ug", "sd", "ceb", "ku", "tl", "mt", "fo"]:
    for lang in ["cdo", "mhr", "xmf"]:
    # +  (to copy and rename )
    #for lang in [ "cs_pud",   "es_pud", "fi_pud", "hi_pud", "it_pud","ja_pud", "ko_pud",
    #             "pl_pud", "pt_pud",  "sv_pud","th_pud", "zh_pud"]:
        #lang = lang[:2]
        for set_ in ["train", "dev", "test"]:

            dir = "/Users/bemuller/Documents/Work/INRIA/dev/data/wikiann-rahini/{}/{}".format(lang, set_)
            target = dir+".conll"
            end_target = "/Users/bemuller/Documents/Work/INRIA/dev/parsing/data/wikiner/{}_wiki-wikiner-{}.conll".format(lang, set_)
            ls_code.append("{}_wiki".format(lang))
            n_line = process(dir, target)
            cmd = "head -{} {} > {}".format(n_line-1, target, end_target)
            os.system(cmd)
            print("COPIED TO {}".format(end_target))
    print(set(ls_code))

