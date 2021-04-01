import sys
sys.path.append("..")
sys.path.append( ".")
from transfer.downstream.finetune.env.imports import np, os, random, pdb, time, listdir, join, isfile
from transfer.downstream.finetune.io_.logger import printing
from transfer.downstream.finetune.env.dir.project_directories import PROJECT_PATH
from transfer.downstream.finetune.env.vars import N_SENT_MAX_CONLL_PER_SHARD



def count_conll_n_sent(dir_file):

    with open(dir_file, 'r') as f:
        n_sent = 0
        for line in f:
            if len(line.strip()) > 0:
                line = line.strip()
                line = line.split('\t')
            if line[0] == "1":
                n_sent += 1
    return n_sent


def create_files(dir_shard, n_shards, label_file):
    ls_files_path = []
    for i in range(n_shards):
        path = os.path.join(dir_shard, "{}_{}.conll".format(label_file, i))
        assert not os.path.isfile(path), "ERROR trying to create existing file {}".format(path)
        open(path, 'a').close()
        ls_files_path.append(path)
    assert len(ls_files_path) == n_shards
    return ls_files_path


def split_randomly(n_shards, dir_shard, dir_file, n_sents, label_file="train", dry_run=False):

    ls_files_path = create_files(dir_shard, n_shards, label_file)
    print("INFO shards files created in {}".format(dir_shard))
    n_sent_written = 0
    start = time.time()
    with open(dir_file, 'r') as f:
        line_former = ""
        for line in f:
            #pdb.set_trace()
            new_sent = line.startswith("#") and len(line_former.strip()) == 0
            if new_sent:
                file = random.choice(ls_files_path)
                n_sent_written += 1
                if dry_run:
                    break
            #print("NEW LINE {} line {}".format(new_sent, line))

            open(file, "a").write(line)
            line_former = line

            if n_sent_written % 10000 == 0:
                print("{} n_sents processed in {} min {} min/sent".format(n_sent_written, (time.time()-start)/60, (time.time()-start)/(60*n_sent_written)))
                sys.stdout.flush()


    if not dry_run:
        assert n_sent_written == n_sents, "ERROR not all counted sentence were writted  counted {}  written {} (shard {})".format(n_sents, n_sent_written, dir_shard)


def build_shard(dir_shard, dir_file, n_sent_max_per_file, format="conll",dry_run=False, verbose=1):

    onlyfiles = [f for f in listdir(dir_shard) if isfile(join(dir_shard, f))]
    if len(onlyfiles) > 0:
        n_shards = len(onlyfiles)
        n_sents = 0
        for file in onlyfiles:
            n_sents += count_conll_n_sent(os.path.join(dir_shard, file))

        printing("INFO : shards already filled in {} files {} sentences total", var=[n_shards, n_sents],
                 verbose=1, verbose_level=1)
        return dir_shard, n_shards, n_sents

    assert format in "conll"
    assert len(dir_file) == 1, "ONLY 1 set of simultaneous task supported for sharding"
    printing("STARTING SHARDING {} of {} ".format(dir_shard, dir_file), verbose=verbose, verbose_level=1)
    dir_file = dir_file[0]
    n_sents = count_conll_n_sent(dir_file)
    n_shards = n_sents//n_sent_max_per_file

    if n_shards == 0:
        printing("INFO SHARDING : n_sent_max_per_file is lower that number of files in {} so only building 1 shard", var=[dir_file], verbose=verbose, verbose_level=1)
        n_shards += 1
    split_randomly(n_shards, dir_shard, dir_file, n_sents, dry_run=dry_run)
    sys.stdout.flush()

    printing("INFO SHARD n_sent written {} splitted in {} files with "
             "in average {} sent per file written to {}",
             var=[n_sents, n_shards,n_sent_max_per_file, dir_shard], verbose=verbose, verbose_level=1)

    return dir_shard, n_shards, n_sents
    

if __name__ == "__main__":
    # "clean_data", "code-mixed_sep_13-train"
    from env.project_variables import MLM_DATA
    #for domain in ["tweets_fr", "wiki_fr", "wiki_en", "reddit_en","reddit_fr", "tweets_en"]:
    #for domain in ["tweets_fr", "wiki_fr"]:
    for domain in ["tweets_fr"]:
    #for domain in [ "reddit_en","reddit_fr"]:

        origin = MLM_DATA[domain]["train"]["large"]
        shard = MLM_DATA[domain]["train"]["shard"]
        try:
        
            print("SHARD starting {} {} ".format(origin, shard))
            build_shard(shard, [origin], N_SENT_MAX_CONLL_PER_SHARD, dry_run=False)
            print("SHARD built for {} {} ".format(origin, shard))
        except Exception as e:
            print(e)
            print("FAILLING  {} {} ".format(origin, shard))
    #shard = os.path.join(os.environ.get("MT_NORM_PARSE_DATA", ".."), "data", "wiki", "fr", "train")
    #data_dir = os.path.join(PROJECT_PATH, "data", "wiki", "fr", "fr.train.conll") #
    #shard = os.path.join(os.environ.get("MT_NORM_PARSE_DATA", ".."), "data", "tweets_en_pan_ganesh", "train")
    #data_dir = os.path.join(PROJECT_PATH, "data", "tweets_en_pan_ganesh", "pan_tweets_en-train.conll") #"/Users/bemuller/Documents/Work/INRIA/dev/parsing/data/Universal-Dependencies-2.4/fr_spoken-ud-train.conllu"
    #data_dir = os.path.join(PROJECT_PATH, "data", "code_mixed", "code-dev-10k.conll.conll")
