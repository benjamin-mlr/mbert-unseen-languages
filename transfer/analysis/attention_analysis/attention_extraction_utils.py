
from transfer.downstream.finetune.env.imports import pdb, torch, OrderedDict, np, linear_model, json, uuid4
from transfer.downstream.finetune.env.dir.data_dir import DATA_UD_RAW, DATA_UD
from transfer.downstream.finetune.env.dir.pretrained_model_dir import BERT_MODEL_DIC
from transfer.downstream.finetune.env.dir.project_directories import PROJECT_PATH


def get_dirs(args):

    if args.init_args_dir is not None:
        with open(args.init_args_dir,"r") as f:
            args_dic = json.load(f)
            args_dic = args_dic["hyperparameters"]
            dict_path = args_dic["dict_path"]
            model_dir = args_dic["model_location"]
            args.tasks = args_dic["tasks"]
    else:
        model_dir = BERT_MODEL_DIC[args.bert_model]["model"]
        dict_path = PROJECT_PATH+"/../../analysis/attention_analysis/dic/"
        args.tasks = [["pos"]]
        print(f"Defining model_dir with pretrained model {args.bert_model}")

    return args, dict_path, model_dir


def load_data(dir, data=None, line_filter=None, id_start=0, id_end=100,  verbose=1):
    if data is None:
        data = []
    if verbose:
        print(f"LOADING {dir}")
    n_sent = 0
    n_appending = 0
    with open(dir, "r") as f:
        for line in f:
            if line_filter is not None:
                if not line.startswith(line_filter):
                    continue
                line = line[len(line_filter):]
            line = line.strip()
            if n_sent >= id_start and n_sent < id_end:
                n_appending += 1
                data.append(line)
            n_sent += 1
    try:
        assert n_appending == (id_end-id_start)
    except Exception as e:
        print(e)
        return None
    if verbose:
        print(f"LOADED {n_appending} form {dir}  ")

    return data


def load_lang_ls(root, lang_ls, postfix="-ud-test-sent_segmented.txt"):
    data_lang = []
    y = []
    for i_lang, lang in enumerate(lang_ls):
        dir = root+f"/{lang}{postfix}"
        data_lang = load_data(dir, data_lang)
        y.extend([i_lang for _ in range(len(data_lang))])
    return np.array(data_lang), np.array(y)
