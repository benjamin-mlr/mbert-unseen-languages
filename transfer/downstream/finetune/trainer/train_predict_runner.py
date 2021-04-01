from transfer.downstream.finetune.env.imports import os, pdb, json

from transfer.downstream.finetune.io_.logger import printing
from transfer.downstream.finetune.evaluate.score.early_stopping_metrics import get_early_stopping_metric
from transfer.downstream.finetune.env.seeds import init_seed
from transfer.downstream.finetune.env.dir.pretrained_model_dir import BERT_MODEL_DIC
from transfer.downstream.finetune.model.constants import NULL_STR
from transfer.downstream.finetune.trainer.run import run

from transformers import BertTokenizer, AutoTokenizer, AutoModel, RobertaTokenizerFast, CamembertTokenizer


def train_predict_eval(args, verbose=0):

    init_seed(args)
    if args.bert_model in BERT_MODEL_DIC:
        model_dir = BERT_MODEL_DIC[args.bert_model]["model"] if args.bert_model else None
        encoder = BERT_MODEL_DIC[args.bert_model]["encoder"] if args.bert_model else None
    else:
        model_dir = None
        encoder = "AutoModel"

    if args.init_args_dir is not None:
        args_checkpoint = json.load(open(args.init_args_dir, "r"))
        args.bert_model = args_checkpoint["hyperparameters"]["bert_model"]

    # if model referenced BERT_MODEL_DIC : using tokenizer directory otherwise loading from hugging face
    if args.bert_model in BERT_MODEL_DIC:
        tokenizer = eval(BERT_MODEL_DIC[args.bert_model]["tokenizer"]) if args.bert_model else None  # , "BertTokenizer"))
        voc_tokenizer = BERT_MODEL_DIC[args.bert_model]["vocab"] if args.bert_model else None
        vocab_size = BERT_MODEL_DIC[args.bert_model].get("vocab_size") if args.bert_model else None
    else:
        print("TOKENIZER Model not in BERT_MODEL_DIC so loading tokenizer from hugging face")
        tokenizer = AutoTokenizer
        voc_tokenizer = args.bert_model
        vocab_size = None

    null_token_index = vocab_size
    description = "grid"

    # We checkpoint the model only if early_stoppin_metric gets better ,
    # early_stoppin_metric choosen in relation to the first task defined in the list
    early_stoppin_metric, subsample_early_stoping_metric_val = get_early_stopping_metric(tasks=args.tasks,early_stoppin_metric=None, verbose=verbose)

    printing("INFO : tasks is {} so setting early_stoppin_metric to {} ", var=[args.tasks, early_stoppin_metric],
             verbose=verbose, verbose_level=1)

    printing("INFO : model {} batch_update_train {} batch_size {} ",
             var=[args.model_id_pref, args.batch_update_train, args.batch_size],
             verbose=verbose, verbose_level=1)

    run(args=args, voc_tokenizer=voc_tokenizer, vocab_size=vocab_size, model_dir=model_dir,
        report_full_path_shared=args.overall_report_dir,
        description=description, null_token_index=null_token_index, null_str=NULL_STR,
        model_suffix="{}".format(args.model_id_pref), debug=False,
        random_iterator_train=True,  bucket_test=False, compute_intersection_score_test=True,
        n_observation_max_per_epoch_train=args.n_iter_max_train if not args.demo_run else 2,
        n_observation_max_per_epoch_dev_test=50000 if not args.demo_run else 2,
        early_stoppin_metric=early_stoppin_metric,
        subsample_early_stoping_metric_val=subsample_early_stoping_metric_val,
        saving_every_epoch=args.saving_every_n_epoch, run_mode="train" if args.train else "test",
        auxilliary_task_norm_not_norm=True, tokenizer=tokenizer, max_token_per_batch=300,
        name_with_epoch=args.name_inflation, encoder=encoder, report=True, verbose=verbose)

    printing("MODEL {} trained and evaluated", var=[args.model_id_pref], verbose_level=1, verbose=verbose)
