
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.join(os.environ["CAMEMBERT_FINE_TUNE"], "..", "..", ".."))
sys.path.insert(0, os.path.join(os.environ["CAMEMBERT_FINE_TUNE"], "..", ".."))
sys.path.insert(0, os.path.join(os.environ["CAMEMBERT_FINE_TUNE"], ".."))
sys.path.insert(0, os.path.join(os.environ["CAMEMBERT_FINE_TUNE"]))

print(sys.path)
from camembert.downstream.finetune.args.args_parse import args_train, args_preprocessing
from camembert.downstream.finetune.trainer.train_predict_runner import train_predict_eval

if __name__ == "__main__":

    args = args_train(training=False)

    args.epochs = 0
    args.batch_size = 1
    args.lr = "0.1"
    args.train = 0
    args.multi_task_loss_ponderation = "pos-pos=1.0,parsing-types=1,parsing-heads=1,mlm-wordpieces_inputs_words=1.0,"
    args.train_path = args.test_paths
    args.dev_path = args.test_paths
    args.seeds = 110
    assert len(args.test_paths) == 1, "ERROR only one test set when prediction"
    args = args_preprocessing(args)
    assert args.init_args_dir is not None, "ERROR {} required".format(args.init_args_dir)

    train_predict_eval(args)
