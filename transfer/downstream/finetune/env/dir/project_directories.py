
from transfer.downstream.finetune.env.imports import os, pdb

PROJECT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..")

#  BERT_MODELS_DIRECTORY = os.path.join(PROJECT_PATH,"..","..","..","..", "representation", "lm", "bert_models")
BERT_MODELS_DIRECTORY = os.path.join(PROJECT_PATH, "..", "pretrained")

# saving checkpoints
if os.environ.get("ENV") == "neff":

    CHECKPOINT_DIR = os.path.join(os.environ["MT_NORM_PARSE_DATA"], "checkpoints")
    print("INFO : Project_variables : CHECKPOINT_DIR set to {}".format(CHECKPOINT_DIR))
else:
    CHECKPOINT_DIR = os.path.join(PROJECT_PATH, "./checkpoints")
CHECKPOINT_BERT_DIR = os.path.join(CHECKPOINT_DIR, "bert")


# google sheet for tracking
CLIENT_GOOGLE_CLOUD = os.path.join(PROJECT_PATH,  "io_", "runs_tracker", "google_api")

SHEET_NAME_DEFAULT, TAB_NAME_DEFAULT = "model_evaluation", "experiments_tracking"

assert os.path.isdir(CHECKPOINT_BERT_DIR), "ERROR {} ".format(CHECKPOINT_BERT_DIR)
try:
    assert os.path.isdir(CLIENT_GOOGLE_CLOUD), "ERROR {} ".format(CLIENT_GOOGLE_CLOUD)
except:
    pass
assert os.path.isdir(BERT_MODELS_DIRECTORY), "ERROR {} ".format(BERT_MODELS_DIRECTORY)