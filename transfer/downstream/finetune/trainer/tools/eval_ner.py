from transfer.downstream.finetune.env.imports import os
from transfer.downstream.finetune.env.dir.project_directories import PROJECT_PATH
import pdb

def evaluate(dataset_name, dir_end_pred, dataset,
             prediction_file, gold_file_name, #args
             ):
    "cp from line 382 https://github.com/ufal/acl2019_nested_ner/blob/master/tagger.py"
    # with open("{}/{}_system_predictions.conll".format(args.logdir, dataset_name), "w", encoding="utf-8") as prediction_file:
    # self.predict(dataset_name, dataset, args, prediction_file, evaluating=True)
    # prediction_file = #get_prediction_file
    # prediction_file =
    f1 = 0.0
    dataset_name = "dev"
    #args.corpus = "CoNLL_es"
    corpus = "CoNLL_es"
    logdir = dir_end_pred
    # !! gold_file_name NB : should be in dir_end_pred
    if corpus in ["CoNLL_en", "CoNLL_de", "CoNLL_nl", "CoNLL_es"]:


        # os.system("cd {} && ../../run_conlleval.sh {} {} {}_system_predictions.conll".format(args.logdir, dataset_name, args.__dict__[dataset_name + "_data"], dataset_name))
        dir_run_conll_eval = "/home/bemuller/projects/experimental_pipe/../transfer/transfer/downstream/finetune/trainer/tools"#os.path.join(PROJECT_PATH, "/camembert/camembert/downstream/finetune/trainer/tools")
        if not os.path.isdir(dir_run_conll_eval):
            dir_run_conll_eval = "/Users/bemuller/Documents/Work/INRIA/dev/camembert/camembert/downstream/finetune/trainer/tools"  # os.path.join(PROJECT_PATH, "/camembert/camembert/downstream/finetune/trainer/tools")
            assert os.path.isdir(dir_run_conll_eval), "ERROR {} do not exit".format(dir_run_conll_eval)
        if not os.path.isfile(os.path.join(logdir, "dev.eval")):
            open(os.path.join(logdir, "dev.eval"), "w").write("")
        print("PROJECT_PATH", PROJECT_PATH)
        print("cd {} && {}/run_conlleval.sh {} {} {}".format(logdir, dir_run_conll_eval, dataset_name, gold_file_name, prediction_file))
        os.system("cd {} && {}/run_conlleval.sh {} {} {}".format(logdir, dir_run_conll_eval, dataset_name, gold_file_name, prediction_file))

        with open("{}/{}.eval".format(logdir, dataset_name), "r", encoding="utf-8") as result_file:
            for line in result_file:
                line = line.strip("\n")
                if line.startswith("accuracy:"):
                    f1 = float(line.split()[-1])
                    # self.session.run(self.metrics_summarize["F1"][dataset_name], {self.metrics["F1"]: f1})

        return f1


if __name__ == "__main__":

    dir = "/Users/bemuller/Documents/Work/INRIA/dev/experimental_pipe/reporting/ner/9987544-ca445-9987544_job-10927_model/"

    f1 = evaluate(dataset_name=None,
                  dataset=None, dir_end_pred=dir,
                  prediction_file="/Users/bemuller/Documents/Work/INRIA/dev/camembert/camembert/downstream/finetune/env/dir/../.././checkpoints/bert/414ee-dc8ee-414ee_job-0644e_model/predictions/0_ep-prediction-.conll",#,os.path.join(dir, "LAST_ep-prediction-fr_ftb_pos_ner-ud-test-.conll"),
                  gold_file_name="/Users/bemuller/Documents/Work/INRIA/dev/camembert/camembert/downstream/finetune/env/dir/../.././checkpoints/bert/414ee-dc8ee-414ee_job-0644e_model/predictions/0_ep-gold--.conll")#os.path.join(dir, "LAST_ep-gold--fr_ftb_pos_ner-ud-test-.conll"))

    print(f1)

