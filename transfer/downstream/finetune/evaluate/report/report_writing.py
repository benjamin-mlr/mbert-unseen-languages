from transfer.downstream.finetune.env.imports import pdb, sys, os
from transfer.downstream.finetune.model.settings import TASKS_PARAMETER

from transfer.downstream.finetune.evaluate.score.confusion_matrix_rates import get_perf_rate

#sys.path.insert(0, os.path.join(os.environ.get("EXPERIENCE", ".."), "experimental_pipe"))
try:
    sys.path.append(os.environ.get("EXPERIENCE"))
    from transfer.write_to_performance_repo import report_template
except:
    from transfer.downstream.finetune.evaluate.score.report_template import report_template
    print("REPORTING modules downloaded from local project ")


def report_score_all(evaluated_task, agg_func_ls, samples, label_heuristic, score_dic, n_tokens_dic, n_sents_dic,
                     model_id, tasks, args_dir, data_label, reports, writer, log_perf,
                     early_stoppin_metric_val, early_stoppin_metric, mode, subsample_early_stoping_metric_val, epoch, verbose=1):

    score = None
    n_tokens = 0
    assert isinstance(samples, dict), "ERROR samples : {}".format(samples)
    for task in list(set(evaluated_task)):
        #pdb.set_trace()
        assert task in samples, "ERROR : task {} was not found in samples dictionary {}".format(task, samples)
        _samples = samples[task]
        for agg_func in agg_func_ls:
            for sample in _samples:
                #print("sample", sample)
                # for binary classification : having 3 samples define [class Positive, class Negative, All]
                #  e.g [NORMED, NEED_NORM , all] for a given agg_func
                # TP : score_dic[agg_func][Positive Class]
                # TN : score_dic[agg_func][Negative Class]
                # P observations = n_tokens_dic[agg_func][Positive Class]
                # N observations  = n_tokens_dic[agg_func][Negative Class]
                # (?to confirm) PP predictions = FP + TP = (N-TN) + TP
                # (?to confirm) NP predictions = FN + TN = (P-TP) + TN
                # recall = TP/P , precision = TP/PP,  tnr = TN/N , npr = TN/NP
                # f1 = hmean(recall, precision) , accuracy = (TN+TP)/(N+P)
                score = score_dic[task][agg_func][sample]
                n_tokens = n_tokens_dic[task][agg_func][sample]
                n_sents = n_sents_dic[task][agg_func][sample]
                # metric_val = "accuracy-exact-{}".format(tasks[1] if len(tasks)>1 else tasks[0])
                metric_val = "accuracy-{}".format(task)

                try:
                    report = report_template(metric_val=metric_val, subsample=sample +label_heuristic,
                                             info_score_val=None,
                                             score_val=score /n_tokens if n_tokens > 0 else None,
                                             n_sents=n_sents,
                                             avg_per_sent=0,
                                             n_tokens_score=n_tokens,
                                             model_full_name_val=model_id, task=tasks,
                                             evaluation_script_val="exact_match",
                                             model_args_dir=args_dir,
                                             token_type="word",
                                             report_path_val=None,
                                             data_val=data_label)
                except Exception as e:
                    print("REPORT : ERROR")
                    raise (e)

                if early_stoppin_metric is not None:
                    if metric_val == early_stoppin_metric \
                            and subsample_early_stoping_metric_val == sample + label_heuristic \
                            and score is not None and n_tokens > 0:
                        early_stoppin_metric_val = -score/n_tokens
                    elif score is None:
                        if verbose>1:
                            print("WARNING : could no apply early stopping metric cause score is None")
                    elif n_tokens == 0:
                        if verbose > 1:
                            print("WARNING : could no apply early sotpping metric cause n_tokens is 0 ")
                if writer is not None and log_perf:
                    writer.add_scalars("perf-{}".format(tasks[0]),
                                       {"{}-{}-{}-{}-{}".format(metric_val, mode, model_id, data_label, sample):
                                            score /n_tokens if n_tokens >0 and score is not None else 0
                                        }, epoch)
                reports.append(report)

        # class negative 0 , class positive 1
        # TOD : make that more consistent with user needs !

        evaluated_tasks_confusion_matrix_based_score = list(set(evaluated_task) & set(["normalize", "normalize_pred"]))

        for task in evaluated_tasks_confusion_matrix_based_score:
            import re
            task_real = re.match("(.*)-(.*)", task).group(1)

            if "all" in _samples and TASKS_PARAMETER[task_real]["predicted_classes"][0] in _samples and TASKS_PARAMETER[task_real]["predicted_classes"][1] in _samples:
                # then we can compute all the confusion matrix rate
                #-TOD : factorize with TASKS_2_METRICS_STR

                for metric_val in ["precision", "f1", "recall", "tnr", "npv"]:
                    metric_val += "-"+task
                    score, n_rate_universe = get_perf_rate(metric=metric_val, n_tokens_dic=n_tokens_dic[task],
                                                           score_dic=score_dic[task], task=task, agg_func=agg_func)
                    try:
                        report = report_template(metric_val=metric_val, subsample="rates" +label_heuristic,
                                                 info_score_val=None, score_val=score, n_sents=n_sents_dic[task][agg_func]["all"],
                                                 avg_per_sent=0, n_tokens_score=n_rate_universe, model_full_name_val=model_id,
                                                 task=tasks, evaluation_script_val="exact_match", model_args_dir=args_dir,
                                                 token_type="word", report_path_val=None, data_val=data_label)
                    except Exception as e:
                        print("REPORT ERROR {} ".format(e))

                    if early_stoppin_metric is not None:
                        if metric_val == early_stoppin_metric and subsample_early_stoping_metric_val == "rates" + label_heuristic and score is not None:
                            early_stoppin_metric_val = -score
                    reports.append(report)

                    if writer is not None and log_perf:
                        writer.add_scalars("perf-{}".format(tasks[0]), {"{}-{}-{}-{}-bpe".format(metric_val, data_label, mode, model_id): score if score is not None else 0}, epoch)

    return reports, early_stoppin_metric_val, score, n_tokens

