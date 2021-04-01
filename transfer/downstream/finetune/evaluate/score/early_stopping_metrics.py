from transfer.downstream.finetune.model.settings import TASKS_PARAMETER
from transfer.downstream.finetune.io_.logger import printing


#from test_._test_early_stoping import sanity_check_early_stop_metric


def get_early_stopping_metric(tasks, verbose, main_task=None, early_stoppin_metric=None, subsample_early_stoping_metric_val=None):
    """
    getting early stopping metric and evaluation subsample
    if early_stoppin_metric is None : uses first eval_metrics stated in TASKS_PARAMETER of the first task of the list passed in args.tasks
    :return:
    """
    if main_task is None:
        printing("INFO : default main task provided is the first of the first list {} ", var=[tasks], verbose=verbose, verbose_level=1)
        if isinstance(tasks[0], list):
            main_task = tasks[0][0]
        else:
            main_task = tasks[0]

    if early_stoppin_metric is None:
        early_stoppin_metric = TASKS_PARAMETER[main_task]["eval_metrics"][0][0]

        printing("INFO : default early_stoppin_metric is early_stoppin_metric  {} first one of "
                 "the first possible in TASK_PARAMETER", var=[early_stoppin_metric],
                 verbose=verbose, verbose_level=1)

    if subsample_early_stoping_metric_val is None:
        get_subsample = TASKS_PARAMETER[main_task].get("default-subsample")
        if get_subsample is None:
            get_subsample = "all"
            printing("INFO : early stopping subsample is set to default {} all as not found in {}", var=["all", TASKS_PARAMETER[main_task]], verbose=verbose, verbose_level=1)
        subsample_early_stoping_metric_val = get_subsample
        assert subsample_early_stoping_metric_val in TASKS_PARAMETER[main_task]["subsample-allowed"], "ERROR task {} subsample not in {} ".format(main_task, subsample_early_stoping_metric_val)
    #sanity_check_early_stop_metric(early_stoppin_metric, TASKS_PARAMETER, tasks)

    return early_stoppin_metric, subsample_early_stoping_metric_val
