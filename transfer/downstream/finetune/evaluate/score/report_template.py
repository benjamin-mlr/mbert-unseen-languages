from transfer.downstream.finetune.env.imports import OrderedDict


def report_template(metric_val, info_score_val, score_val, model_full_name_val, report_path_val,
                    evaluation_script_val, data_val, model_args_dir,
                    n_tokens_score=None, task=None, n_sents=None, token_type=None, subsample=None,
                    avg_per_sent=None, min_per_epoch=None, layer_i=None):

    return OrderedDict([("metric", metric_val), ("info_score", info_score_val), ("score", score_val),
                        ("model_full_name", model_full_name_val),
                        ("n_tokens_score", n_tokens_score), ("n_sents", n_sents), ("token_type", token_type),
                        ("subsample", subsample),
                        ("avg_per_sent", avg_per_sent),
                        ("min/epoch", min_per_epoch),
                        ("model_args_dir", model_args_dir),
                        ("report_path", report_path_val),
                        ("evaluation_script", evaluation_script_val),
                        ("task", task),
                        ("layer", layer_i),
                        ("data", data_val)])


