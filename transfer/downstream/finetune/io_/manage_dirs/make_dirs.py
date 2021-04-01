from transfer.downstream.finetune.env.imports import uuid4, os
from transfer.downstream.finetune.io_.logger import printing


def setup_repoting_location(root_dir_checkpoints, model_suffix="", shared_id=None, data_sharded=None,
                            verbose=1):
    """
    create an id for a model and locations for checkpoints, dictionaries, tensorboard logs, data
    :param model_suffix:
    :param verbose:
    :return:
    """
    model_local_id = str(uuid4())[:5]
    if shared_id is not None:
        if len(shared_id) > 0:
            model_local_id = shared_id+"-"+model_local_id
    if model_suffix != "":
        model_local_id += "-"+model_suffix
    model_location = os.path.join(root_dir_checkpoints, model_local_id)
    dictionaries = os.path.join(root_dir_checkpoints, model_local_id, "dictionaries")
    tensorboard_log = os.path.join(root_dir_checkpoints, model_local_id, "tensorboard")
    end_predictions = os.path.join(root_dir_checkpoints, model_local_id, "predictions")

    os.mkdir(model_location)

    if data_sharded is None:
        data_sharded = os.path.join(root_dir_checkpoints, model_local_id, "shards")
        os.mkdir(data_sharded)
    else:
        assert os.path.isdir(data_sharded), "ERROR data_sharded not dir {} ".format(data_sharded)
        printing("INFO DATA already sharded in {}",var=[data_sharded], verbose=verbose, verbose_level=1)
    printing("CHECKPOINTING model location:{}", var=[model_location], verbose=verbose, verbose_level=1)
    printing("CHECKPOINTING model ID:{}", var=[model_local_id], verbose=verbose, verbose_level=1)
    os.mkdir(dictionaries)
    os.mkdir(tensorboard_log)
    os.mkdir(end_predictions)
    printing("CHECKPOINTING \n- {} for checkpoints \n- {} for dictionaries created \n- {} predictions {} ",
             var=[model_location, dictionaries, end_predictions, data_sharded], verbose_level=1, verbose=verbose)
    return model_local_id, model_location, dictionaries, tensorboard_log, end_predictions, data_sharded


