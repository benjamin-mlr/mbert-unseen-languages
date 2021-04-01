

def get_key_name_num_label(task, task_parameters):
    """
    for each task get the name of the prediction/loss tensor
        tasks that have are single tensor prediction (all labelling tasks except parsing basically) have a  name after task+"-"+label_name
        tasks that have are double tensor got  task + "-" + task_parameters[task]["num_labels_mandatory_to_check"][0] (parsing-types)


    :param task:
    :param task_parameters:
    :return:
    """
    if len(task_parameters[task]["label"]) == 1:
        return task + "-" + task_parameters[task]["label"][0]
    else:
        return task + "-" + task_parameters[task]["num_labels_mandatory_to_check"][0]  # assuming 1 in num_labels_mandatory_to_check
