

# argument that are passed as list into ArgParse and transformed as dictionaries
DIC_ARGS = ["multi_task_loss_ponderation", "lr", "ponderation_per_layer", "norm_order_per_layer"]
# loss ponderation can be passes as dictionary or predifined mode : here is the list of available predifinitions
MULTI_TASK_LOSS_PONDERATION_PREDEFINED_MODE = ["uniform", "normalization_100", "pos_100","all", "pos", "normalize", "norm_not_norm"]
# 2 penalization mode available
AVAILALE_PENALIZATION_MODE = ["layer_wise", "pruning"]
