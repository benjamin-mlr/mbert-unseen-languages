from transfer.downstream.finetune.env.imports import Iterable, sys

DEBUG = False
LOGGING_SPECIFIC_INFO_AVAILABLE = ["cuda", "raw_data", "alignement", "mask", "pred", "reader", "gpu"]


def printing(message, verbose, verbose_level, var=None):
    """
    # if verbose is string then has to be in LOGGING_SPECIFIC_INFO_AVAILABLE
    :param message:
    :param verbose:
    :param verbose_level:
    :param var:
    :param carasteristic:
    :return:
    """
    if isinstance(verbose_level, str):
        assert verbose_level in LOGGING_SPECIFIC_INFO_AVAILABLE, "ERROR unavailble verbosity {} not in {}".format(verbose_level, LOGGING_SPECIFIC_INFO_AVAILABLE)

    verbose_level = 0 if DEBUG else verbose_level
    if isinstance(verbose, int) and isinstance(verbose_level, int):
        if verbose >= verbose_level:
            if var is not None:
                if isinstance(var, Iterable):
                    print(message.format(*var))
                else:
                    print(message.format(var))
            else:
                print(message)
    elif isinstance(verbose, str) and isinstance(verbose_level, str):
        if verbose == verbose_level:
            if isinstance(var, Iterable):
                print(message.format(*var))
            else:
                print(message.format(var))
    sys.stdout.flush()


def disable_tqdm_level(verbose, verbose_level):
    return False if isinstance(verbose,int) and verbose >= verbose_level else True
