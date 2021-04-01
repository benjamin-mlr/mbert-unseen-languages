
from transfer.downstream.finetune.io_.logger import printing
from transfer.downstream.finetune.env.imports import re


def get_normalized_token(norm_field, n_exception, verbose, predict_mode_only=False):

  match = re.match("^Norm=([^|]+)|.+", norm_field)

  try:
    assert match.group(1) is not None, " ERROR : not normalization found for norm_field {} ".format(norm_field)
    normalized_token = match.group(1)

  except:
    match_double_bar = re.match("^Norm=([|]+)|.+", norm_field)

    if match_double_bar.group(1) is not None:
      match = match_double_bar
      n_exception += 1
      printing("Exception handled we match with {}".format(match_double_bar.group(1)), verbose=verbose, verbose_level=2)
      normalized_token = match.group(1)

    else:
      exc = Exception("Failed to handle exception with | on field {} ".format(norm_field))
      if not predict_mode_only:
        raise(exc)
      else:
        print("REPLACING with UNK",exc)
        normalized_token = "UNK"


  return normalized_token, n_exception
