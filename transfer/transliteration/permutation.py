import numpy as np
import string
import pdb
from tqdm import tqdm
from collections import OrderedDict
from lang_trans.arabic import buckwalter
from transliterate import translit
import pykakasi
import random
#from transformers
import os


def get_permute(alphanumerical_char=None):
    if alphanumerical_char is None:
        alphanumerical_char = string.printable[:62]
    #pdb.set_trace()
    
    if "\\"  in alphanumerical_char:
        rm_i = None
        for i, a in enumerate(alphanumerical_char):
            if a=="\\":
                rm_i = i
                break
        if rm_i is not None:
            del alphanumerical_char[rm_i]
    assert "\\" not in alphanumerical_char
    
    # removing break line symbol
    #alphanumerical_char = alphanumerical_char.replace("\n","")
    #alphanumerical_char = alphanumerical_char.replace("\n", "")
    alphanumerical_char_ls = list(alphanumerical_char)

    alphanumerical_char_ls_or = alphanumerical_char_ls.copy()
    random.shuffle(alphanumerical_char_ls)

    perm = OrderedDict()

    for s, t in zip(alphanumerical_char_ls_or, alphanumerical_char_ls):
        if s == "\n" or t == "\n":
            pdb.set_trace()
        perm[s] = t

    return perm



