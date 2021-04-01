#!/usr/bin/env python
# coding=utf-8
#
# Copyright 2018 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

# original first line was :  #!/usr/bin/env python3

"""Converts CoNLL file from BILOU to BIO encoding."""

import sys


if __name__ == "__main__":
    import argparse

    lines = []
    #print("STATING bilou2bio")
    for line in sys.stdin:
        #print("line-", line)
        line = line.rstrip("\r\n")
        if line.startswith("#"):
            continue
        if not line:
            pass
            #print("not line")
        else:
            #print("line-- ", line)
            #form, lemma, tag, label = line.split("\t")
            _, form, lemma, label, _, _, _, _, _, _ = line.split("\t")
            tag = "_"
            if label.startswith("U-"):
                label = label.replace("U-", "B-")
            if label.startswith("L-"):
                label = label.replace("L-", "I-")
            print("\t".join([form, lemma, tag, label]))
