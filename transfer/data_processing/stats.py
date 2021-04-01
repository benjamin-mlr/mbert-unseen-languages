
import os
import numpy as np


def file_check(dir):
    assert os.path.isfile(dir), f"ERROR {dir} not a file"
    num_lines = sum(1 for line in open(dir))
    assert num_lines>0, f"ERROR : {dir} empty"


def n_lines_check(dir, test_n_lines=None, set="train"):
    if test_n_lines is None:
        assert set is not None
        test_n_lines = 5000 if set == "test" else 1000000
    #num_lines = sum(1 for _ in open(dir))
    num_lines = sum([1 for i in open(dir,"r").readlines() if i.strip()])
    try:
        assert num_lines == test_n_lines , f"ERROR {dir} should have {test_n_lines} but has {num_lines}"
    except Exception as e:
        if num_lines>test_n_lines:
            print(f"WARNING: {dir} file has {num_lines}")
        else:
            raise(Exception(e))

def get_n_char_stat(dir):
    with open(dir) as f:
        len_line = []
        count_short_sent = 0
        for line in f:
            line = line.strip()
            len_line.append(len(line))
            if len(line)<20:
                count_short_sent=+1
    len_line = np.array(len_line)
    print(f"{dir} mean:{np.mean(len_line)}, median:{np.median(len_line)} max:{np.max(len_line)} min:{np.min(len_line)} n_sent:{count_short_sent}(shorter than 20 char)")


if __name__ == "__main__":

    lang_ls = ["ar", "de", "en", "fa", "fi", "fr", "he", "hi", "hu", "it", "ja", "ko", "ru", "tr",
                #"ug",# "mt","ckb"
                 ]
    for lang in lang_ls:
        for set in ["train", "test"]:
            dir = f"{os.environ.get('OSCAR')}/{lang}_oscar-{set}.txt"
            file_check(dir)
            n_lines_check(dir, set=set)
            # same with demo
            print(f"{dir} tested")
            get_n_char_stat(dir)

        demo_file=True
        if demo_file:
            dir = f"{os.environ.get('OSCAR')}/{lang}_oscar_demo-train.txt"
            file_check(dir)
            n_lines_check(dir, test_n_lines=5000)
            print(f"{dir} tested")

