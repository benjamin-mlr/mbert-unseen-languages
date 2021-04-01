import os
import random
from uuid import uuid4
src_dir=os.environ.get("OSCAR")
target_lang = "mt"
LANG_FAM = {"mt":["en"]}


def get_oscar_file(lang, src_dir, post_fix="oscar", sanity_check=True):
	file = os.path.join(src_dir,f"{lang}_{post_fix}-train.txt")
	assert os.path.isfile(file)
	if sanity_check:
		num_lines = sum(1 for line in open(file))
		assert num_lines>0, "ERROR : file is empty "
	return file

target_dir  = get_oscar_file(target_lang,src_dir)
num_lines_target = sum(1 for line in open(target_dir))
concat_file_ls_dirs  = [get_oscar_file(lang, src_dir) for lang in LANG_FAM[target_lang]]+[target_dir]
lang_ls = LANG_FAM[target_lang]+[target_lang]

id_file = str(uuid4())[:4]

lang_fam_concat_pref = target_lang+"_"+"_".join(LANG_FAM[target_lang])+f"_{id_file}"

concat_file_dir = f"{src_dir}/{lang_fam_concat_pref}-train.txt"
print(f"Creating {concat_file_dir}")

def concat(concat_file_ls_dirs, num_lines_target, dir_to_write, lang_ls, min_line_len=10, shuffle=True):
	
	lines = []
	num_per_lang = num_lines_target//(len(concat_file_ls_dirs)-1)
	
	info_text = ""
	assert len(lang_ls)==len(concat_file_ls_dirs)
	
	for i, (file,lang) in enumerate(zip(concat_file_ls_dirs,lang_ls)):
		counter_lang = 0
		skipping = 0
		if i == (len(lang_ls)-1):
			num_per_lang=num_lines_target
		with open(file, "r") as read:
			for line in read:
				if len(line.strip())>min_line_len: 	
					counter_lang+=1
					lines.append(line)
				else:
					skipping+=1
				if counter_lang>num_per_lang:
					break
			assert counter_lang>=num_per_lang, f"Error not enough sentences from {file} to reach {num_per_lang} lines "
			info_text+=f"{counter_lang} lines of lang {lang} (cond: {min_line_len} min_line_len); \n"
			print(f"From {file}  {counter_lang} copied {skipping} row skipped")
	
	if shuffle:
		print("Shuffling list")
		random.shuffle(lines)
	
	with open(dir_to_write,"w") as f:
		cp = 0
		for line in lines:
			cp+=1
			f.write(line)
	with open(dir_to_write+"-info.txt","w") as g:
		g.write(info_text)
	print(f"{dir_to_write} created {cp} lines")


concat(concat_file_ls_dirs, num_lines_target,concat_file_dir, lang_ls)
