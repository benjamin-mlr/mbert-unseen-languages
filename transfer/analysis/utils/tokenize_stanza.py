
import stanza 
import time
import os
dir_src=""
lang="en"
dir_target=""


def tokenize(dir_src, dir_target, lang_1, dir_src_2=None, dir_target_2=None, lang_2=None):
	tok = stanza.Pipeline(lang=lang_1, processors='tokenize')
	line_ind=0
	print(f"Starting reading {dir_src} with tokenizer {lang_1} and writing {dir_target} ")
	print(f"in parrallel reading {dir_src_2} with tokenizer {lang_2} and writing {dir_target_2} ")
	start = time.time()
	
	if dir_src_2 is not None:
		src_2 = open(dir_src_2, "r")
		
		if os.path.isfile(dir_target_2):
			target_2 = open(dir_target_2, "w")
			target_2.close()

		target_2 = open(dir_target_2, "a")
		tok_2 = stanza.Pipeline(lang=lang_2, processors='tokenize')
	with open(dir_src, "r") as src:
		with open(dir_target, "w") as target:
			for line in src:
				line_ind+=1
				line = line.strip()

				if dir_src_2 is not None:
					line_2 = src_2.readline()
					line_2 = line_2.strip()
					doc_2 = tok_2(line_2)
				else:
					doc_2=doc

				doc = tok(line)
				
				if len(doc.sentences)>0 and len(doc_2.sentences)>0:
					sentence = doc.sentences[0]
					sentence_2 = doc_2.sentences[0]
				else:
					print(f"Skipping '{line}' and '{line_2} 'cause doc.sentences empty list : line_ind `{line_ind}`")
					continue
				
				to_write = " ".join([token.text for token in sentence.tokens])
				if dir_src_2 is not None:
					to_write_2 = " ".join([token.text for token in sentence_2.tokens])
				target.write(to_write+"\n")
				if dir_src_2 is not None:
					target_2.write(to_write_2+"\n")
				assert "\n" not in to_write_2, f"ERROR skip line found in to_write line {line_ind}"
				assert "\n" not in to_write, f"ERROR skip line found in to_writ {line_ind}"
				if line_ind%100==0:
					end = time.time()
					print(line_ind, f" processed {end-start} s", )

	target_2.close()
	print(f"{dir_src} to {dir_target}")



if __name__ == "__main__":
	import argparse
	args = argparse.ArgumentParser()
	args.add_argument("--dir_src", type=str, required=True)
	args.add_argument("--dir_src_2", type=str, default=None, required=False)
	args.add_argument("--lang_1", type=str, required=True)
	args.add_argument("--lang_2", type=str, required=False, default=None)
	args = args.parse_args()
	
	dir_src = args.dir_src
	lang = args.lang_1
	dir_target=dir_src+".tok.txt"
	
	if args.dir_src_2 is not None:
		dir_src_2 = args.dir_src_2
		dir_target_2= args.dir_src_2+".tok.txt"
	else:
		dir_src_2 = None
		dir_target_2=None


	tokenize(dir_src, dir_target,lang_1=lang, dir_src_2=dir_src_2, dir_target_2=dir_target_2, lang_2=args.lang_2)
	print("SRC", dir_src)
	print("LANG ", lang, " tokenized to ", dir_target)

	#list of  codes : en-en_ru, ru-en_ru
	# 

	if False:
		for lang in ["fr", "de", "tr"]:# , "ru", "id", "ar"]:


			loc="/data/almanach/user/bemuller/projects/mt"
			print("starting lang ", lang)
			# src 
			dir_src = loc+f"/{lang}-en_{lang}-raw-sample.txt"
			dir_target = loc+f"/{lang}-en_{lang}-raw-tokenized.txt"
			tokenize(dir_src, dir_target, lang)
			# target 
			dir_src = loc+f"/en-en_{lang}-raw-sample.txt"
			dir_target = loc+f"/en-en_{lang}-raw-tokenized.txt"
			tokenize(dir_src, dir_target, "en")
			print("end lang ", lang)
