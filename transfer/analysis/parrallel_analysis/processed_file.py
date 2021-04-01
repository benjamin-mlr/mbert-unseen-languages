
import os
def process(merged, src, target, filter=None):

	"""
	two raw txt aligned files into a unique ||| seperated file
	:return:
	"""
	ind = 0

	src_file = open(src, "r")
	target_file = open(target, "r")
	print(f"merging from {src_file} and {target_file} to {merged}")
	with open(merged,"a") as f:
		i_sent=0
		for line_src in src_file:

	#while True:
		#try:
			i_sent+=1
			#line_src = src_file.readline().strip()
			line_src = line_src.strip()
			line_target = target_file.readline().strip()
			ind += 1
			if ind % 100000 == 0:
				print(f"{ind} proocessed line_src {line_src+' ||| '+line_target}")
		#except Exception as e:
	#		print(e)
			#break
			f.write(line_src+" ||| "+line_target+"\n")

	print(f"{merged} written {i_sent}")


def tokenized_conll(src, target):

	with open(target, "a") as writing:

		with open(src, "r") as f:
			i_sent = 0
			for line in f:
				line = line.strip()
				if line.startswith("# sent_id"):
					if i_sent > 0:
						sent += "\n"
						writing.write(sent)
					i_sent += 1
					sent = ""
				elif not line.startswith("# ") and len(line)>0:
					line = line.split("\t")
					if "-" not in line[0]:
						#print(line)
						word = line[1]
						
						sent += word+" "

	print(f"{src} writted based on {target} {i_sent} written ")


if __name__=="__main__":
	root = "/data/almanach/user/bemuller/projects/mt"
	src = root + "/MultiUN4WMT12/undoc.2000.fr-en.en"
	target = root + "/MultiUN4WMT12/undoc.2000.fr-en.fr"
	merged = root + "/MultiUN4WMT12/merged-en.fr.txt"
	
	preprocess = True
	merge = False

	if preprocess:
		#for lang in ["en","fr", "de", "ru", "tr", "id", "ar"]:
		for lang in [ "pt_pud",  "es_pud", "fi_pud",  "it_pud", "sv_pud", "cs_pud", "pl_pud", "hi_pud", "zh_pud", "ko_pud", "ja_pud","th_pud"]:
		#for lang in ["en"]:
			src = os.environ.get("DATA_UD")+f"/{lang}-ud-test.conllu"
			src_2 = f"/data/almanach/user/bemuller/projects/mt/{lang}_pud-ud-test-tok.txt"
			print("start writing ", src_2)
			tokenized_conll(src, src_2)
			print("done ", src_2)

	if merge:	
		for lang in ["ru"]:#["fr", "de", "ru", "tr", "id", "ar"]:
			src_2 = "/data/almanach/user/bemuller/projects/mt/ru-en_ru_pud-ud-test-tok.txt"#f"/data/almanach/user/bemuller/projects/mt/en_pud-ud-test-tok.txt"
			target = "/data/almanach/user/bemuller/projects/mt/en-en_ru_pud-ud-test-tok.txt"#f"/data/almanach/user/bemuller/projects/mt/{lang}_pud-ud-test-tok.txt"
			merged = f"/data/almanach/user/bemuller/projects/mt/en_{lang}_all_pud-ud-test-tok-merged.txt"#f"/data/almanach/user/bemuller/projects/mt/en_{lang}_pud-ud-test-tok-merged.txt"
			process(merged, src_2, target, filter="# text =")
