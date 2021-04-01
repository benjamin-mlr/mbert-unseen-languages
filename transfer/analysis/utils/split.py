


def split(dir):

	ind=0
	target_dir = dir+f"{ind}.txt"

	dir_ls = []

	target=open(target_dir, "w")

	print("Splitting ", dir)
	with open(dir,"r") as src:
		counter=0
		for line in src:
			counter+=1
			target.write(line)
			if counter%10000==0:
				ind+=1
				print(f"Target file {target_dir} is done moving to ...")
				target_dir = dir+f"{ind}.txt"
				print(f"New Target file {target_dir} is done moving to ...")
				target=open(target_dir, "a")
				dir_ls.append(target_dir)

		print(f"{counter} sent processed")
	print("dir_ls ", " ".join(dir_ls))



if __name__ == "__main__":
	for lang in ["fr", "de", "tr" , "ru"]:#, "id", "ar"]:


		loc="/data/almanach/user/bemuller/projects/mt"
		print("starting lang ", lang)

		print("")
		# src 
		dir_src = loc+f"/{lang}-en_{lang}-raw-sample.txt"
		#dir_target = loc+f"/{lang}-en_{lang}-raw-tokenized.txt"
		#tokenize(dir_src, dir_target, lang)
		# target 
		split(dir_src)
		dir_src = loc+f"/en-en_{lang}-raw-sample.txt"

		split(dir_src)
		#dir_target = loc+f"/en-en_{lang}-raw-tokenized.txt"
		
		#print("end lang ", lang)
