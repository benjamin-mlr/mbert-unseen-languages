


for code in ["en-en_ru", "ru-en_ru",
				"fr-en_fr","en-en_fr",
				"de-en_de", "en-en_de",
				 "tr-en_tr", "en-en_tr"]:
	script_dir=f"/data/almanach/user/bemuller/projects/mt/script/tok-{code}.py"
	script_dir_shell=f"/data/almanach/user/bemuller/projects/mt/script/tok-shell-{code}.sh"
	with open(script_dir,"w") as script:
		for ind in range(100):
			script.write(f"python /home/bemuller/projects/transfer/transfer/analysis/utils/tokenize_stanza.py --dir_src /data/almanach/user/bemuller/projects/mt/{code}-raw-sample.txt{ind}.txt --lang {code[:2]} \n")

	print(script_dir, "written")
	with open(script_dir_shell,"w") as shell:
		shell.write(f"module load conda\n")
		shell.write(f"source activate stanza\n")
		shell.write(f"sh $EXPERIENCE/train/distribute.sh {script_dir} 16\n")
		shell.write(f"echo {script_dir} done")
	print("shell ", script_dir_shell)

