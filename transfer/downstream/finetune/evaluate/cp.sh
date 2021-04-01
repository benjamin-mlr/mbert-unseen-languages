

target_folder="./reporting/ablation_ner/camembert_oscar_subword_seed_3/"


mkdir $target_folder
for var in "10058186-63326-10058186_job-6554f_model"
do 
for type in "prediction" "gold"  
	do 
		for set in dev test 
		do 
			scp neff:/data/almanach/user/bemuller/projects/mt_norm_parse/checkpoints/bert/$var/predictions/LAST_ep-$type-*$set*"" $target_folder
		done
	done
done

echo "cp $var model to $target_folder"