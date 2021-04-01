#!/usr/bin/env bash
#dir="/data/almanach/user/bemuller/projects/data/oscar"
dir="/data/almanach/user/bemuller/projects/data/wikiner"
#dir="/Users/bemuller/Documents/Work/INRIA/dev/parsing/data/wikiner"
#dir="/data/almanach/user/bemuller/projects/data/Universal-Dependencies-2.4"
script="/home/bemuller/projects/transfer/transfer/transliteration"

#target_lang="ku"
target_lang="la"

for set in "test" "train" "dev"  ; do
	for lang_src in "ar"; do
		#set="test"
		#lang_src="ug"
		#data_set_name_src="ug_dedup"
		data_set_name_src="ar_wiki"
		type="-wikiner"
		#data_set_name_src=${lang_src}_dedup
		#repo="oscar"
		#postfix="txt"
		#postfix="dedup"
		postfix="conll"
		echo "Transliterrate $lang_src file $data_set_name_src set $set"
		#target_dir=$dir/${data_set_name_src}_translit-$repo-$set.$postfix
		#src_dir=$dir/$data_set_name_src-$repo-$set.$postfix
		target_dir=$dir/${data_set_name_src}_translit$type-$set.$postfix
		src_dir=$dir/$data_set_name_src$type-$set.$postfix
		
		touch $target_dir
		chgrp scratch $target_dir
		cat $src_dir | perl $script/alTranscribe.pl -f $lang_src -t $target_lang > $target_dir
		n_count=`wc -l $target_dir`
		echo "$src_dir transliterated to $target_dir $n_count>0 ?"
		#cat $dir/ug_translit_dedup-$set.txt  $dir/ug_dedup-$set.txt > $dir/ug_ug_translit_dedup-$set.txt
	done
done 