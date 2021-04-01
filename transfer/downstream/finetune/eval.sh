
#!/usr/bin/env bash

data_src="/data/almanach/user/bemuller/projects/data/fr_pred_tokens_conll_2018/pred_filtered/"
gold_src="/data/almanach/user/bemuller/projects/data/fr_pred_tokens_conll_2018/gold_filtered/"

src_model="/data/almanach/user/bemuller/projects/mt_norm_parse/checkpoints/bert/"
task="pos"
dir_pred="/data/almanach/user/bemuller/projects/data/pred"

module load conda 
source activate lm


for args_model in "9990958-66052-9990958_job-c0ed8_model/9990958-66052-9990958_job-c0ed8_model-args.json" ; do
    args_model=$src_model$args_model
    for data in "fr_sequoia" ; do
        data_dir=$data_src$data"-udpipe_multi_filtered.conllu"
        gold=$gold_src$data"-ud-test-filtered.conllu"
        echo "EVAL src $data_dir on gold $gold_dir"
        python $CAMEMBERT_FINE_TUNE/predict.py --test_paths $data_dir  --init_args_dir $args_model --tasks $task --end_predictions $dir_pred > pred.txt
        pred=`grep "CREATING NEW FILE (io_/dat/normalized_writer) :" ./pred.txt | tail -1 | cut -c 49-`
        echo "FOUND pred file $pred"
        echo "EVAL --$gold-- vs --$pred--"
        python $CAMEMBERT_FINE_TUNE/evaluate/conll18_ud_eval.py --verbose $gold $pred  > ./results.txt
        cat ./results.txt
    done 

done




#python ./evaluate/conll18_ud_eval.py   $gold  $pred  --v
