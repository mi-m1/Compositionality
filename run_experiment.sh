#!/bin/bash
source venv/bin/activate

dir_examples=./in_ncs
dir_bert_estimates=./out_compos_bert
df_estimates=estimates.pkl
df_eval=eval.pkl
#df_gold=gold.pkl         # this contains all rows
df_gold=gold_test.pkl     # only contains car park and head hunter (the test instances)
log_file=log.txt

# Model corpus examples
python3 compos_bert.py $dir_examples $dir_bert_estimates --min 0 --max 12 --pooling avg sum --len 3 20 --num 10 --no-preproc 2>>$log_file

# Get estimates
python3 compos_estimates.py $dir_bert_estimates $df_estimates 2>>$log_file

# Evaluate on gold standard ratings
python3 eval.py $df_estimates $df_gold $df_eval 2>>$log_file