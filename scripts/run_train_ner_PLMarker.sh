# This is PL-Marker's NER model, you can follow their approach to get NER results or use HGERE's NER results. 
# Our RE model(SURE) is based on PL-Marker's ACE04 NER results and HGERE's ACE05/SciERC NER results.
# HGERE model(https://github.com/yanzhh/HGERE) has published their NER results (it can be found in /saves) and you can easily use it!


GPU_ID=0

# For ALBERT-xxlarge, change learning_rate from 2e-5 to 1e-5

# ACE05
mkdir ace05ner_models
for seed in 42 43 44 45 46; do 
CUDA_VISIBLE_DEVICES=$GPU_ID  python3  run_acener.py  --model_type bertspanmarker  \
    --model_name_or_path  bert_models/bert-base-uncased  --do_lower_case  \
    --data_dir ace05  \
    --learning_rate 2e-5  --num_train_epochs 10  --per_gpu_train_batch_size  8  --per_gpu_eval_batch_size 16  --gradient_accumulation_steps 1  \
    --max_seq_length 512  --save_steps 2500  --max_pair_length 256  --max_mention_ori_length 8    \
    --do_train  --do_eval  --evaluate_during_training   --eval_all_checkpoints  \
    --fp16  --seed $seed  --onedropout  --lminit  \
    --train_file train.json --dev_file dev.json --test_file test.json  \
    --output_dir ace05ner_models/PL-Marker-ace05-bert-$seed  --overwrite_output_dir  --output_results
done;
Average the scores
python3 sumup.py ace05ner PL-Marker-ace05-bert


# SciERC
mkdir sciner_models
for seed in 42 43 44 45 46; do 
CUDA_VISIBLE_DEVICES=$GPU_ID  python3  run_acener.py  --model_type bertspanmarker  \
    --model_name_or_path  bert_models/scibert-uncased  --do_lower_case  \
    --data_dir scierc  \
    --learning_rate 2e-5  --num_train_epochs 50  --per_gpu_train_batch_size  8  --per_gpu_eval_batch_size 16  --gradient_accumulation_steps 1  \
    --max_seq_length 512  --save_steps 2000  --max_pair_length 256  --max_mention_ori_length 8    \
    --do_train  --do_eval  --evaluate_during_training   --eval_all_checkpoints  \
    --fp16  --seed $seed  --onedropout  --lminit  \
    --train_file train.json --dev_file dev.json --test_file test.json  \
    --output_dir sciner_models/PL-Marker-scierc-scibert-$seed  --overwrite_output_dir  --output_results
done;
# Average the scores
python3 sumup.py sciner PL-Marker-ace05-bert


# ACE04
mkdir ace04ner_models
for data_spilt in 0 1 2 3 4; do 
CUDA_VISIBLE_DEVICES=$GPU_ID  python3  run_acener.py  --model_type bertspanmarker  \
    --model_name_or_path  bert_models/bert-base-uncased  --do_lower_case  \
    --data_dir ace04  \
    --learning_rate 2e-5  --num_train_epochs 15  --per_gpu_train_batch_size  8  --per_gpu_eval_batch_size 16  --gradient_accumulation_steps 1  \
    --max_seq_length 512  --save_steps 2500  --max_pair_length 256  --max_mention_ori_length 8    \
    --do_train  --do_eval  --evaluate_during_training   --eval_all_checkpoints  \
    --fp16  --seed 42  --onedropout  --lminit  \
    --train_file train/$data_spilt.json  --dev_file dev/$data_spilt.json  --test_file test/$data_spilt.json\
    --output_dir ace04ner_models/PL-Marker-ace04-bert-$data_spilt  --overwrite_output_dir  --output_results
done;
# Average the scores
python3 sumup.py ace04ner PL-Marker-ace05-bert










