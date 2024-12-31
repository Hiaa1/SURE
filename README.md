# SURE
Source code for COLING 2025 long paper "SURE: Mutually Visible Objects and Self-generated Candidate Labels For Relation Extraction"
## Setup
### Install Dependencies/Prepare the datasets
Please follow [PL-Marker](https://github.com/thunlp/PL-Marker) to install dependencies and download the datasets including ACE04/ACE05/SciERC.

**NOTE**: Please run those commands in our directory, because editable transformers must be installed in our directory

### Download Pre-trained Language Models
Please download SciBert, Bert, Albert-xxlarge from huggingface, and place them in the folder "bert_models"

## Quickstart
You can download trained models on [google drive](https://drive.google.com/drive/folders/1SGgKHo6GTJifYRlPfhe5ZVUGqaOGxgcS?usp=sharing). And you can use this trained model to evaluate.

**NOTE**: Since we report the average numbers based on 5 seeds, the performance of models might be a little different. Also the model on google drive is based on PL-Marker NER results while our best SciERC score is based on HGERE NER results. We advise you just run our models with PL-Marker NER results(follow our settings) first, and continuely apply HGERE NER results(change some file directories) to evaluate our RE model.

```python
CUDA_VISIBLE_DEVICES=0 python3  run_sure.py  --model_type bertsub  \
    --model_name_or_path  bert_models/scibert_scivocab_uncased  --do_lower_case  \
    --data_dir scierc  \
    --learning_rate 2e-5  --num_train_epochs 20  --per_gpu_train_batch_size  8  --per_gpu_eval_batch_size 16  --gradient_accumulation_steps 1  \
    --max_seq_length 256  --max_pair_length 16  --save_steps 2500  \
    --do_eval  --evaluate_during_training   --eval_all_checkpoints  --eval_logsoftmax  \
    --fp16  --seed 42      \
    --test_file sciner_models/PL-Marker-scierc-scibert-42/ent_pred_test.json  \
    --use_ner_results \
    --output_dir scire_models/scire-scibert-42-epoch20-n2-m4  --overwrite_output_dir \
    --candidate_top_n 2 --candidate_worst_m 4 \
    --att_left --att_right \
    --st1_warming_up 0
```


## Training
Train PL-Marker's NER Models:
```
bash scripts/run_train_ner_PLMarker.sh
```
Train Our RE Models:
```
bash scripts/run_train/re.sh
```


