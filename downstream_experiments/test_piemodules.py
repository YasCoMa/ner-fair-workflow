import os
import sys
import json
import torch
import random
import argparse
import logging

import pandas as pd
from datasets import Dataset, load_dataset, load_from_disk
from transformers import AutoTokenizer

sys.path.append( '/aloy/home/ymartins/match_clinical_trial/ner_subproj/' )
from utils.utils_evaluation import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataDir = '/aloy/home/ymartins/match_clinical_trial/experiments/biobert_trial/preprocessing/dataset_train_valid_test_split_v0.1'
datasets = load_from_disk( dataDir )
label_list = datasets['train'].features[f"ner_tags"].feature.names
idsxlabel = {i: label for i, label in enumerate( label_list)}
labelxids = {label: i for i, label in enumerate( label_list)}

tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.2')
flag_tokenizer = None
# Preprocessing the data
label_all_tokens = True
tokenized_datasets = datasets.map(tokenize_and_align_labels, batched=True, fn_kwargs={"flag_tokenizer": flag_tokenizer, "tokenizer": tokenizer, "label_all_tokens": label_all_tokens })
save_path='/aloy/home/ymartins/match_clinical_trial/experiments/biobert_trial/biobert-base-cased-v1.2-finetuned-ner/'

model_files = [file for file in os.listdir(save_path) if file.startswith("model_")]
model_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

input_ids = torch.tensor(tokenized_datasets["test"]["input_ids"]).to(device)
attention_mask = torch.tensor(tokenized_datasets["test"]["attention_mask"]).to(device)     
test_data = {'input_ids': input_ids, 'attention_mask': attention_mask}

labels = tokenized_datasets['test']['labels']
i=0
model_file = model_files[0]
model = torch.load(f"{save_path}/{model_file}", weights_only=False)
model.to(device)
model.eval()
with torch.no_grad():
    outputs = model(**test_data)

predictions = torch.argmax(outputs.logits, dim=2).to("cpu").numpy()
torch.cuda.empty_cache()