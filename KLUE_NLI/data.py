import pandas as pd
import os
import torch

import datasets
from datasets import Dataset

from sklearn.model_selection import StratifiedShuffleSplit


LABEL2ID = {"entailment" : 0, "contradiction" : 1, "neutral" : 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def load_data(args, path, ratio = 0.2):
    print("######### loading dataset #########")
    train_df = pd.read_csv(os.path.join(path, 'train_data.csv'), index_col = 'index')
    train_df['label'] = train_df['label'].map(LABEL2ID)

    if args.do_eval:
        shuffle = StratifiedShuffleSplit(n_splits=1, test_size=ratio)

        for train_idx, valid_idx in shuffle.split(train_df, train_df['label']):
            train_data = train_df.loc[train_idx]
            valid_data = train_df.loc[valid_idx]

        return Dataset.from_pandas(train_data), Dataset.from_pandas(valid_data)
    else :
        return Dataset.from_pandas(train_data), None

def preprocess_function(examples : datasets, tokenizer, args) -> datasets:
    premise = examples['premise']
    hypothesis = examples['hypothesis']
    label = examples['label']

    # if args.use_SIC:
    #     input_ids = tokenizer.encode(premise, hypothesis)
    #     length = torch.LongTensor([len(input_ids)])
    #     input_ids = torch.LongTensor(input_ids)
    #     label = torch.LongTensor([label])
    #     model_inputs = {'input_ids':torch.LongTensor(input_ids), 'length':length, 'labels':label}
    if args.use_SIC:
        input_ids = tokenizer(premise, hypothesis, truncation=True, return_token_type_ids = False)['input_ids']
        length = [len(one_input) for one_input in input_ids]
        model_inputs = {'input_ids':input_ids, 'labels':label, 'length':length}
    else :
        model_inputs = tokenizer(premise, hypothesis, truncation=True, padding=True, return_token_type_ids = False)
        model_inputs['labels'] = label

    return model_inputs