import pandas as pd
import os
import torch

import datasets
from datasets import Dataset

from sklearn.model_selection import StratifiedShuffleSplit


LABEL2ID = {"entailment" : 0, "contradiction" : 1, "neutral" : 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def load_train_data(args, path, ratio = 0.2):
    train_df = pd.read_csv(os.path.join(path, 'train_data.csv'))
    train_df['label'] = train_df['label'].map(LABEL2ID)

    if args.do_eval:
        print("######### Loading train & valid dataset #########")
        shuffle = StratifiedShuffleSplit(n_splits=1, test_size=ratio)

        for train_idx, valid_idx in shuffle.split(train_df, train_df['label']):
            train_data = train_df.loc[train_idx]
            valid_data = train_df.loc[valid_idx]
        
        print(f"#### train dataset length : {len(train_data)} ####")
        print(f"#### validation dataset length : {len(valid_data)} ####")
        
        return Dataset.from_pandas(train_data), Dataset.from_pandas(valid_data)
    else :
        print("######### Loading full train dataset #########")
        print(f"#### train dataset length : {len(train_df)} ####")
        train_df = train_df.sample(frac=1).reset_index(drop=True)
        train_df['index'] = train_df.index
        return Dataset.from_pandas(train_df), None
    
def load_test_data(path):
    print("######### Loading test dataset #########")
    test_df = pd.read_csv(os.path.join(path, 'test_data.csv'))
    test_df['label'] = 0
    return test_df, Dataset.from_pandas(test_df)


def preprocess_function(examples : datasets, tokenizer, args) -> datasets:
    premise = examples['premise']
    hypothesis = examples['hypothesis']
    label = examples['label']

    if args.use_SIC:
        input_ids = tokenizer(premise, hypothesis, truncation=True, return_token_type_ids = False)['input_ids']
        length = [len(one_input) for one_input in input_ids]
        model_inputs = {'input_ids':input_ids, 'labels':label, 'length':length}
    else :
        model_inputs = tokenizer(premise, hypothesis, truncation=True, padding=True, return_token_type_ids = False)
        model_inputs['labels'] = label

    return model_inputs