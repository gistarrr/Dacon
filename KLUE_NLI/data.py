import pandas as pd
import os
import torch

import datasets
from datasets import Dataset, concatenate_datasets
from sklearn.model_selection import StratifiedShuffleSplit


LABEL2ID = {"entailment" : 0, "contradiction" : 1, "neutral" : 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

def load_aeda_data(train_df, path):
    print("######### Adding aeda data #########")
    aeda_df = pd.read_csv(os.path.join(path,'train_aeda.csv'))
    aeda_df['label'] = aeda_df['label'].map(LABEL2ID)
    print(f"#### Aeda dataset length : {len(aeda_df)} ####")
    if isinstance(train_df, pd.DataFrame):
        train_df = pd.concat([train_df, aeda_df]).reset_index(drop=True)
        train_df['index'] = train_df.index
    elif isinstance(train_df, Dataset):
        aeda_dataset = Dataset.from_pandas(aeda_df)
        train_df = concatenate_datasets([train_df, aeda_dataset]).shuffle(seed=42)
    return train_df

def load_train_data(args, path):
    train_df = pd.read_csv(os.path.join(path, args.data_name))
    valid_df = pd.read_csv(os.path.join(path, 'validation.csv'))
    train_df['label'] = train_df['label'].map(LABEL2ID)
    valid_df['label'] = valid_df['label'].map(LABEL2ID)

    if args.k_fold == 0 :
        if args.aeda :
            train_df = load_aeda_data(train_df, path)
        
    return Dataset.from_pandas(train_df), Dataset.from_pandas(valid_df)
    
    # else :
    #     print("######### Loading full train dataset #########")
    #     train_df = pd.concat([train_df, valid_df])
        
    #     print(f"#### Train dataset length : {len(train_df)} ####")
    #     train_df = train_df.reset_index(drop=True)
    #     train_df['index'] = train_df.index
    #     return Dataset.from_pandas(train_df), None
    
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