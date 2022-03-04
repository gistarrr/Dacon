import os
import pandas as pd
import torch
import torch.nn.functional as F
import random
import numpy as np

from functools import partial
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    HfArgumentParser,
    DataCollatorWithPadding,
)

from arguments import ModelArguments, DataTrainingArguments, MyTrainingArguments
from data import load_test_data, preprocess_function
from data_collator import DataCollatorForSIC, DataCollatorWithPadding
from trainer import CustomTrainer
from model import ExplainableModel, RobertaLSTM

PATH = './input'
LABEL2ID = {"entailment" : 0, "contradiction" : 1, "neutral" : 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, MyTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    seed_everything(1)
    
    test_df, test_dataset = load_test_data(PATH)
    
    print(f"#### Test dataset length : {len(test_dataset)} ####")
    
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    
    print(f"#### Tokenized dataset !!! ####")
    
    column_names = test_dataset.column_names
    prep_fn  = partial(preprocess_function, tokenizer=tokenizer, args=training_args)
    
    test_dataset = test_dataset.map(
        prep_fn,
        batched=True,
        num_proc=4,
        remove_columns=column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on test dataset",
    )
    
    data_collator = DataCollatorForSIC() if training_args.use_SIC else DataCollatorWithPadding(tokenizer=tokenizer)
    
    if data_args.k_fold == 0:
        
        if training_args.use_SIC :
            model = ExplainableModel.from_pretrained(data_args.save_path)  
        elif training_args.use_lstm :
            model = RobertaLSTM.from_pretrained(data_args.save_path)  
        else :
            model = AutoModelForSequenceClassification.from_pretrained(data_args.save_path)

        trainer = CustomTrainer(
            model=model,
            args=training_args,  # define metrics function
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
        
        outputs = trainer.predict(test_dataset)
        submission = pd.DataFrame({'index':test_df.index, 'label' : outputs[0].argmax(axis=1)})
        submission['label'] = submission['label'].map(ID2LABEL)
        submission.to_csv(os.path.join('./output', data_args.output_name), index=False)
    else :
        soft_voting = 0
        for i in range(data_args.k_fold):
            print(f"######### Fold : {i+1} !!! ######### ")
            
            fold_path = data_args.save_path + f"_fold_{i+1}"

            if training_args.use_SIC :
                model = ExplainableModel.from_pretrained(fold_path)  
            elif training_args.use_lstm :
                model = RobertaLSTM.from_pretrained(fold_path)  
            else :
                model = AutoModelForSequenceClassification.from_pretrained(fold_path)

            trainer = CustomTrainer(
                model=model,
                args=training_args,  # define metrics function
                data_collator=data_collator,
                tokenizer=tokenizer,
            )
            
            outputs = trainer.predict(test_dataset)
            soft_voting += F.softmax(torch.tensor(outputs[0]), dim=1)
            
            # for hard voting
            submission = pd.DataFrame({'index':test_df.index, 'label' : outputs[0].argmax(axis=1)})
            submission['label'] = submission['label'].map(ID2LABEL)
            
            k_fold_folder_name = os.path.join('./output', data_args.output_name[:-4])
            os.makedirs(k_fold_folder_name, exist_ok=True)
            
            submission.to_csv(os.path.join(k_fold_folder_name, f"fold_{i+1}.csv"), index=False)
        
        soft_submission = pd.DataFrame({'index':test_df.index, 'label' : soft_voting.argmax(axis=1)})
        soft_submission['label'] = soft_submission['label'].map(ID2LABEL)
        soft_submission.to_csv(os.path.join(k_fold_folder_name, data_args.output_name[:-4] + f"_soft_voting.csv"), index=False)
    
if __name__ == "__main__" :
    main()