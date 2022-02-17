import os
import pandas as pd
import torch
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
from model import ExplainableModel

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
    
    tokenizer = AutoTokenizer.from_pretrained(data_args.save_path)
    model = ExplainableModel.from_pretrained(data_args.save_path) if training_args.use_SIC else AutoModelForSequenceClassification.from_pretrained(data_args.save_path)
    
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
    
if __name__ == "__main__" :
    main()