import os
from sklearn.decomposition import TruncatedSVD
import torch
import numpy as np
import random

from functools import partial
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    HfArgumentParser,
    DataCollatorWithPadding,
    EarlyStoppingCallback    
)

from datasets import load_metric

from arguments import ModelArguments, DataTrainingArguments, MyTrainingArguments
from data import load_train_data, preprocess_function
from data_collator import DataCollatorForSIC, DataCollatorWithPadding
from trainer import CustomTrainer
from model import ExplainableModel


XNLI_METRIC = load_metric('xnli')
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

def compute_metrics(EvalPrediction):
    preds, labels = EvalPrediction
    preds = np.argmax(preds, axis=1)

    return XNLI_METRIC.compute(predictions = preds, references = labels)

def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, MyTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    seed_everything(1)
    
    train_dataset, validation_dataset = load_train_data(training_args, PATH)
    print(f"#### train dataset length : {len(train_dataset)} ####")
    print(f"#### validation dataset length : {len(validation_dataset)} ####")

    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    config.num_labels = 3
    
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    model = ExplainableModel.from_pretrained(model_args.model_name_or_path, config=config) if training_args.use_SIC else AutoModelForSequenceClassification.from_pretrained(model_args.model_name_or_path, config=config)
    print(model)
    
    column_names = train_dataset.column_names
    prep_fn  = partial(preprocess_function, tokenizer=tokenizer, args=training_args)
    
    print(f"#### Tokenized dataset !!! ####")

    train_dataset = train_dataset.map(
        prep_fn,
        batched=True,
        num_proc=4,
        remove_columns=column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on train dataset",
    )

    if training_args.do_eval:
        validation_dataset = validation_dataset.map(
            prep_fn,
            batched=True,
            num_proc=4,
            remove_columns=column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on validation dataset",
        )
        
    data_collator = DataCollatorForSIC() if training_args.use_SIC else DataCollatorWithPadding(tokenizer=tokenizer)
    
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        compute_metrics=compute_metrics,  # define metrics function
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=5)] if training_args.do_eval else None
    )
    
    if training_args.do_train:
        train_result = trainer.train()
        print("######### Train result: ######### ", train_result)
        trainer.args.output_dir = data_args.save_path
        
        trainer.save_model()
        
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        
    if training_args.do_eval:
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(validation_dataset)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    
    
    
if __name__ == '__main__':
    main()

