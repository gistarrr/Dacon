import os
from sklearn.decomposition import TruncatedSVD
import torch
import numpy as np
import random

from functools import partial
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    DataCollatorWithPadding,
    EarlyStoppingCallback    
)

from datasets import load_metric

from arguments import ModelArguments, DataTrainingArguments
from data import load_data, preprocess_function
from data_collator import DataCollatorForSIC
from trainer import SICTrainer
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
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    seed_everything(1)

    if training_args.do_eval :
        train_dataset, validation_dataset = load_data(PATH, ratio = 0.2)
    else :
        train_dataset= load_data(PATH, ratio = 0)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    model = ExplainableModel.from_pretrained(model_args.model_name_or_path, num_labels=3) if model_args.use_SIC else AutoModelForSequenceClassification(model_args.model_name_or_path, num_labels=3)

    print(model)

    column_names = train_dataset.column_names
    prep_fn  = partial(preprocess_function, tokenizer=tokenizer, model_args=model_args)
    
    print(model_args.use_SIC)

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

    data_collator = DataCollatorForSIC(tokenizer=tokenizer) if model_args.use_SIC else DataCollatorWithPadding(tokenizer=tokenizer)

    if model_args.use_SIC :
        pass
        # trainer = SICTrainer(

        # )
    else :
        trainer = Trainer(
            model=model,
            args=training_args,

        )


    breakpoint()
    print('end')


if __name__ == '__main__':
    main()

