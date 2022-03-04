import os
import torch
import numpy as np
import random
from dotenv import load_dotenv
from functools import partial

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    HfArgumentParser,
    DataCollatorWithPadding,
    EarlyStoppingCallback    
)
from datasets import load_metric, concatenate_datasets
from sklearn.model_selection import StratifiedKFold

import wandb

from arguments import ModelArguments, DataTrainingArguments, MyTrainingArguments, LoggingArguments
from data import load_train_data, preprocess_function, load_aeda_data, load_train_rtt_data, load_valid_rtt_data
from data_collator import DataCollatorForSIC, DataCollatorWithPadding
from trainer import CustomTrainer
from model import ExplainableModel, RobertaLSTM


XNLI_METRIC = load_metric('xnli')
PATH = './input'
LABEL2ID = {"entailment" : 0, "contradiction" : 1, "neutral" : 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    np.random.default_rng(seed)
    random.seed(seed)

def compute_metrics(EvalPrediction):
    preds, labels = EvalPrediction
    preds = np.argmax(preds, axis=1)

    return XNLI_METRIC.compute(predictions = preds, references = labels)

def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, MyTrainingArguments, LoggingArguments)
    )
    model_args, data_args, training_args, logging_args = parser.parse_args_into_dataclasses()
    
    seed_everything(training_args.seed)
    
    train_dataset, validation_dataset = load_train_data(data_args, PATH)
    
    if data_args.k_fold != 0 :
        total_dataset = concatenate_datasets([train_dataset, validation_dataset]).shuffle(seed=training_args.seed)
        print(f"#### Total dataset length : {len(total_dataset)} ####")
        print("-"*100)
        print(f"#### Example of total dataset : {total_dataset[0]['premise'], total_dataset[0]['hypothesis']} ####")
    elif data_args.k_fold == 0 and training_args.do_eval:
        train_dataset = train_dataset.shuffle(seed=training_args.seed)
        print(f"#### Train dataset length : {len(train_dataset)} ####")
        print(f"#### Validation dataset length : {len(validation_dataset)} ####")
        print("-"*100)
        print(f"#### Example of train dataset : {train_dataset[0]['premise'], train_dataset[0]['hypothesis']} ####")
        print(f"#### Example of validation dataset : {validation_dataset[0]['premise'], validation_dataset[0]['hypothesis']} ####")
    elif not training_args.do_eval:
        train_dataset = concatenate_datasets([train_dataset, validation_dataset]).shuffle(seed=training_args.seed)
        validation_dataset = None
        if data_args.aeda :
            train_dataset = load_aeda_data(train_dataset, PATH)
            print(f"#### Train dataset length : {len(train_dataset)} ####")
        if data_args.train_rtt :
            train_dataset = load_train_rtt_data(train_dataset, PATH)
            print(f"#### Train dataset length : {len(train_dataset)} ####")
        if data_args.valid_rtt :
            train_dataset = load_valid_rtt_data(train_dataset, PATH)
            print(f"#### Train dataset length : {len(train_dataset)} ####")
        print(f"#### Total dataset length : {len(train_dataset)} ####")
        print("-"*100)
        print(f"#### Example of total dataset : {train_dataset[0]['premise'], train_dataset[0]['hypothesis']} ####")
    
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    
    column_names = train_dataset.column_names
    prep_fn  = partial(preprocess_function, tokenizer=tokenizer, args=training_args)
    
    print(f"#### Tokenized dataset !!! ####")

    def model_init():
        if training_args.use_SIC :
            model = ExplainableModel.from_pretrained("leeeki/roberta-large_Explainable")
        elif training_args.use_lstm :
            model = RobertaLSTM.from_pretrained(model_args.model_name_or_path, num_labels=3)
        else :
            model = AutoModelForSequenceClassification.from_pretrained(model_args.model_name_or_path, num_labels=3)
        return model
    
    if data_args.k_fold == 0:
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
        
        # wandb
        load_dotenv(dotenv_path=logging_args.dotenv_path)
        WANDB_AUTH_KEY = os.getenv("WANDB_AUTH_KEY")
        wandb.login(key=WANDB_AUTH_KEY)

        wandb.init(
            entity="leeeki",
            project=logging_args.project_name,
            name=training_args.run_name
        )
        wandb.config.update(training_args)
        
        trainer = CustomTrainer(
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            compute_metrics=compute_metrics,  # define metrics function
            data_collator=data_collator,
            tokenizer=tokenizer,
            model_init = model_init,
            # callbacks = [EarlyStoppingCallback(early_stopping_patience=5)] if training_args.do_eval else None
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
            
    else:
        skf = StratifiedKFold(n_splits=data_args.k_fold, shuffle=True)
    
        for i, (train_idx, valid_idx) in enumerate(skf.split(total_dataset, total_dataset['label'])):
            # if i == 0 or i==1 or i==2 :
            #     continue
            print(f"######### Fold : {i+1} !!! ######### ")
            train_fold = total_dataset.select(train_idx.tolist())
            
            if data_args.aeda :
                train_fold = load_aeda_data(train_fold, PATH)
                print(f"#### Train dataset length : {len(train_fold)} ####")
            if data_args.train_rtt :
                train_fold = load_train_rtt_data(train_fold, PATH)
                print(f"#### Train dataset length : {len(train_fold)} ####")
            if data_args.valid_rtt :
                train_fold = load_valid_rtt_data(train_fold, PATH)
                print(f"#### Train dataset length : {len(train_fold)} ####")
            
            valid_fold = total_dataset.select(valid_idx.tolist())
            
            train_fold = train_fold.map(
                prep_fn,
                batched=True,
                num_proc=4,
                remove_columns=column_names,
                load_from_cache_file=False,
                desc="Running tokenizer on train dataset",
            )
            valid_fold = valid_fold.map(
                prep_fn,
                batched=True,
                num_proc=4,
                remove_columns=column_names,
                load_from_cache_file=False,
                desc="Running tokenizer on train dataset",
            )
            
            data_collator = DataCollatorForSIC() if training_args.use_SIC else DataCollatorWithPadding(tokenizer=tokenizer)
            
            # wandb
            load_dotenv(dotenv_path=logging_args.dotenv_path)
            WANDB_AUTH_KEY = os.getenv("WANDB_AUTH_KEY")
            wandb.login(key=WANDB_AUTH_KEY)

            wandb.init(
                entity="leeeki",
                project=logging_args.project_name,
                name=training_args.run_name + f"_fold_{i+1}"
            )
            wandb.config.update(training_args)
            
            trainer = CustomTrainer(
                args=training_args,
                train_dataset=train_fold,
                eval_dataset=valid_fold,
                compute_metrics=compute_metrics,  # define metrics function
                data_collator=data_collator,
                tokenizer=tokenizer,
                model_init = model_init,
                # callbacks = [EarlyStoppingCallback(early_stopping_patience=5)] if training_args.do_eval else None
            )
            
            if training_args.do_train:
                train_result = trainer.train()
                
                default_path = trainer.args.output_dir
                
                print("######### Train result: ######### ", train_result)
                trainer.args.output_dir = data_args.save_path + f"_fold_{i+1}"
                
                trainer.save_model()
                
                metrics = train_result.metrics
                metrics["train_samples"] = len(train_fold)
                trainer.log_metrics("train", metrics)
                trainer.save_metrics("train", metrics)
                trainer.save_state()
                
            if training_args.do_eval:
                metrics = trainer.evaluate()
                metrics["eval_samples"] = len(valid_fold)

                trainer.log_metrics("eval", metrics)
                trainer.save_metrics("eval", metrics)

            trainer.args.output_dir = default_path
            wandb.finish()
            
        
    
    
if __name__ == '__main__':
    main()

