from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments

@dataclass
class ModelArguments : 
    model_name_or_path: str = field(
        default="klue/roberta-large",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )

@dataclass
class DataTrainingArguments:
    data_name: str = field(
        default="train_data.csv",
    )
    save_path: str = field(
        default="./checkpoints/roberta-large",
        metadata={
            "help": "Path to pretrained model or model identifier from local"
        },
    )
    output_name: str = field(
        default="./output/roberta-large",
        metadata={
            "help": "Path to pretrained model or model identifier from local"
        },
    )
    aeda : bool = field(
        default=False
    )
    k_fold : int = field(
        default=0,
    )

@dataclass
class MyTrainingArguments(TrainingArguments):
    do_eval : bool = field(
        default=False
    )
    use_lstm : bool = field(
        default=False
    )
    use_SIC : bool = field(
        default=False
    )
    use_rdrop : bool = field(
        default=False
    )
    lamb : float = field(
        default=1.0,
        metadata={
            "help" : "regularizer lambda"
        }
    )
    alpha : float = field(
        default=1,
        metadata={
            "help" : "regularizer lambda"
        }
    )
    report_to: Optional[str] = field(
        default='wandb',
    )

@dataclass
class LoggingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dotenv_path: Optional[str] = field(
        default='./wandb.env',
        metadata={"help":'input your dotenv path'},
    )
    project_name: Optional[str] = field(
        default="KLUE-NLI",
        metadata={"help": "project name"},
    )