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

@dataclass
class MyTrainingArguments(TrainingArguments):
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
        default=5,
        metadata={
            "help" : "regularizer lambda"
        }
    )
    