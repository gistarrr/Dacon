from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ModelArguments : 
    model_name_or_path: str = field(
        default="klue/roberta-large",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    use_SIC : bool = field(
        default=False
    )

@dataclass
class DataTrainingArguments:
    pass