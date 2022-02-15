from transformers import Trainer

class SICTrainer(Trainer):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)