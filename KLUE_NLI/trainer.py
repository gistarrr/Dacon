from typing import Any, Dict, Union, Optional, List, Tuple

import torch
import torch.nn as nn
import inspect
import datasets
import os
import json

from transformers import Trainer

from packaging import version

def nested_detach(tensors):
    "Detach `tensors` (even if it's a nested list/tuple of tensors)."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_detach(t) for t in tensors)
    return tensors.detach()

def is_sagemaker_mp_enabled():
    # Get the sagemaker specific mp parameters from smp_options variable.
    smp_options = os.getenv("SM_HP_MP_PARAMETERS", "{}")
    try:
        # Parse it and check the field "partitions" is included, it is required for model parallel.
        smp_options = json.loads(smp_options)
        if "partitions" not in smp_options:
            return False
    except json.JSONDecodeError:
        return False

class CustomTrainer(Trainer):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def _remove_unused_columns(self, dataset: "datasets.Dataset", description: Optional[str] = None):
        if not self.args.remove_unused_columns:
            return dataset
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            # Labels may be named label or label_ids, the default data collator handles that.
            self._signature_columns += ["length", "labels"]
            self._signature_columns += ["label", "label_ids"]
        columns = [k for k in self._signature_columns if k in dataset.column_names]
        ignored_columns = list(set(dataset.column_names) - set(self._signature_columns))
        if len(ignored_columns) > 0:
            dset_description = "" if description is None else f"in the {description} set "

        if version.parse(datasets.__version__) < version.parse("1.4.0"):
            dataset.set_format(
                type=dataset.format["type"], columns=columns, format_kwargs=dataset.format["format_kwargs"]
            )
            return dataset
        else:
            return dataset.remove_columns(ignored_columns)
        
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        if self.args.use_SIC:        
            model.train()
            inputs = self._prepare_inputs(inputs)

            with self.autocast_smart_context_manager():
                loss = self.compute_loss(model, inputs)

            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
                # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
                loss = loss / self.args.gradient_accumulation_steps
                
            if self.do_grad_scaling:
                self.scaler.scale(loss).backward()
            elif self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            elif self.deepspeed:
                # loss gets scaled under gradient_accumulation_steps in deepspeed
                loss = self.deepspeed.backward(loss)
            else:
                loss.backward()

            return loss.detach()
        
        else :
            return super().training_step(model, inputs)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        # SIC Model을 사용하는 경우
        if self.args.use_SIC :    
            if "labels" in inputs:
                labels = inputs.pop("labels").view(-1)
            else:
                labels = None
            outputs = model(**inputs)
            # Save past state if it exists
            # TODO: this needs to be fixed and made cleaner later.
            if self.args.past_index >= 0:
                self._past = outputs[self.args.past_index]

            if labels is not None:
                logits, a_ij = outputs
                loss_fct = nn.CrossEntropyLoss()
                ce_loss = loss_fct(logits, labels)
                reg_loss = self.args.lamb * a_ij.pow(2).sum(dim=1).mean()
                loss = ce_loss - reg_loss
            else:
                # We don't use .loss here since the model may return tuples instead of ModelOutput.
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            return (loss, outputs) if return_outputs else loss        
        else:
            return super().compute_loss(model, inputs, return_outputs)
        
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.
        Subclass and override to inject custom behavior.
        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`Lst[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        if self.args.use_SIC:
            has_labels = all(inputs.get(k) is not None for k in self.label_names)
            inputs = self._prepare_inputs(inputs)
            if ignore_keys is None:
                if hasattr(self.model, "config"):
                    ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
                else:
                    ignore_keys = []

            # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
            if has_labels:
                labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
                if len(labels) == 1:
                    labels = labels[0]
            else:
                labels = None

            with torch.no_grad():
                if has_labels:
                    with self.autocast_smart_context_manager():
                        loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                    loss = loss.mean().detach()

                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        logits = outputs[0]
                else:
                    loss = None
                    with self.autocast_smart_context_manager():
                        outputs = model(**inputs)
                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                    else:
                        logits = outputs
                    # TODO: this needs to be fixed and made cleaner later.
                    if self.args.past_index >= 0:
                        self._past = outputs[self.args.past_index - 1]

            if prediction_loss_only:
                return (loss, None, None)

            logits = nested_detach(logits)
            if len(logits) == 1:
                logits = logits[0]
            return (loss, logits, labels)
        
        else :
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)