from transformers import Trainer, PreTrainedTokenizerBase, training_args
import torch.nn as nn
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from sklearn.metrics import f1_score
from transformers.optimization import AdamW, get_scheduler

@dataclass
class SelfCorrectionDataCollator:
    """
    A custom data collator that correctly pads all fields for our self-correction model.
    It pads `input_ids` and `attention_mask` using the tokenizer's padding logic,
    and pads `labels` and our custom `hallucination_labels` with -100.
    """
    tokenizer: PreTrainedTokenizerBase
    label_pad_token_id: int = -100

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        labels = [feature.pop("labels") for feature in features]
        hallucination_labels = [feature.pop("hallucination_labels") for feature in features]

        batch = self.tokenizer.pad(
            features,
            return_tensors="pt",
        )

        max_length = batch['input_ids'].shape[1]
        
        batch['labels'] = torch.tensor([
            l + [self.label_pad_token_id] * (max_length - len(l)) for l in labels
        ])
        
        batch['hallucination_labels'] = torch.tensor([
            hl + [self.label_pad_token_id] * (max_length - len(hl)) for hl in hallucination_labels
        ])

        return batch


class SelfCorrectionTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        """
        A custom trainer that uses a weighted loss.
        
        Args:
            alpha (float): The weight for the token prediction loss. 
                           The hallucination loss will be weighted by (1 - alpha).
            correction_weights (List[float]): A list of weights for the 4 correction classes 
                                            (0: no-op, 1: del-s, 2: del-a).
        """
        # Pop our custom kwarg before passing to the parent to avoid a TypeError.
        self.custom_head_learning_rate = kwargs.pop("custom_head_learning_rate", None)
        super().__init__(*args, **kwargs)

    def create_optimizer(self):
        """
        Overrides the default optimizer creation to enable a differential learning rate.
        """
        if self.optimizer is None:
            optimizer_cls = AdamW
            
            # --- Separate parameters into two groups ---
            lora_params = []
            head_params = []
            
            # Define the names of the custom, fully-trained modules
            head_module_names = [
                "new_token_embeddings", 
            ]

            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                
                is_head_param = any(head_name in name for head_name in head_module_names)
                
                if is_head_param:
                    head_params.append(param)
                else:
                    lora_params.append(param)

            # --- Create optimizer with parameter groups ---
            optimizer_grouped_parameters = [
                {
                    "params": lora_params,
                    "lr": self.args.learning_rate,
                },
                {
                    "params": head_params,
                    "lr": self.custom_head_learning_rate if self.custom_head_learning_rate is not None else self.args.learning_rate,
                },
            ]

            optimizer_kwargs = {
                "betas": (self.args.adam_beta1, self.args.adam_beta2),
                "eps": self.args.adam_epsilon,
            }
            
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            
        return self.optimizer

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        """
        Overrides the default scheduler creation to ensure it works correctly with
        multi-group optimizer.
        """
        if self.lr_scheduler is None:
            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                optimizer=self.optimizer if optimizer is None else optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
            )
        return self.lr_scheduler

    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        """
        Overrides the default logging behavior to add both learning rates to the logs.
        """
        # Add the learning rates from both parameter groups to the logs
        if self.state.is_local_process_zero and self.optimizer is not None:
            if 'learning_rate' in logs:
                logs['lr_lora'] = self.optimizer.param_groups[0]['lr']
                logs['lr_head'] = self.optimizer.param_groups[1]['lr']
                logs.pop('learning_rate') # remove the ambiguous default
            
            # Add custom component losses for training steps
            if 'loss' in logs:
                logs.update(self._last_component_losses)

        super().log(logs, *args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        token_labels = inputs.get("labels")

        outputs = model(**inputs)
        token_logits = outputs.get("logits")

        # --- Calculate Token Prediction Loss (Cross-Entropy) ---
        loss_fct_token = nn.CrossEntropyLoss(ignore_index=-100)
        
        shift_logits = token_logits[..., :-1, :].contiguous()
        shift_labels = token_labels[..., 1:].contiguous()
        
        vocab_size = token_logits.shape[-1]
        shift_logits = shift_logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1).to(shift_logits.device)
        
        token_loss = loss_fct_token(shift_logits, shift_labels)

        return (token_loss, outputs) if return_outputs else token_loss