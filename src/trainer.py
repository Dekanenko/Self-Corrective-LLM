from transformers import Trainer, PreTrainedTokenizerBase
import torch.nn as nn
import torch
from dataclasses import dataclass
from typing import Any, Dict, List

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
    def __init__(self, *args, alpha=0.5, **kwargs):
        """
        A custom trainer that uses a weighted loss.
        
        Args:
            alpha (float): The weight for the token prediction loss. 
                           The hallucination loss will be weighted by (1 - alpha).
        """
        super().__init__(*args, **kwargs)
        self.alpha = alpha

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Pop the labels from the inputs dictionary so they aren't passed to the model
        token_labels = inputs.pop("labels")
        hallucination_labels = inputs.pop("hallucination_labels")

        outputs = model(**inputs)
        token_logits = outputs.get("logits")
        hallucination_probs = outputs.get("p_hall")
    
        # --- 1. Calculate Token Prediction Loss (Cross-Entropy) ---
        loss_fct_token = nn.CrossEntropyLoss(ignore_index=-100)
        
        shift_logits = token_logits[..., :-1, :].contiguous()
        shift_labels = token_labels[..., 1:].contiguous()
        
        shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
        shift_labels = shift_labels.view(-1).to(shift_logits.device)
        
        token_loss = loss_fct_token(shift_logits, shift_labels)

        # --- 2. Calculate Hallucination Detection Loss (Binary Cross-Entropy) ---
        loss_fct_hallucination = nn.BCELoss()

        shift_hallucination_probs = hallucination_probs[..., :-1, :].contiguous().view(-1)
        shift_hallucination_labels = hallucination_labels[..., 1:].contiguous().view(-1)
        shift_hallucination_labels = shift_hallucination_labels.to(shift_hallucination_probs.device)
        
        # Manually filter out the ignored indices (-100) for BCELoss.
        active_loss_mask = shift_hallucination_labels != -100
        
        active_probs = shift_hallucination_probs[active_loss_mask]
        active_labels = shift_hallucination_labels[active_loss_mask].float()

        if active_labels.numel() > 0:
            hallucination_loss = loss_fct_hallucination(active_probs, active_labels)
        else:
            # If there are no active labels, the loss is 0 for this batch.
            # Ensure the loss tensor is on the correct device.
            hallucination_loss = torch.tensor(0.0).to(shift_logits.device)
        
        # --- 3. Combine the losses with your alpha weighting ---
        custom_loss = self.alpha * token_loss + (1 - self.alpha) * hallucination_loss
        
        return (custom_loss, outputs) if return_outputs else custom_loss