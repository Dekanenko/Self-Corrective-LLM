from transformers import Trainer, PreTrainedTokenizerBase
import torch.nn as nn
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from sklearn.metrics import f1_score

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
    def __init__(self, *args, alpha=0.5, pos_weight=1.0, **kwargs):
        """
        A custom trainer that uses a weighted loss.
        
        Args:
            alpha (float): The weight for the token prediction loss. 
                           The hallucination loss will be weighted by (1 - alpha).
            pos_weight (float): The positive weight for the BCE loss.
        """
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.pos_weight = pos_weight
        # A list to store component losses during evaluation
        self._eval_losses = []

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Pop the labels from the inputs dictionary so they aren't passed to the model
        token_labels = inputs.pop("labels")
        hallucination_labels = inputs.pop("hallucination_labels")

        outputs = model(**inputs)
        token_logits = outputs.get("logits")
        hallucination_logits = outputs.get("hallucination_logits")
    
        # --- 1. Calculate Token Prediction Loss (Cross-Entropy) ---
        loss_fct_token = nn.CrossEntropyLoss(ignore_index=-100)
        
        shift_logits = token_logits[..., :-1, :].contiguous()
        shift_labels = token_labels[..., 1:].contiguous()
        
        shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
        shift_labels = shift_labels.view(-1).to(shift_logits.device)
        
        token_loss = loss_fct_token(shift_logits, shift_labels)

        # --- 2. Calculate Hallucination Detection Loss (Binary Cross-Entropy) ---
        pos_weight_tensor = torch.tensor(self.pos_weight).to(token_logits.device)
        loss_fct_hallucination = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

        shift_hallucination_logits = hallucination_logits[..., :-1, :].contiguous().view(-1)
        shift_hallucination_labels = hallucination_labels[..., 1:].contiguous().view(-1)
        shift_hallucination_labels = shift_hallucination_labels.to(shift_hallucination_logits.device)
        
        # Manually filter out the ignored indices (-100) for BCELoss.
        active_loss_mask = shift_hallucination_labels != -100
        
        active_logits = shift_hallucination_logits[active_loss_mask]
        active_labels = shift_hallucination_labels[active_loss_mask].float()

        if active_labels.numel() > 0:
            hallucination_loss = loss_fct_hallucination(active_logits, active_labels)
        else:
            # If there are no active labels, the loss is 0 for this batch.
            # Ensure the loss tensor is on the correct device.
            hallucination_loss = torch.tensor(0.0).to(shift_logits.device)
        
        # --- 3. Combine the losses with your alpha weighting ---
        custom_loss = self.alpha * token_loss + (1 - self.alpha) * hallucination_loss

        # --- 4. Log Metrics (only on the main process) ---
        if self.state.is_local_process_zero:
            # During training, we only need to log the component losses.
            if model.training:
                self.log({
                    "token_loss": token_loss.item(),
                    "hallucination_loss": hallucination_loss.item(),
                })
            # During evaluation, we log the losses AND compute/log the F1 score.
            else:
                preds = torch.sigmoid(active_logits) > 0.5
                active_preds = preds.cpu().numpy().astype(int)
                active_labels_np = active_labels.cpu().numpy().astype(int)

                f1 = 0.0
                if len(active_labels_np):
                    f1 = f1_score(active_labels_np, active_preds, zero_division=0)
                
                self.log({
                    "eval_token_loss": token_loss.item(),
                    "eval_hallucination_loss": hallucination_loss.item(),
                    "eval_f1_score": f1,
                })
        
        return (custom_loss, outputs) if return_outputs else custom_loss