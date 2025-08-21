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
    def __init__(self, *args, alpha=0.5, correction_weights: List[float] = None, **kwargs):
        """
        A custom trainer that uses a weighted loss.
        
        Args:
            alpha (float): The weight for the token prediction loss. 
                           The hallucination loss will be weighted by (1 - alpha).
            correction_weights (List[float]): A list of weights for the 4 correction classes 
                                            (0: no-op, 1: del-w, 2: del-s, 3: del-a).
        """
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        # Convert the list of weights to a tensor and move it to the correct device.
        if correction_weights:
            self.correction_weight_tensor = torch.tensor(correction_weights)
        else:
            self.correction_weight_tensor = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Get the labels from the inputs dictionary, but keep them to pass to the model
        token_labels = inputs.get("labels")
        hallucination_labels = inputs.get("hallucination_labels")

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

        # --- 2. Calculate Hallucination Detection Loss (Cross-Entropy) ---
        # Move the weight tensor to the same device as the logits
        if self.correction_weight_tensor is not None:
            self.correction_weight_tensor = self.correction_weight_tensor.to(hallucination_logits.device)
            
        loss_fct_hallucination = nn.CrossEntropyLoss(
            weight=self.correction_weight_tensor, 
            ignore_index=-100
        )
        
        shift_hallucination_logits = hallucination_logits[..., :-1, :].contiguous()
        shift_hallucination_labels = hallucination_labels[..., 1:].contiguous()
        
        num_correction_classes = shift_hallucination_logits.shape[-1]
        shift_hallucination_logits = shift_hallucination_logits.view(-1, num_correction_classes)
        shift_hallucination_labels = shift_hallucination_labels.view(-1).to(shift_hallucination_logits.device)
        
        hallucination_loss = loss_fct_hallucination(shift_hallucination_logits, shift_hallucination_labels)
        
        # --- 3. Combine the losses with your alpha weighting ---
        custom_loss = self.alpha * token_loss + (1 - self.alpha) * hallucination_loss

        # --- 4. Log Metrics (only on the main process) ---
        if self.state.is_local_process_zero:
            if model.training:
                self.log({
                    "token_loss": token_loss.item(),
                    "hallucination_loss": hallucination_loss.item(),
                })
        return (custom_loss, outputs) if return_outputs else custom_loss