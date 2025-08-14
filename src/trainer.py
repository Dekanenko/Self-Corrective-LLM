from transformers import Trainer, PreTrainedTokenizerBase
import torch.nn as nn
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

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

        if model.training:
            # --- 4. Log aggregated losses in a distributed setting during training ---
            if self.args.world_size > 1:
                losses_to_reduce = torch.tensor([token_loss.detach(), hallucination_loss.detach()]).to(custom_loss.device)
                torch.distributed.all_reduce(losses_to_reduce, op=torch.distributed.ReduceOp.AVG)
                if self.state.is_local_process_zero:
                    self.log({
                        "token_loss": losses_to_reduce[0].item(),
                        "hallucination_loss": losses_to_reduce[1].item(),
                    })
            else:
                if self.state.is_local_process_zero:
                    self.log({
                        "token_loss": token_loss.item(),
                        "hallucination_loss": hallucination_loss.item(),
                    })
        else:
            # --- During evaluation, just collect the losses on each device ---
            self._eval_losses.append(
                torch.tensor([token_loss.detach(), hallucination_loss.detach()])
            )
        
        return (custom_loss, outputs) if return_outputs else custom_loss

    def evaluate(
        self,
        eval_dataset: Optional[torch.utils.data.Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        # Reset our loss collector at the beginning of evaluation
        self._eval_losses = []

        # Run the standard evaluation loop, which will populate self._eval_losses
        eval_output = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        # After the loop, self._eval_losses contains component losses from this process.
        # In a distributed setting, we need to gather these from all processes.
        if self.args.world_size > 1:
            # This gathers lists of tensors from all GPUs. It's a list of lists.
            all_gpu_losses = [None] * self.args.world_size
            torch.distributed.all_gather_object(all_gpu_losses, self._eval_losses)
            # Flatten the list of lists into a single list on the main process
            if self.state.is_local_process_zero:
                all_losses_list = [item for sublist in all_gpu_losses for item in sublist]
            else:
                all_losses_list = []
        else:
            all_losses_list = self._eval_losses

        # Now, on the main process, calculate the mean of the collected losses
        if self.state.is_local_process_zero and all_losses_list:
            all_losses_tensor = torch.stack(all_losses_list)
            mean_losses = all_losses_tensor.mean(dim=0)
            
            # Add our custom metrics to the output dictionary
            eval_output.metrics[f"{metric_key_prefix}_token_loss"] = mean_losses[0].item()
            eval_output.metrics[f"{metric_key_prefix}_hallucination_loss"] = mean_losses[1].item()
        
        # Clean up the collector for the next potential evaluation
        self._eval_losses = []

        return eval_output.metrics