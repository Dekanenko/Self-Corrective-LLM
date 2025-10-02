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

        super().log(logs, *args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        token_labels = inputs.get("labels")
        hallucination_labels = inputs.get("hallucination_labels")
        
        outputs = model(**inputs)
        token_logits = outputs.get("logits")
        hallucination_logits = outputs.get("hallucination_logits")

        # --- Calculate Token Prediction Loss (Cross-Entropy) ---
        loss_fct_token = nn.CrossEntropyLoss(ignore_index=-100)
        
        shift_logits = token_logits[..., :-1, :].contiguous()
        shift_labels = token_labels[..., 1:].contiguous()
        
        vocab_size = token_logits.shape[-1]
        shift_logits = shift_logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1).to(shift_logits.device)
        
        token_loss = loss_fct_token(shift_logits, shift_labels)

        # --- 2. Calculate Hallucination Detection Loss (Binary Cross-Entropy) ---
        # We now use BCEWithLogitsLoss for the binary (0 or 1) hallucination task.
        pos_weight = None
        if self.correction_weight_tensor is not None:
            # For binary classification, correction_weights should be a 2-element list:
            # [weight_for_class_0, weight_for_class_1]
            assert len(self.correction_weight_tensor) == 2, "correction_weights must have 2 elements for binary BCE loss."
            # pos_weight is the ratio of negative to positive weights.
            pos_weight = self.correction_weight_tensor[1] / (self.correction_weight_tensor[0] + 1e-6)
            pos_weight = torch.tensor([pos_weight]).to(hallucination_logits.device)

        loss_fct_hallucination = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        shift_hallucination_logits = hallucination_logits[..., :-1, :].contiguous()
        shift_hallucination_labels = hallucination_labels[..., 1:].contiguous()
        
        # Reshape for BCEWithLogitsLoss
        shift_hallucination_logits = shift_hallucination_logits.view(-1)
        shift_hallucination_labels = shift_hallucination_labels.view(-1).float()
        
        # Create a mask to ignore padding tokens (-100)
        active_loss_mask = shift_hallucination_labels != -100
        
        # Apply the mask to get only the active logits and labels
        active_logits = shift_hallucination_logits[active_loss_mask]
        active_labels = shift_hallucination_labels[active_loss_mask]
        
        # Calculate loss only on active elements
        hallucination_loss = loss_fct_hallucination(active_logits, active_labels)
        
        # --- 3. Combine the losses with your alpha weighting ---
        custom_loss = self.alpha * token_loss + (1 - self.alpha) * hallucination_loss

        # --- 4. Store metrics for logging ---
        if self.state.is_local_process_zero:
            if model.training:
                self._last_component_losses = {
                    "token_loss": token_loss.item(),
                    "hallucination_loss": hallucination_loss.item(),
                }
            else:
                # For evaluation, we need to get binary predictions (0 or 1)
                active_preds_probs = torch.sigmoid(active_logits)
                active_preds = (active_preds_probs > 0.5).long()

                self._eval_accumulator["token_losses"].append(token_loss.item())
                self._eval_accumulator["hallucination_losses"].append(hallucination_loss.item())
                self._eval_accumulator["preds"].extend(active_preds.cpu().numpy())
                self._eval_accumulator["labels"].extend(active_labels.cpu().numpy())
        
        return (custom_loss, outputs) if return_outputs else custom_loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        # The dataloader needs to be created from the dataset.
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        
        self._eval_accumulator = {
            "token_losses": [], "hallucination_losses": [], "preds": [], "labels": [],
        }

        # Pass the dataloader to the evaluation loop
        output = self.evaluation_loop(
            eval_dataloader,
            description="Evaluation",
            prediction_loss_only=True,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        if self.state.is_local_process_zero:
            token_losses = self._eval_accumulator["token_losses"]
            hall_losses = self._eval_accumulator["hallucination_losses"]
            preds = self._eval_accumulator["preds"]
            labels = self._eval_accumulator["labels"]

            if token_losses:
                output.metrics[f"{metric_key_prefix}_token_loss"] = sum(token_losses) / len(token_losses)
            if hall_losses:
                output.metrics[f"{metric_key_prefix}_hallucination_loss"] = sum(hall_losses) / len(hall_losses)
            if labels:
                f1 = f1_score(labels, preds, average='macro', zero_division=0)
                output.metrics[f"{metric_key_prefix}_f1_score"] = f1
        
        self.log(output.metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)
        return output.metrics
