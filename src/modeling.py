import torch
from torch import nn
import torch.nn.functional as F
from transformers import LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from dataclasses import dataclass

@dataclass
class SelfCorrectiveLlamaOutput(CausalLMOutputWithPast):
    hallucination_logits: torch.FloatTensor = None

class SelfCorrectiveLlama(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        
        self.num_new_tokens = 3

        intermediate_size = config.intermediate_size
        self.hallucination_gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.hallucination_up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.hallucination_down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.hallucination_detector = nn.Linear(config.hidden_size, self.num_new_tokens + 1)
    
    def forward(
        self, 
        input_ids, 
        attention_mask=None, 
        labels=None, 
        hallucination_labels=None, 
        **kwargs
    ):
        # 1. Get the last hidden state from the base transformer model.
        transformer_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        last_hidden = transformer_outputs.last_hidden_state

        # 2. Calculate token logits and hallucination logits from the last hidden state.
        # Token logits
        logits = self.lm_head(last_hidden)

        # SwiGLU-based hallucination detector
        gate_output = self.hallucination_gate_proj(last_hidden)
        up_output = self.hallucination_up_proj(last_hidden)
        gated_hidden = F.silu(gate_output) * up_output
        detector_hidden = self.hallucination_down_proj(gated_hidden)

        # Hallucination logits
        all_hallucination_logits = self.hallucination_detector(detector_hidden)

        # 3. Modify the token logits conditionally.
        deletion_logits = all_hallucination_logits[..., 1:] # skip the first token (no hallucination)
        additional_logits = torch.zeros_like(logits)

        # Conditionally add the deletion logits if we are in training and labels are provided.
        if hallucination_labels is not None and labels is not None:
            # Condition 1: The hallucination label is 0 (no hallucination).
            mask_no_hallucination = (hallucination_labels == 0)

            # Condition 2: The next token is one of the deletion tokens.
            # Check if the token ID is within the range of the last `num_new_tokens` in the vocab.
            vocab_size = logits.shape[-1]
            mask_is_deletion_token = (labels >= (vocab_size - self.num_new_tokens)) & (labels < vocab_size)

            # Combine the masks. The addition happens if either condition is true.
            # We need to align the shapes for broadcasting.
            combined_mask = (mask_no_hallucination | mask_is_deletion_token).unsqueeze(-1)

            # Use the mask to conditionally apply the deletion logits.
            additional_logits[:, :, -self.num_new_tokens:] = torch.where(
                combined_mask,
                deletion_logits,
                torch.zeros_like(deletion_logits)
            )
        else:
            # Inference case: always add the deletion logits to the token logits.
            additional_logits[:, :, -self.num_new_tokens:] = deletion_logits

        logits = logits + additional_logits

        # 4. Return the custom output object.
        return SelfCorrectiveLlamaOutput(
            loss=None, # Loss calculation is handled by the Trainer
            logits=logits,
            hallucination_logits=all_hallucination_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=None,
            attentions=transformer_outputs.attentions
        )