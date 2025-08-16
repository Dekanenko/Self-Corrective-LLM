import torch
from torch import nn
import torch.nn.functional as F
from transformers import LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from dataclasses import dataclass
from typing import Tuple, Union

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
        self.hallucination_detector = nn.Linear(config.hidden_size, 1)
    
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
        logits = self.lm_head(last_hidden)

        gate_output = self.hallucination_gate_proj(last_hidden)
        up_output = self.hallucination_up_proj(last_hidden)
        gated_hidden = F.silu(gate_output) * up_output
        detector_hidden = self.hallucination_down_proj(gated_hidden)

        hallucination_logits = self.hallucination_detector(detector_hidden)

        # 3. Modify the token logits.
        additional_logits = torch.zeros_like(logits)
        additional_logits[:, :, -self.num_new_tokens:] = hallucination_logits
        logits = logits + additional_logits

        # 4. Return the custom output object.
        return SelfCorrectiveLlamaOutput(
            loss=None, # Loss calculation is handled by the Trainer
            logits=logits,
            hallucination_logits=hallucination_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=None,
            attentions=transformer_outputs.attentions
        )

    def generate(self, *args, output_hallucination_logits: bool = False, **kwargs) -> Union[torch.LongTensor, Tuple[torch.LongTensor, torch.FloatTensor]]:
        if not output_hallucination_logits:
            # Default behavior: return only the generated token IDs
            return super().generate(*args, **kwargs)

        # Custom behavior: attach a temporary list to the model to collect outputs.
        self._hallucination_logits_outputs = []

        try:
            # Call the original generate method. It will populate our list via the
            # _update_model_kwargs_for_generation hook.
            sequences = super().generate(*args, **kwargs)

            # Concatenate the collected hallucination_logits values. This will have a length
            # equal to the number of generated tokens.
            hallucination_logits = torch.cat(self._hallucination_logits_outputs, dim=1)
                
            return (sequences, hallucination_logits)
        finally:
            # Crucial: clean up the temporary attribute afterwards to ensure the model
            # state is clean for the next call.
            del self._hallucination_logits_outputs