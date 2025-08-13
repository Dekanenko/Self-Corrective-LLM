import torch
from torch import nn
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
        self.hallucination_detector = nn.Linear(config.hidden_size, 1)
    
    def forward(
        self, 
        input_ids, 
        attention_mask=None, 
        labels=None, 
        hallucination_labels=None, 
        **kwargs
    ):
        kwargs["output_hidden_states"] = True
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # output_hidden_states=True,
            **kwargs
        )

        # Get last hidden state and calculate hallucination probability
        last_hidden = outputs.hidden_states[-1] # [batch, seq_len, hidden_size]
        hallucination_logits = self.hallucination_detector(last_hidden) # [batch, seq_len, 1]

        # Apply hallucination probability to the logits of the last 3 (special) tokens
        logits = outputs.logits
        additional_logits = torch.zeros_like(logits)
        additional_logits[:, :, -self.num_new_tokens:] = hallucination_logits
        logits = logits + additional_logits

        return SelfCorrectiveLlamaOutput(
            loss=None, # Loss calculation should be handled by a custom Trainer
            logits=logits,
            hallucination_logits=hallucination_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )

    def _update_model_kwargs_for_generation(
        self,
        outputs: SelfCorrectiveLlamaOutput,
        model_kwargs: dict,
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
    ) -> dict:
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder, standardize_cache_format
        )

        # If we are collecting p_hall values, check for the attribute on the model instance.
        if hasattr(self, "_hallucination_logits_outputs"):
            # We take the p_hall from the *last* token in the sequence, which corresponds
            # to the prediction for the *next* token being generated. This is critical.
            hallucination_logits_at_step = outputs.hallucination_logits[:, -1, :]
            self._hallucination_logits_outputs.append(hallucination_logits_at_step)
            
        return model_kwargs

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

            # Concatenate the collected p_hall values. This will have a length
            # equal to the number of generated tokens.
            hallucination_logits = torch.cat(self._hallucination_logits_outputs, dim=1)
                
            return (sequences, hallucination_logits)
        finally:
            # Crucial: clean up the temporary attribute afterwards to ensure the model
            # state is clean for the next call.
            del self._hallucination_logits_outputs