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
        
        self.lookup_length = getattr(config, "lookup_length", 30)
        self.num_new_tokens = 2
        self.original_vocab_size = config.vocab_size

        # Create a new, small embedding layer for only the special tokens
        self.new_token_embeddings = nn.Embedding(self.num_new_tokens, config.hidden_size)

        # Initialize new embeddings with the mean of the original ones
        with torch.no_grad():
            original_embeddings = self.model.embed_tokens.weight
            mean_embeddings = original_embeddings.mean(dim=0)
            self.new_token_embeddings.weight.data.copy_(
                mean_embeddings.unsqueeze(0).expand(self.num_new_tokens, -1)
            )

        intermediate_size = config.intermediate_size
        self.hallucination_gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.hallucination_up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.hallucination_down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.hallucination_detector = nn.Linear(config.hidden_size, self.num_new_tokens + 1)
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        # Get the full sequence of input IDs from the past, if available
        past_input_ids = kwargs.get("past_input_ids", None)

        # If past_input_ids exists, concatenate it with the new input_ids
        if past_input_ids is not None:
            input_ids = torch.cat([past_input_ids, input_ids], dim=-1)
        
        # Call the original prepare_inputs_for_generation method
        model_inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, **kwargs)

        # Update model_kwargs to include the full input_ids sequence for the next step
        model_inputs["past_input_ids"] = input_ids
        
        return model_inputs
    
    def forward(
        self, 
        input_ids, 
        attention_mask=None, 
        labels=None, 
        hallucination_labels=None,
        past_input_ids=None,
        **kwargs
    ):
        # 1. Manually construct the input embeddings
        clamped_input_ids = torch.clamp(input_ids, max=self.original_vocab_size - 1)
        inputs_embeds = self.model.embed_tokens(clamped_input_ids)

        # Overwrite the embeddings for our new special tokens
        special_token_mask = input_ids >= self.original_vocab_size
        if special_token_mask.any():
            special_ids = input_ids[special_token_mask] - self.original_vocab_size
            special_embeds = self.new_token_embeddings(special_ids)
            inputs_embeds[special_token_mask] = special_embeds

        # 2. Pass the constructed embeddings through the base transformer model
        kwargs["inputs_embeds"] = inputs_embeds
        transformer_outputs = self.model(
            attention_mask=attention_mask,
            **kwargs
        )
        last_hidden = transformer_outputs.last_hidden_state

        # 3. Calculate token logits by combining outputs from both heads
        # Main logits from the original, frozen lm_head
        main_logits = self.lm_head(last_hidden)

        # New token logits from small, trainable embedding layer
        new_logits = F.linear(last_hidden, self.new_token_embeddings.weight)

        # Concatenate to get logits over the full, expanded vocabulary
        logits = torch.cat([main_logits, new_logits], dim=-1)

        # 4. During inference, prevent deletion tokens if one was used recently.
        if not self.training and self.lookup_length > 0:
            # Use past_input_ids if available (during generation), otherwise use input_ids
            ids_to_check = past_input_ids if past_input_ids is not None else input_ids
            
            if ids_to_check.shape[1] > 0:
                # Check the last `lookup_length` tokens for deletion tokens.
                lookback_window = ids_to_check[:, -self.lookup_length :]

                del_s_token_id = self.original_vocab_size
                del_a_token_id = self.original_vocab_size + 1

                # Check if deletion tokens are present in the lookback window for each sequence
                had_del_s = (lookback_window == del_s_token_id).any(dim=1)
                had_del_a = (lookback_window == del_a_token_id).any(dim=1)
                mask = had_del_s | had_del_a

                if mask.any():
                    # For sequences with a recent deletion token, suppress the logits of deletion tokens
                    suppress_value = torch.finfo(logits.dtype).min
                    logits[mask, -1, del_s_token_id] = suppress_value
                    logits[mask, -1, del_a_token_id] = suppress_value

        # 5. Return the custom output object
        return SelfCorrectiveLlamaOutput(
            loss=None, # Loss calculation is handled by the Trainer
            logits=logits,
            hallucination_logits=all_hallucination_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=None,
            attentions=transformer_outputs.attentions
        )