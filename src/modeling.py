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
    
    def forward(
        self, 
        input_ids, 
        attention_mask=None, 
        labels=None, 
        hallucination_labels=None, 
        **kwargs
    ):
        # 1. Manually construct the input embeddings by combining the original and new token embeddings
        
        # Create masks for original and new tokens
        original_token_mask = input_ids < self.original_vocab_size
        special_token_mask = ~original_token_mask

        # Get embeddings from the original, frozen embedding layer
        # We use a clamped version of input_ids to avoid out-of-bounds errors.
        # The values for special tokens will be ignored anyway by the masking.
        clamped_input_ids = torch.clamp(input_ids, max=self.original_vocab_size - 1)
        original_embeds = self.model.embed_tokens(clamped_input_ids)
        
        # Zero out the embeddings where special tokens are supposed to be.
        original_embeds = original_embeds * original_token_mask.unsqueeze(-1).to(original_embeds.dtype)

        # Get embeddings from the new, trainable embedding layer
        if special_token_mask.any():
            # Adjust IDs to be valid indices for the new embedding layer
            special_ids = input_ids[special_token_mask] - self.original_vocab_size
            new_embeds = self.new_token_embeddings(special_ids)
            
            # Create a full-size tensor for the new embeddings to allow for addition
            full_new_embeds = torch.zeros_like(original_embeds)
            full_new_embeds[special_token_mask] = new_embeds
            
            # Combine the embeddings by adding them. Since one is zero where the other has values,
            # this works as a clean, non-in-place merge.
            inputs_embeds = original_embeds + full_new_embeds
        else:
            inputs_embeds = original_embeds

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

        # 4. SwiGLU-based hallucination detector
        gate_output = self.hallucination_gate_proj(last_hidden)
        up_output = self.hallucination_up_proj(last_hidden)
        gated_hidden = F.silu(gate_output) * up_output
        detector_hidden = self.hallucination_down_proj(gated_hidden)

        # Hallucination logits, which also serve as the logits for the new tokens
        all_hallucination_logits = self.hallucination_detector(detector_hidden)

        # The logits for the new deletion tokens are the outputs from the detector
        # (excluding the first logit, which corresponds to the "no-op" class)
        new_logits = all_hallucination_logits[..., 1:]

        # 5. Concatenate to get logits over the full, expanded vocabulary
        logits = torch.cat([main_logits, new_logits], dim=-1)

        # 6. Return the custom output object
        return SelfCorrectiveLlamaOutput(
            loss=None, # Loss calculation is handled by the Trainer
            logits=logits,
            hallucination_logits=all_hallucination_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=None,
            attentions=transformer_outputs.attentions
        )