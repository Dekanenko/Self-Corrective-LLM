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
        self.alpha_boost = config.alpha_boost if "alpha_boost" in config else 5.0
        self.tau = config.tau if "tau" in config else 0.7
        self.max_boost = config.max_boost if "max_boost" in config else 8.0

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
        self.hallucination_norm = nn.LayerNorm(config.hidden_size)
        self.hallucination_detector = nn.Linear(config.hidden_size, self.num_new_tokens + 1)
    
    def forward(
        self, 
        input_ids, 
        attention_mask=None, 
        labels=None, 
        hallucination_labels=None, 
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

        # 4. SwiGLU-based hallucination detector
        gate_output = self.hallucination_gate_proj(last_hidden)
        up_output = self.hallucination_up_proj(last_hidden)
        gated_hidden = F.silu(gate_output) * up_output
        detector_hidden = self.hallucination_down_proj(gated_hidden)

        # Add a residual connection and a LayerNorm to stabilize the detector head.
        normalized_hidden = self.hallucination_norm(detector_hidden + last_hidden)

        # Hallucination logits
        all_hallucination_logits = self.hallucination_detector(normalized_hidden)

        # 5. Modify the token logits conditionally.
        no_hall, del_s, del_a = all_hallucination_logits.split(1, dim=-1)
        margin_s, margin_a = del_s - no_hall, del_a - no_hall
    
        boost_del_s = (self.alpha_boost * F.softplus(margin_s - self.tau)).clamp(max=self.max_boost)
        boost_del_a = (self.alpha_boost * F.softplus(margin_a - self.tau)).clamp(max=self.max_boost)

        # Conditionally add the deletion logits.
        if labels is not None:
            # Training case: Apply boosts precisely based on the ground truth label.
            # This prevents signal conflict by ensuring we only boost the correct token.
            
            # Create separate masks for each deletion token.
            mask_s = (labels == self.original_vocab_size).unsqueeze(-1)
            mask_a = (labels == self.original_vocab_size + 1).unsqueeze(-1)
            
            # Apply boosts only where the label matches the specific token.
            to_add_s = torch.where(mask_s, boost_del_s, torch.zeros_like(boost_del_s))
            to_add_a = torch.where(mask_a, boost_del_a, torch.zeros_like(boost_del_a))
            
            # Combine into the final tensor to add.
            to_add = torch.cat([to_add_s, to_add_a], dim=-1)
        else:
            # Inference case: Apply boosts precisely based on the margin threshold.
            # This prevents "logit bleed" by only boosting the token that meets the criterion.
            
            # Create separate masks for each token's margin.
            mask_s_active = margin_s > self.tau
            mask_a_active = margin_a > self.tau
            
            # Calculate the boost for each token individually.
            to_add_s = torch.where(mask_s_active, boost_del_s, torch.zeros_like(boost_del_s))
            to_add_a = torch.where(mask_a_active, boost_del_a, torch.zeros_like(boost_del_a))

            # Combine into the final tensor to add.
            to_add = torch.cat([to_add_s, to_add_a], dim=-1)
        
        logits[:, :, -self.num_new_tokens:].add_(to_add)

        # 6. Return the custom output object
        return SelfCorrectiveLlamaOutput(
            loss=None, # Loss calculation is handled by the Trainer
            logits=logits,
            hallucination_logits=all_hallucination_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=None,
            attentions=transformer_outputs.attentions
        )