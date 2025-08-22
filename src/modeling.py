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
        self.original_vocab_size = config.vocab_size

        # Create a new, small embedding layer for only the special tokens.
        self.new_token_embeddings = nn.Embedding(self.num_new_tokens, config.hidden_size)

        # --- Initialize new embeddings with the mean of the original ones ---
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
        # 1. Manually construct the input embeddings.
        # This allows us to use a separate embedding layer for our new tokens, saving memory.
        special_token_mask = input_ids >= self.original_vocab_size
        print(f"special_token_mask:\n{special_token_mask}")

        if not special_token_mask.any():
            inputs_embeds = self.model.embed_tokens(input_ids)
        else:
            normal_token_mask = ~special_token_mask
            normal_ids = input_ids.clone()
            normal_ids[special_token_mask] = 0
            print(f"normal_ids:\n{normal_ids}")
            normal_embeds = self.model.embed_tokens(normal_ids)
            
            inputs_embeds = torch.empty_like(normal_embeds)
            inputs_embeds[normal_token_mask] = normal_embeds[normal_token_mask]

            special_ids = input_ids[special_token_mask] - self.original_vocab_size
            print(f"special_ids:\n{special_ids}")
            special_embeds = self.new_token_embeddings(special_ids)
            print(f"special_embeds:\n{special_embeds.shape}")
            inputs_embeds[special_token_mask] = special_embeds

        # 2. Pass the constructed embeddings through the base transformer model.
        # Note: We pass `inputs_embeds` directly, so the model skips its own embedding layer.
        kwargs["inputs_embeds"] = inputs_embeds
        transformer_outputs = self.model(
            attention_mask=attention_mask,
            **kwargs
        )
        last_hidden = transformer_outputs.last_hidden_state

        # 3. Calculate token logits by combining outputs from both heads.
        # Main logits from the original, frozen lm_head.
        main_logits = self.lm_head(last_hidden)
        print(f"main_logits:\n{main_logits}")
        # New token logits from our small, trainable embedding layer.
        new_logits = F.linear(last_hidden, self.new_token_embeddings.weight)
        print(f"new_logits:\n{new_logits}")

        # Concatenate to get logits over the full, expanded vocabulary.
        logits = torch.cat([main_logits, new_logits], dim=-1)
        print(f"logits:\n{logits}")

        # 4. SwiGLU-based hallucination detector (logic is unchanged).
        gate_output = self.hallucination_gate_proj(last_hidden)
        up_output = self.hallucination_up_proj(last_hidden)
        gated_hidden = F.silu(gate_output) * up_output
        detector_hidden = self.hallucination_down_proj(gated_hidden)

        # Hallucination logits
        all_hallucination_logits = self.hallucination_detector(detector_hidden)

        # 5. Modify the token logits conditionally.
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

        # 6. Return the custom output object.
        return SelfCorrectiveLlamaOutput(
            loss=None, # Loss calculation is handled by the Trainer
            logits=logits,
            hallucination_logits=all_hallucination_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=None,
            attentions=transformer_outputs.attentions
        )