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
        self.deletion_boost_multiplier = config.deletion_boost_multiplier if "deletion_boost_multiplier" in config else 1.0
        self.threshold = config.threshold if "threshold" in config else 0.6
        self.eps = config.eps if "eps" in config else 1e-5

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
        self.hallucination_detector = nn.Linear(config.hidden_size, 1)
    
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

        # 4. SwiGLU-based hallucination detector
        gate_output = self.hallucination_gate_proj(last_hidden)
        up_output = self.hallucination_up_proj(last_hidden)
        gated_hidden = F.silu(gate_output) * up_output
        detector_hidden = self.hallucination_down_proj(gated_hidden)

        # Add a residual connection and a LayerNorm to stabilize the detector head.
        normalized_hidden = self.hallucination_norm(detector_hidden + last_hidden)

        # Hallucination logits
        all_hallucination_logits = self.hallucination_detector(normalized_hidden)
        hallucination_probs = torch.sigmoid(all_hallucination_logits)

        # 5. Apply the gate to the deletion token logits.
        # This implements a "soft gate" for training and a "hard gate" for inference.
        if self.training:
            # --- Training: A Precisely Masked Soft Gate ---
            # To avoid conflicting gradients, we build a scaling factor tensor that
            # modulates the deletion logits based on the ground truth labels.
            
            # 1. Create masks based on the ground truth labels.
            mask_s_label = (labels == self.original_vocab_size)
            mask_a_label = (labels == self.original_vocab_size + 1)
            # A mask for when we are predicting a normal token (hallucination_label is 0).
            mask_normal_token = (hallucination_labels == 0)

            # 2. Build the scaling factor for each deletion logit.
            # - If the label is <DEL_S>, we scale the <DEL_S> logit by the gate's probability.
            # - If the label is <DEL_A>, we scale the <DEL_A> logit by the gate's probability.
            # - If the label is a normal token, we scale BOTH deletion logits by the gate's probability.
            # - Otherwise, the scaling factor is 1.0 (no scaling).
            
            scaling_factor_s = torch.where(mask_s_label | mask_normal_token, hallucination_probs.squeeze(-1), 1.0)
            scaling_factor_a = torch.where(mask_a_label | mask_normal_token, hallucination_probs.squeeze(-1), 1.0)
            
            # Combine and apply the gate.
            scaling_factors = torch.stack([scaling_factor_s, scaling_factor_a], dim=-1)
            gated_del_logits = new_logits * scaling_factors
        else:
            # --- Inference: Hard Gate ---
            # We use a threshold to make a firm on/off decision.
            gate_mask = hallucination_probs > self.threshold
            
            # Where the gate is open, use the original deletion logits with a boost.
            # Where it's closed, suppress them to negative infinity.

            boosted_logits = new_logits + hallucination_probs * self.deletion_boost_multiplier

            gated_del_logits = torch.where(
                gate_mask,
                boosted_logits,
                torch.full_like(new_logits, torch.finfo(new_logits.dtype).min)
            )

        # Concatenate the main logits with the gated deletion logits.
        logits = torch.cat([main_logits, gated_del_logits], dim=-1)

        # 6. Return the custom output object
        return SelfCorrectiveLlamaOutput(
            loss=None, # Loss calculation is handled by the Trainer
            logits=logits,
            hallucination_logits=all_hallucination_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=None,
            attentions=transformer_outputs.attentions
        )