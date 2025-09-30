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

        # # 4. During inference, prevent consecutive deletion tokens.
        # if not self.training:
        #     # Get the last token for each sequence in the batch.
        #     prev_token = input_ids[:, -1]
            
        #     # Create a boolean mask for sequences where the last token was a deletion token.
        #     prev_was_del_s = (prev_token == self.original_vocab_size)
        #     prev_was_del_a = (prev_token == self.original_vocab_size + 1)
        #     mask = prev_was_del_s | prev_was_del_a

        #     # If any sequence in the batch ended with a deletion token...
        #     if mask.any():
        #         # ...suppress the deletion token logits for the current step in those sequences.
        #         # We only modify the last token in the sequence, which is the current prediction.
        #         suppress_value = torch.finfo(logits.dtype).min
        #         logits[mask, -1, self.original_vocab_size] = suppress_value
        #         logits[mask, -1, self.original_vocab_size + 1] = suppress_value

        # 5. Return the custom output object
        return SelfCorrectiveLlamaOutput(
            loss=None, # Loss calculation is handled by the Trainer
            logits=logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=None,
            attentions=transformer_outputs.attentions
        )