import torch
from torch import nn
import torch.nn.functional as F
from transformers import LlamaForCausalLM, PreTrainedTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
from dataclasses import dataclass


@dataclass
class SelfCorrectiveLlamaOutput(CausalLMOutputWithPast):
    hallucination_logits: torch.FloatTensor = None


class SelfCorrectiveLlama(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)

        self.correction_cooldown = getattr(config, "correction_cooldown", 30)
        self.num_new_tokens = 2
        self.deletion_threshold = (
            config.deletion_threshold if "deletion_threshold" in config else 0.7
        )

        intermediate_size = config.intermediate_size
        self.hallucination_gate_proj = nn.Linear(
            config.hidden_size, intermediate_size, bias=False
        )
        self.hallucination_up_proj = nn.Linear(
            config.hidden_size, intermediate_size, bias=False
        )
        self.hallucination_down_proj = nn.Linear(
            intermediate_size, config.hidden_size, bias=False
        )
        self.hallucination_detector = nn.Linear(
            config.hidden_size, self.num_new_tokens + 1
        )
        self.hallucination_norm = nn.LayerNorm(config.hidden_size)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        # Get the full sequence of input IDs from the past, if available
        past_input_ids = kwargs.get("past_input_ids", None)

        # If past_input_ids exists, concatenate it with the new input_ids
        if past_input_ids is not None:
            input_ids = torch.cat([past_input_ids, input_ids], dim=-1)

        # Call the original prepare_inputs_for_generation method
        model_inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, **kwargs
        )

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
        **kwargs,
    ):
        # Pass inputs through the base LLaMA model.
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs
        )
        last_hidden = outputs.last_hidden_state

        # Calculate main token logits from the original lm_head.
        logits = self.lm_head(last_hidden)

        # Detach the hidden state to prevent gradients from the hallucination loss
        # from flowing back into the base model's LoRA adapters.
        detector_input = last_hidden.detach()

        # Pass the detached hidden state through the SwiGLU-based detector.
        gate_output = self.hallucination_gate_proj(detector_input)
        up_output = self.hallucination_up_proj(detector_input)
        gated_hidden = F.silu(gate_output) * up_output
        detector_hidden = self.hallucination_down_proj(gated_hidden)

        # Apply a residual connection and LayerNorm using the detached input.
        normalized_hidden = self.hallucination_norm(detector_hidden + detector_input)

        # Get the final hallucination logits from the detector head.
        hallucination_logits = self.hallucination_detector(normalized_hidden)

        # Return a custom output object with both sets of logits.
        return SelfCorrectiveLlamaOutput(
            loss=None,  # Loss calculation is handled by the Trainer
            logits=logits,
            hallucination_logits=hallucination_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=None,
            attentions=outputs.attentions,
        )

    def _initialize_instruction_tokens(self, tokenizer: PreTrainedTokenizer):
        """A helper to tokenize and cache instruction phrases."""
        if not hasattr(self, "rewrite_sentence_ids"):
            self.rewrite_sentence_ids = tokenizer(
                "[rewrite sentence]",
                return_tensors="pt",
                add_special_tokens=False,
            ).input_ids.to(self.device)
        if not hasattr(self, "rewrite_response_ids"):
            self.rewrite_response_ids = tokenizer(
                "[rewrite response]",
                return_tensors="pt",
                add_special_tokens=False,
            ).input_ids.to(self.device)

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        tokenizer: PreTrainedTokenizer,
        max_new_tokens=512,
        temperature=0.3,
        **kwargs,
    ):
        """
        Custom generate method to orchestrate self-correction.

        NOTE: This implementation currently only supports a batch size of 1.
        """
        # Set the model to evaluation mode and cache instruction tokens.
        self.eval()
        self._initialize_instruction_tokens(tokenizer)

        # Initialize the sequence with the prompt and its attention mask.
        generated_ids = input_ids
        attention_mask = torch.ones_like(input_ids)

        # Initialize a counter to track tokens since the last correction.
        # Start it at the cooldown value to allow immediate correction if needed.
        tokens_since_correction = self.correction_cooldown

        # The first forward pass processes the prompt and gets the initial KV cache.
        outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=None,
            return_dict=True,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values

        # Start the generation loop with the logits for the token after the prompt.
        next_token_logits = outputs.logits[:, -1, :]
        hallucination_logits = outputs.hallucination_logits[:, -1, :]

        # Autoregressively generate tokens one by one.
        for _ in range(max_new_tokens):
            # Apply softmax to get hallucination probabilities.
            hallucination_probs = F.softmax(hallucination_logits, dim=-1)

            # Check if the cooldown period has passed.
            can_correct = tokens_since_correction >= self.correction_cooldown

            # Conditionally choose the next tokens based on the detector's output and the cooldown.
            if can_correct and hallucination_probs[0, 1] > self.deletion_threshold:
                current_tokens = self.rewrite_sentence_ids
                tokens_since_correction = 0  # Reset the counter
            elif can_correct and hallucination_probs[0, 2] > self.deletion_threshold:
                current_tokens = self.rewrite_response_ids
                tokens_since_correction = 0  # Reset the counter
            else:
                if temperature > 0.0:
                    scaled_logits = next_token_logits / temperature
                    probs = F.softmax(scaled_logits, dim=-1)
                    current_tokens = torch.multinomial(probs, num_samples=1)
                else:
                    current_tokens = torch.argmax(next_token_logits, dim=-1).unsqueeze(
                        -1
                    )

                # Increment the counter by the number of tokens just generated.
                tokens_since_correction += current_tokens.shape[1]

            generated_ids = torch.cat([generated_ids, current_tokens], dim=-1)

            # Stop generating if an EOS token is produced.
            if torch.any(current_tokens == tokenizer.eos_token_id):
                break

            # Prepare for the next iteration.
            cache_position = torch.arange(
                attention_mask.shape[1],
                attention_mask.shape[1] + current_tokens.shape[1],
                device=self.device,
            )
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        (1, current_tokens.shape[1]),
                        device=self.device,
                        dtype=torch.long,
                    ),
                ],
                dim=1,
            )

            # Perform a forward pass with only the new tokens, the KV cache, and the correct positions.
            outputs = self(
                input_ids=current_tokens,
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_values=past_key_values,
                return_dict=True,
                use_cache=True,
            )

            # Update the state for the next loop.
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            hallucination_logits = outputs.hallucination_logits[:, -1, :]

        # Return the final generated sequence.
        return generated_ids
