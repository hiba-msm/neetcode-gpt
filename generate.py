import torch
import torch.nn as nn
from torchtyping import TensorType


class Solution:
    def generate(
        self,
        model,
        new_chars: int,
        context: TensorType[int],
        context_length: int,
        int_to_char: dict
    ) -> str:

        generator = torch.manual_seed(0)
        initial_state = generator.get_state()

        generated = []

        model.eval()

        with torch.no_grad():
            for i in range(new_chars):

                # Crop context to max allowed length
                context_crop = context[:, -context_length:]

                # Forward pass
                logits = model(context_crop)

                # Last time-step logits only
                last_logits = logits[:, -1, :]

                # Convert logits to probabilities
                probs = torch.softmax(last_logits, dim=-1)

                # Sample next token
                next_token = torch.multinomial(probs, 1, generator=generator)

                # Do not alter this fixed line
                generator.set_state(initial_state)

                # Append token to context
                context = torch.cat((context, next_token), dim=1)

                # Convert sampled token ID to character
                token_id = next_token.item()
                generated.append(int_to_char[token_id])

        return "".join(generated)