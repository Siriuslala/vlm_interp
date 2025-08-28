import torch
import torch.nn as nn
import torch.nn.functional as F

class VisionTokenDecoder(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)

        print(f"VisionTokenDecoder initialized: hidden_size={hidden_size}, vocab_size={vocab_size}")

    def forward(self, vision_embedding: torch.Tensor) -> torch.Tensor:
        # (batch_size, num_vision_tokens, hidden_size) -> (batch_size, num_vision_tokens, vocab_size)
        logits = self.decoder(vision_embedding)
        return logits
