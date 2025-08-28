from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VLForConditionalGeneration,
    Qwen2VisionTransformerPretrainedModel,
)

from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration,
)

import torch
import torch.nn as nn
from typing import Optional


class VisionRotaryEmbedding_scaling_rope(nn.Module):
    def __init__(
        self,
        dim: int,
        theta: float = 10000.0,
    ) -> None:
        super().__init__()
        
        # 1. Calculate standard inverse frequencies (theta_i)
        
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # 2. Calculate the compensatory scaling factor g(i)
        # Note: In RoPE, 'i' is the group index from 0 to dim/2 - 1.
        # The term `torch.arange(0, dim, 2)` corresponds to `2i`.

        i_normed = torch.arange(0, dim, 2, dtype=torch.float) / dim
        self.register_buffer("i_normed", i_normed, persistent=False)
        
        # self.sig_alpha = nn.Parameter(torch.tensor(99.0, dtype=torch.float), requires_grad=True)
        # self.sig_mid_point = nn.Parameter(torch.tensor(0.6, dtype=torch.float), requires_grad=True)
        # self.sig_k = nn.Parameter(torch.tensor(40.0, dtype=torch.float), requires_grad=True)
        
        self.scaling_type= "poly"  # poly, sigmoid
        self.poly_alpha = 99.0
        self.poly_p = 6.0
        self.sig_alpha = 99.0
        self.sig_mid_point = 0.6
        self.sig_k = 40.0

        # self.register_buffer("compensation", compensation, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        
        # scaling
        if self.scaling_type == "poly":
            compensation = 1.0 + self.poly_alpha * (self.i_normed ** self.poly_p)
        else:
            sigmoid_input = self.sig_k * (self.i_normed - self.sig_mid_point)
            compensation = 1.0 + self.sig_alpha * torch.sigmoid(sigmoid_input)
        
        compensated_freqs = freqs * compensation
        return compensated_freqs


class Qwen2VisionTransformerPretrainedModel_rope_scaling(Qwen2VisionTransformerPretrainedModel):
    """
    This class is a modified version of Qwen2VisionTransformerPretrainedModel that supports rope scaling.
    """
    def __init__(self, config):
        super().__init__(config)
        head_dim = config.embed_dim // config.num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding_scaling_rope(head_dim // 2)


class Qwen2VLForConditionalGeneration_rope_scaling(Qwen2VLForConditionalGeneration):
    """
    This class is a modified version of Qwen2VLForConditionalGeneration that supports rope scaling.
    """
    def __init__(self, config):
        super().__init__(config)
        self.visual = Qwen2VisionTransformerPretrainedModel_rope_scaling._from_config(config.vision_config)


class Qwen2_5_VLForConditionalGeneration_rope_scaling(Qwen2_5_VLForConditionalGeneration):
    """
    This class is a modified version of Qwen2_5_VLForConditionalGeneration that supports rope scaling.
    """
    def __init__(self, config):
        super().__init__(config)
        # self.visual = Qwen2VisionTransformerPretrainedModel_rope_scaling._from_config(config.vision_config)
