from flash_attn import flash_attn_func
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from .rope import LlamaRotaryEmbedding, apply_rotary_pos_emb
from .config import LlamaConfig

class LlamaAttention(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_attention_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_attention_heads`: {self.num_heads})."
            )
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.rotary_emb = LlamaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        bsz, _, _ = hidden_states.size()
        
        if position_ids is None:
            position_ids = repeat(
                torch.arange(seq_len, device=input_ids.device),
                'l -> b l',
                b=bsz
            )
        
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = rearrange(query_states, "b s (h d) -> b h s d", h=self.num_heads)
        key_states = rearrange(key_states, "b s (h d) -> b h s d", h=self.num_key_value_heads)
        value_states = rearrange(value_states, "b s (h d) -> b h s d", h=self.num_key_value_heads)

        # @Z TODO:: rope expects (b h s d) 
        cos, sin = self.rotary_emb(position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        # FA Expects (b s h d) 
        query_states = rearrange(query_states, "b h s d -> b s h d")
        key_states = rearrange(key_states, "b h s d -> b s h d")
        value_states = rearrange(value_states, "b h s d -> b s h d")
        
        attn_output = flash_attn_func(
            query_states,
            key_states,
            value_states,
            dropout_p=0.0,
            causal=attention_mask is None
        )

        attn_output = rearrange(attn_output, "b s h d -> b s (h d)")
        return self.o_proj(attn_output)
