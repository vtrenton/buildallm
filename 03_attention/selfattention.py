#!/usr/bin/env python3
import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias) #nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias) #nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias) #nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

    def forward(self, x):
        queries = x @ self.W_query
        keys = x @ self.W_key
        values = x @ self.W_value
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        context_vec = attn_weights @ values
        return context_vec

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

torch.manual_seed(789)
d_in = inputs.shape[1]
d_out = 2

sa = SelfAttention(d_in, d_out)
#print(sa(inputs))


# masked attention method
queries = sa.W_query(inputs)
keys = sa.W_key(inputs)
attn_scores = queries @ keys.T
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
print(f"attention weights:\n", attn_weights)

# let's now mask the attention weights before getting the context vector

context_length = attn_scores.shape[0]
mask_simple = torch.tril(torch.ones(context_length, context_length))
print(f"Input Mask:\n", mask_simple)

# apply the mask to the attention weights
mask_simple = attn_weights*mask_simple
print(f"Mask applied to weights:\n", mask_simple)

# Normalize the masked weights
row_sums = mask_simple.sum(dim=-1, keepdim=True)
mask_simple_norm = mask_simple / row_sums # tensor denominator propegation
print(f"Masked weights normalized:\n", mask_simple_norm)
