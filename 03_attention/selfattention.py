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

class CasualAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
            super().__init__()
            self.d_out = d_out
            self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.dropout = nn.Dropout(dropout)
            self.register_buffer(
                    'mask',
                    torch.triu(torch.ones(context_length, context_length), diagonal=1)
            )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(
                self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights) # apply dropout to weights

        context_vec = attn_weights @ values
        return context_vec

class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList([CasualAttention(d_in, d_out, context_length, dropout, qkv_bias)
                                   for i in range(num_heads)])

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)



inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

batch = torch.stack((inputs, inputs), dim=0)
print(batch.shape)

#torch.manual_seed(789)
#d_in = inputs.shape[1]
#d_out = 2
#
#sa = SelfAttention(d_in, d_out)
##print(sa(inputs))
#
#
## masked attention method
#queries = sa.W_query(inputs)
#keys = sa.W_key(inputs)
#attn_scores = queries @ keys.T

###### OLD METHOD #####
#attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
#print(f"attention weights:\n", attn_weights)
#
## let's now mask the attention weights before getting the context vector
#context_length = attn_scores.shape[0]
#mask_simple = torch.tril(torch.ones(context_length, context_length))
#print(f"Input Mask:\n", mask_simple)
#
## apply the mask to the attention weights
#mask_simple = attn_weights*mask_simple
#print(f"Mask applied to weights:\n", mask_simple)
#
## Normalize the masked weights
#row_sums = mask_simple.sum(dim=-1, keepdim=True)
#mask_simple_norm = mask_simple / row_sums # tensor denominator propegation
#print(f"Masked weights normalized:\n", mask_simple_norm)

##### CLEANER METHOD #####
# we can Normalize the mask directly via the softmax function
#context_length = attn_scores.shape[0]
#mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
#masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
#
## Now apply the softmax
#attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=-1)
#print(attn_weights)
#
#
## apply dropout method
#torch.manual_seed(123)
#dropout = nn.Dropout(0.5)
#
#print(dropout(attn_weights))
torch.manual_seed(123)
context_length = batch.shape[1] # number of tokens
d_in, d_out = 3, 2
mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs = mha(batch)

print(context_vecs)
print("Shape of context vector:\n", context_vecs.shape)

