#!/usr/bin/env python3
import torch

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

query = inputs[1] # start with the the second index "Journey"

# initialize an empty tensor with the batch_size of inputs (6) 
attn_scores_2 = torch.empty(inputs.shape[0])

# i = Vector or i_x
# i_x individual scalar values
# This will populate the new initialized tensor
# starting by taking input and finding the product of it against each row

# Dot Product:
# [a1, a2, a3]
# [b1, b2, b3]
# a1*b1 + a2*b2 + a3*b3

for i, i_x in enumerate(inputs):
    attn_scores_2[i] = torch.dot(i_x, query)

# Note: This is not a python list (Vector) / Scalar
# These are tensor Objects part of pytorch
# the .sum() return a scalar tensor which broadcasts across the Vector during division
# this is a primative way of normalizing the numbers
attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
# attn_weights_2_tmp.sum() == 1.0

# Normalizing our attention scores via a softmax function
# is much more flexible than a primative division

# here is a niave softmax function to start
# .exp() will 'exponentiate' a number
# which effectively takes it to the power of e
# e being eulers number -> an irrational constant for scaling
def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

attn_weights_2_naive = softmax_naive(attn_scores_2)


# this is the more *correct* way of normalizing
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)

print("Attention weights normalized using a Torch Softmax function:\n", 
      attn_weights_2)

# multiply the attention weights with the input embedding Vector
context_vect_2 = torch.zeros(query.shape) 
for i, i_x in enumerate(inputs):
    context_vect_2 += attn_weights_2[i]*i_x

# The wieghted sum of all input vectors
print("Weighted sum:\n", context_vect_2)
