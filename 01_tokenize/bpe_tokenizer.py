#!/usr/bin/env python3

import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

with open("../the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

enc_text = tokenizer.encode(raw_text)

enc_sample = enc_text[50:] # remove the first 50 tokens from the set for the memes

context_size = 4
x = enc_sample[:context_size]
y = enc_sample[1:context_size]
print(f"x: {x}")
print(f"y:      {y}")

# next word prediction task
for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(context, "--->", desired)

for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(tokenizer.decode(context), "--->", tokenizer.decode([desired]))
