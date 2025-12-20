#!/usr/bin/env python3
import re

class SimpleTokenizer:
    def __init__(self, vocab):
        self.str_to_int = vocab # stores the vocab in a class method for access in encode and decode methods.
        self.int_to_str = {i:s for s,i in vocab.items()} # inverses vocab to map IDs back to text

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text) 
        preprocessed = [item for item in preprocessed if item.strip()]
        preprocessed = [item if item in self.str_to_int
                        else "<|unk|>" for item in preprocessed] # if the item doesn't match anything in the vocab use <|unk|>
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.:;?_!"()\'])', r'\1', text)
        return text


# read in values
with open("../the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# split on chars
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
# remove whitespace
preprocessed = [item for item in preprocessed if item.strip()]

all_tokens = sorted(list(set(preprocessed))) # sort and unique all words
all_tokens.extend(["<|endoftext|>", "<|unk|>"]) # then append indicator words.

vocab = {token:integer for integer, token in enumerate(all_tokens)}

text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))

print(text)
tokenizer = SimpleTokenizer(vocab)
print(tokenizer.encode(text))
print(tokenizer.decode(tokenizer.encode(text)))

#tokenizer = SimpleTokenizer(vocab)
#text = """"It's the last he painted, you know,"
#Mrs. Gisburn said with pardonable pride."""
#ids = tokenizer.encode(text)
#
## print the IDs
#print(ids)
#
## Get the Text back out
#print(tokenizer.decode(ids))
