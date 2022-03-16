
"""
Getting the Bert vectors for the audio samples
"""

import numpy as np
import pandas as pd

import torch
from transformers import BertModel, BertTokenizerFast

from sklearn.feature_extraction.text import TfidfTransformer

tokenizer = BertTokenizerFast.from_pretrained('onlplab/alephbert-base')
alephbert = BertModel.from_pretrained('onlplab/alephbert-base')
tfidf_transformer = TfidfTransformer()
# if not finetuning - disable dropout
alephbert.eval()

df = pd.read_csv('../data/merav3-names.csv', encoding="ISO-8859-8")
df = df.dropna(how='all')
max_len = 0

vectors=[]
for i in df["Type and content"]:
    print(i) # The sentence
    tokenized = tokenizer.encode(i, add_special_tokens=True)
    if len(tokenized) > max_len:
        max_len = len(i)

    padded = np.array([tokenized + [0] * (max_len - len(tokenized))])

    attention_mask = np.where(padded != 0, 1, 0)
    input_ids = torch.tensor(padded)

    attention_mask = torch.tensor(attention_mask)

    with torch.no_grad():
        last_hidden_states = alephbert(input_ids)

    features = last_hidden_states[0][:, 0, :].numpy()
    print("befor nirmul")
    print(features)

    X_tfidf = tfidf_transformer.fit_transform(features)
    print("after nirmul")
    vectors.append(X_tfidf.toarray())
    print(len(X_tfidf.toarray()[0]))

df["bert"] = vectors
df.to_csv("../data/merav3_names_bert.csv", encoding="ISO-8859-8")