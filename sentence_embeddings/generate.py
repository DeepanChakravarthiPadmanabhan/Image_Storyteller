import os
import time
import pickle
import datetime
import warnings
from config import *
import tensorflow as tf
from model import layers
import nltk
import json
from numpy import array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import numpy as np
import random
num_lines = 2000
with open('new_tokenizer_30000.json') as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)
# with open('/scratch/smuthi2s/NLP_data/books/books_large_p1.txt', 'r') as in_file:
# # with open('/home/smuthi2s/perl5/NLP/Image_Storyteller/tf2-skip-thoughts/data.txt', 'r') as in_file:
#     total_sentences = in_file.read().splitlines()[:num_lines]
text = "we were laughing haha thats funny but then we were saying damn it all"
text_2 = "this is just great seth shouted ."
idx2word = {v:k for k,v in tokenizer.word_index.items()}
encoded_data = tokenizer.texts_to_sequences([text])[0]
encoded_data_2 = tokenizer.texts_to_sequences([text_2])[0]
max_length = 6
total_sentences = [encoded_data,encoded_data_2]

model = layers.skip_thoughts(thought_size=thought_size, word_size=embed_dim, vocab_size=vocab_size,
                             max_length=max_length)
# optimizer = tf.optimizers.Adam(learning_rate=0.001, clipnorm=5.0)
# checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
checkpoint_dir = "/scratch/smuthi2s/NLP_data/logs"
checkpoint_path = "/scratch/smuthi2s/NLP_data/logs/200_ckpt-50"
latest = tf.train.latest_checkpoint(checkpoint_dir)
print(latest)
lengths = np.array([len(encoded_data),len(encoded_data_2)])
# tf.train.load_checkpoint(latest)
# checkpoint.restore(checkpoint_path)
# model = tf.keras.models.load_model(checkpoint_path)
for i in range(100):
    # print(total_sentences[-2:])
    padded_data = pad_sequences(total_sentences[-2:],maxlen = max_length,padding='pre')
    masked_prev_pred, masked_next_pred = model(padded_data,lengths)
    total_sentences.append(list(np.argmax(masked_next_pred,axis=2)[0]))
    lengths = np.append(lengths,len(np.argmax(masked_next_pred,axis=2)[0]))
    # print(lengths.shape[0],lengths)
story=''
for i in total_sentences:
    i =set(i)
    for j in i:
        if j!=0:
            story+=idx2word[j]+" "
print(story)
