from numpy import array
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import load_model
import re
from preprocess import preprocess_data, generate_one_hot_encoding

book_corpus = open('/scratch/smuthi2s/NLP_data/books/books_large_p1.txt', 'rb').readlines()[:20000]
print ('Length of text: {} characters'.format(len(book_corpus)))
corpus = preprocess_data(book_corpus)
chars = sorted(list(set(corpus)))




char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

print(char_indices)
print(indices_char)



maxlen = 40 # The window size
step = 3 # The steps between the windows
sentences = []
next_chars = []
for i in range(0, len(corpus) - maxlen, step):
    sentences.append(corpus[i: i + maxlen]) # range from current index i for max length characters
    next_chars.append(corpus[i + maxlen]) # the next character after that
sentences = np.array(sentences)
next_chars = np.array(next_chars)
print('Number of sequences:', len(sentences))
print(sentences[0],'\n',next_chars[0])




X, y = generate_one_hot_encoding(sentences,next_chars,maxlen,chars,char_indices)
print('Build model...')
model = Sequential()
model.add(LSTM(128,input_shape=(maxlen, len(chars))))
model.add(Dropout(0.1))
model.add(Dense(len(chars),activation="softmax"))

model.compile(loss='categorical_crossentropy', optimizer='adam')

print("Compiling model complete...")
model.summary()

EPOCHS = 50
history = model.fit(X, y,batch_size=32, epochs=EPOCHS)
model.save('char_story_generator_20000_'+str(EPOCHS)+'.h5')
