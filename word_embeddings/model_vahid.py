import re
import pickle
import random
import numpy

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


class Model:
    def __init__(self, text):
        self.sliding_token_size = 15
        self.path = None
        self.model = None
        self.tokens = text.split()
        self.tokenizer = Tokenizer(filters='')
        self.tokenizer.fit_on_texts(self.tokens)
        self.x = numpy.zeros((len(self.tokens) - self.sliding_token_size, self.sliding_token_size))
        self.y = numpy.zeros((len(self.tokens) - self.sliding_token_size, 1))
        sequences = self.tokenizer.texts_to_sequences(self.tokens)
        for token_n in range(len(self.tokens) - self.sliding_token_size):
            for sliding_token_n in range(self.sliding_token_size):
                self.x[token_n][sliding_token_n] = sequences[token_n + sliding_token_n][0]
            self.y[token_n] = sequences[token_n + self.sliding_token_size][0]

    def save(self):
        pickle.dump(self.model, open(self.path + 'model.bin', 'wb'))
        pickle.dump(self.tokens, open(self.path + 'tokens.bin', 'wb'))
        pickle.dump(self.tokenizer, open(self.path + 'tokenizer.bin', 'wb'))

    def load(self):
        self.model = pickle.load(open(self.path + 'model.bin', 'rb'))
        self.tokens = pickle.load(open(self.path + 'tokens.bin', 'rb'))
        self.tokenizer = pickle.load(open(self.path + 'tokenizer.bin', 'rb'))

    def generate(self):
        text = random.choice(self.tokens)
        for _ in range(100):
            sequences = pad_sequences([self.tokenizer.texts_to_sequences([text])[0]], self.sliding_token_size)
            text += ' ' + self.tokenizer.sequences_to_texts([self.model.predict_classes(sequences)])[0]
        return text
