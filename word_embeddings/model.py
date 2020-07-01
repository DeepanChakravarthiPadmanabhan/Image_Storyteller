from numpy import array
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import load_model

class Model():
    def __init__(self,params):
        self.model = None
        self.history = None
        self.x = None
        self.y = None
        self.vocab_size = params['vocab_size']
        self.max_len = params['max_len']
        self.activation = params['activation']
        self.optimizer = params['optimizer']
        self.epochs = params['epochs']
        self.metrics = params['metrics']


    def create_model(self):
        self.model = Sequential()
        self.model.add(Embedding(self.vocab_size,10,input_length=self.max_len-1))
        self.model.add(LSTM(50))
        self.model.add(Dropout(0.1))
        self.model.add(Dense(self.vocab_size,activation=self.activation))
        self.model.compile(loss='categorical_crossentropy',optimizer=self.optimizer,metrics=self.metrics)
        print(self.model.summary())
    def run(self):
        self.history = self.model.fit(self.x,self.y,epochs=self.epochs)

    def save(self):
        self.model.save("lang_model.h5")
