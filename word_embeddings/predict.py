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

class Prediction():
    def __init__(self,tokenizer,max_len):
        self.model = None
        self.tokenizer = tokenizer
        self.idx2word = {v:k for k,v in self.tokenizer.word_index.items()}
        self.max_length = max_len

    def load_model(self):
        self.model = load_model("lang_model.h5")

    def predict_sequnce(self,text,num_words):
        for id in range(num_words):
            encoded_data = self.tokenizer.texts_to_sequences([text])[0]
            padded_data = pad_sequences([encoded_data],maxlen = self.max_length-1,padding='pre')
            y_pred = self.model.predict(padded_data)
            y_pred = np.argmax(y_pred)
            predict_word = self.idx2word[y_pred]
            text += ' ' + predict_word
        return text
