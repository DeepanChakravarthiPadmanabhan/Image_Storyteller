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
import json

class Preprocessing():

    def __init__(self,input_file):
        self.input_data_file = input_file
        self.data = None
        self.vocab_size = None
        self.encoded_data = None
        self.max_length = None
        self.sequences = None
        self.x = None
        self.y = None
        self.tokenizer = None
        self.num_lines = 0

    def load_data(self,num_lines):
        self.num_lines = num_lines
        fp = open(self.input_data_file,'r')
        self.data = fp.read().splitlines()[:num_lines]
        fp.close()
        print("done loading")

    def encode_data(self):
        top_k = 10000
        self.tokenizer = Tokenizer(num_words=top_k,oov_token="<unk>",filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
        self.tokenizer.fit_on_texts(self.data)
        self.encoded_data = self.tokenizer.texts_to_sequences(self.data)
        self.vocab_size = len(self.tokenizer.word_counts)+1

    def generate_sequence(self):
        seq_list = list()
        for item in self.encoded_data:
            l = len(item)
            for id in range(1,l):
                seq_list.append(item[:id+1])
        self.max_length = max([len(seq) for seq in seq_list])
        self.sequences = pad_sequences(seq_list, maxlen=self.max_length, padding='pre')
        self.sequences = array(self.sequences)

    def get_data(self):
        self.x = self.sequences[:,:-1]
        self.y = self.sequences[:,-1]
        self.y = to_categorical(self.y,num_classes=self.vocab_size)
    def get_config(self):
        return self.tokenizer.to_json()
    def save_config(self):
        with open('new_tokenizer_'+str(self.num_lines)+'.json', 'w', encoding='utf-8') as f:
            f.write(json.dumps(self.get_config(), ensure_ascii=False))
        # with open('tokenizer_config.json', 'w') as f:
        #     f.write(json.dumps(self.get_config()))
            # json.dump(self.get_config(), f)
    def get_max_length(self):
        return self.max_length
