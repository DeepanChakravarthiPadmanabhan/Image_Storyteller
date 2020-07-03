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
from preprocess import Preprocessing
from model import Model

# Argument is PATH to dataset
pr = Preprocessing('/scratch/smuthi2s/NLP_data/books/books_large_p1.txt')
# pr = Preprocessing('data.txt')
pr.load_data(num_lines=60000)
pr.encode_data()
pr.generate_sequence()
pr.get_data()
print("Maximum length of sequence : ",pr.get_max_length())
pr.save_config()

params = {"activation":"softmax","epochs":100,"verbose":2,"loss":"categorical_crossentropy",
          "optimizer":"adam","metrics":['accuracy'],"vocab_size":pr.vocab_size,"max_len":pr.max_length}
model_obj = Model(params)
model_obj.x = pr.x
model_obj.y = pr.y
model_obj.create_model()

model_obj.run()
model_obj.save()
