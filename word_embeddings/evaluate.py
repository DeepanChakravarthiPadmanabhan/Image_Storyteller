from numpy import array
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from predict import Prediction
import json

with open('tokenizer.json') as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)
max_length = 7
pred = Prediction(tokenizer,max_length)
pred.load_model()
print(pred.predict_sequnce("Jack and",5))
