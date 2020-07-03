from numpy import array
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from predict import Prediction
import json

# Load saved tokenizer
with open('tokenizer_30000.json') as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)
max_length = 179
pred = Prediction(tokenizer,max_length)
pred.load_model()
#we were laughing , haha thats funny but then we were saying damn it all , what kind of example is this guy setting for the kids ?
#this is just great , seth shouted .
#he might be into it .
#just try to be less annoying than usual
#this may say something important

# Predict for given sequence
print(pred.predict_sequnce("just try to be less annoying than usual",100))
