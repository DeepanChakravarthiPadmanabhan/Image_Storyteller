from lstm_model import *

with open('data.txt') as f:
    dataset = list(f)
lstm_model = LSTMModel(dataset)
lstm_model.train()
lstm_model.save()
