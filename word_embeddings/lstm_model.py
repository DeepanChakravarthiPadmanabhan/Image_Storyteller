from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.optimizers import Adam

from model_vahid import Model
# from onfig import batch_size, epochsc
batch_size = 32
epochs = 2

class LSTMModel(Model):
    def __init__(self, text):
        super(LSTMModel, self).__init__(text)
        self.path = '..\\save\\lstm\\'

    def train(self):
        self.model = Sequential()
        self.model.add(Embedding(len(self.tokenizer.word_index) + 1, 100, input_length=self.sliding_token_size))
        self.model.add(LSTM(100, activation='relu', return_sequences=True))
        self.model.add(LSTM(100, activation='relu', return_sequences=True))
        self.model.add(LSTM(100, activation='relu'))
        self.model.add(Dense(100, activation='relu'))
        self.model.add(Dense(100, activation='relu'))
        self.model.add(Dense(len(self.tokenizer.word_index) + 1, activation='softmax'))
        self.model.compile(Adam(learning_rate=0.0005), 'sparse_categorical_crossentropy')
        self.model.fit(self.x, self.y, batch_size, epochs)
