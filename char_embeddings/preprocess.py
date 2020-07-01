import re
import numpy as np
def preprocess_data(book_corpus):
    corpus = ''
    for sentence in book_corpus:
        # print(sentence)
        corpus+=str(sentence)
    corpus = re.sub(r'[^a-z\s]','',corpus)
    return corpus

def generate_one_hot_encoding(sentences, next_chars,maxlen,chars,char_indices):
    X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    length = len(sentences)
    index = 0
    for i in range(len(sentences)):
        sentence = sentences[i]
        for t, char in enumerate(sentence):
            X[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1
    return X, y

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)
