from tensorflow.keras.models import load_model
from preprocess import preprocess_data
import numpy as np
import re
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# PATH for dataset location
book_corpus = open('/scratch/smuthi2s/NLP_data/books/books_large_p1.txt', 'rb').readlines()[:20]
print ('Length of text: {} characters'.format(len(book_corpus)))
# def preprocess_data(book_corpus):
#     corpus = ''
#     for song in book_corpus:
#         corpus+=song
#     corpus = re.sub(r'[^a-z\s]','',corpus)
#     return corpus


corpus = preprocess_data(book_corpus)
corpus = corpus[:1000000]
chars = sorted(list(set(corpus)))
print(chars)
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# PATH to load saved model
model = load_model("char_story_generator_20100.h5")
variance = 0.5
print('Variance: ', variance)
maxlen = 100
# Input sentence
sentence = 'once upon a time'
# sentence = 'and both that morning and both'
generated = ''
original = sentence
window = sentence
# Predict the next 400 characters based on the seed
for i in range(100):
    x = np.zeros((1, maxlen, len(chars)))
    for t, char in enumerate(window):
        x[0, t, char_indices[char]] = 1.

    preds = model.predict(x, verbose=0)[0]
    next_index = sample(preds, variance)
    next_char = indices_char[next_index]

    generated += next_char
    window = window[1:] + next_char

print(original + generated)
