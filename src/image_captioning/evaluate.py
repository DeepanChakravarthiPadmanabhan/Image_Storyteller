import tensorflow as tf
import time
import argparse

from constants import top_k, embedding_dim, units, attention_features_shape, checkpoint_path
from encoder import CNN_Encoder
from decoder import RNN_Decoder
from optimizers import select_optimizer
from util import get_Inception, load_image
from data_prep import DataLoader
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import nltk
import pandas as pd

# Parser

parser = argparse.ArgumentParser("./evaluate.py")
parser.add_argument(
    '--optimizer', '-o',
    type=str,
    required=True,
    choices=['Adam', 'SGD', 'RMSProp'],
    help='Optimizer choice',
)
parser.add_argument(
    '--batch_size', '-bs',
    type=int,
    default=64,
    help='Batch size',
)
parser.add_argument(
    '--visualize', '-v',
    type=str,
    default=False,
    help='Visualize each validation images with attention',
)
parser.add_argument(
    '--test_validation', '-t',
    type=str,
    default=True,
    help='Load validation data directly for validation',
)
parser.add_argument(
    '--architecture', '-a',
    type=str,
    required=True,
    choices=['Inception', 'ResNet101', 'ResNet50'],
    help='Architecture choice',
)

FLAGS, unparsed = parser.parse_known_args()


optimizer_name = FLAGS.optimizer
architecture = FLAGS.architecture
test_validation = FLAGS.test_validation
BATCH_SIZE = FLAGS.batch_size
VISUALIZE = FLAGS.visualize

optimizer = select_optimizer(optimizer_name)


vocab_size = top_k + 1
encoder = CNN_Encoder(embedding_dim)
decoder = RNN_Decoder(embedding_dim, units, vocab_size) # (256, 512, 5001)


data_getter = DataLoader()
data_getter.get_imgnames_captions()
max_length, cap_vector, tokenizer = data_getter.get_tokenizer('validation')
dataset_val = data_getter.get_dataset('validation', optimizer_name, architecture, BATCH_SIZE)
img_name_train, cap_train = data_getter.img_name_train, data_getter.cap_train
img_name_val, cap_val = data_getter.img_name_val, data_getter.cap_val

ckpt = tf.train.Checkpoint(encoder=encoder,
                           decoder=decoder,
                           optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)


def evaluate(image):
    # Testing- Restore the checkpoint and predict- https://www.tensorflow.org/tutorials/text/nmt_with_attention

    ckpt.restore(ckpt_manager.latest_checkpoint)

    image_features_extract_model = get_Inception()
    attention_plot = np.zeros((max_length, attention_features_shape))

    hidden = decoder.reset_state(batch_size=1)

    temp_input = tf.expand_dims(load_image(image)[0], 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    features = encoder(img_tensor_val)

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)

        attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        result.append(tokenizer.index_word[predicted_id])

        if tokenizer.index_word[predicted_id] == '<end>':
            return result, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot

def plot_attention(image, result, attention_plot, rid):
    temp_image = np.array(Image.open(image))
    plt.imshow(temp_image)
    fig = plt.figure(figsize=(10, 10))
    len_result = len(result)
    for l in range(len_result):
        temp_att = np.resize(attention_plot[l], (8, 8))
        ax = fig.add_subplot(len_result//2, len_result//2, l+1)
        ax.set_title(result[l])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

    plt.tight_layout()
    plt.show()
    fig.savefig('evaluation_result_'+str(rid)+'.pdf', bbox_inches='tight')


def compute_BLEU_score(reference, hypothesis):
    BLEU_1 = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis, weights=(1.0, 0.0, 0.0, 0.0))
    BLEU_2 = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis, weights=(0.5, 0.5, 0.0, 0.0))
    BLEU_3 = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis, weights=(0.33, 0.33, 0.33, 0.0))
    BLEU_4 = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis, weights=(0.25, 0.25, 0.25, 0.25))
    return BLEU_1, BLEU_2, BLEU_3, BLEU_4

# rid = np.random.randint(0, len(img_name_val))

BLEU = []

if test_validation=='True':
    print('Using Validation for validation')
    images_validation = img_name_val
    captions_validation = cap_val
else:
    print('Using Train for validation')
    images_validation = img_name_train
    captions_validation = cap_train



for rid, image in enumerate(images_validation[:5]):
    image = images_validation[rid]
    real_caption = ' '.join([tokenizer.index_word[i] for i in captions_validation[rid] if i not in [0]])
    hypothesis, attention_plot = evaluate(image)
    reference =  [tokenizer.index_word[i] for i in captions_validation[rid] if i not in [0]]
    BLEU.append(list(compute_BLEU_score(reference, hypothesis)))

    print ('Real Caption:', real_caption)
    print ('Prediction Caption:', ' '.join(hypothesis))
    if VISUALIZE:
        plot_attention(image, hypothesis, attention_plot, rid)


BLEU_mean = np.round(np.mean(np.array(BLEU), axis=0),4)
metrics = {'Val_images': [len(images_validation)],'BLEU-1': [BLEU_mean[0]], 'BLEU-2': [BLEU_mean[1]],
           'BLEU-3': [BLEU_mean[2]], 'BLEU-4': [BLEU_mean[3]]}
metrics_df = pd.DataFrame(metrics)
metrics_filename = 'metrics.csv'
metrics_df.to_csv(metrics_filename, mode='w+', header=True)
