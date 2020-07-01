import tensorflow as tf
import time
import argparse

from constants import top_k, embedding_dim, units, attention_features_shape
from encoder import CNN_Encoder
from decoder import RNN_Decoder
from optimizers import select_optimizer
from util import get_architecture, load_image, validation_requirements
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Parser

parser = argparse.ArgumentParser("./evaluate.py")
parser.add_argument(
    '--optimizer', '-o',
    type=str,
    required=False,
    default='Adam',
    choices=['Adam', 'SGD', 'RMSProp'],
    help='Optimizer choice',
)
parser.add_argument(
    '--architecture', '-a',
    type=str,
    required=False,
    default='Inception',
    choices=['Inception', 'Inception-ResNet'],
    help='Architecture choice',
)
parser.add_argument(
    '--test_image', '-t',
    type=str,
    required=False,
    default='Valid',
    help='Path of test image. If no argument is passed, random images in the dataset is used.',
)
parser.add_argument(
    '--learning_rate', '-lr',
    type=float,
    default=0.001,
    help='Learning rate of optimizer to build checkpoints',
)

FLAGS, unparsed = parser.parse_known_args()
optimizer_name = FLAGS.optimizer
architecture = FLAGS.architecture
learning_rate = FLAGS.learning_rate
test_image = FLAGS.test_image

optimizer = select_optimizer(optimizer_name, learning_rate)

vocab_size = top_k + 1
# ToDo: Change encoder and decoder checkpoints depending on the architecture.
encoder = CNN_Encoder(embedding_dim)
decoder = RNN_Decoder(embedding_dim, units, vocab_size) # (256, 512, 5001)

tokenizer, max_length, cap_val, img_name_val = validation_requirements()

checkpoint_path = 'checkpoints/train/'
ckpt = tf.train.Checkpoint(encoder=encoder,
                           decoder=decoder,
                           optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
ckpt.restore(ckpt_manager.latest_checkpoint)

def evaluate(image):
    # Testing- Restore the checkpoint and predict- https://www.tensorflow.org/tutorials/text/nmt_with_attention
    image_features_extract_model = get_architecture(architecture)
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

def plot_attention(image, result, attention_plot):
    temp_image = np.array(Image.open(image))
    fig_raw = plt.figure(figsize=(10,10))
    plt.imshow(temp_image)
    plt.show()
    fig_raw.savefig('raw_images_' +'.pdf', bbox_inches='tight')
    fig = plt.figure(figsize=(10, 10))
    len_result = len(result)
    row = len_result // 2
    if row % 2 != 0:
        row = row + 1
    if len_result > 1:
        for l in range(len_result):
            temp_att = np.resize(attention_plot[l], (8, 8))
            ax = fig.add_subplot(row, row + 1, l + 1)
            ax.set_title(result[l])
            img = ax.imshow(temp_image)
            ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

    plt.tight_layout()
    plt.show()
    fig.savefig('evaluation_result_'+'.pdf', bbox_inches='tight')

# captions on the validation set
if test_image=='Valid':
    print('VAL image ')
    rid = np.random.randint(0, len(img_name_val))
    image = img_name_val[rid]
    real_caption = ' '.join([tokenizer.index_word[i] for i in cap_val[rid] if i not in [0]])
    reference = [tokenizer.index_word[i] for i in cap_val[rid] if i not in [0]]
    result, attention_plot = evaluate(image)
    print('Real Caption:', real_caption)

else:
    print('TEST image:', test_image)
    image = test_image
    result, attention_plot = evaluate(test_image)

print ('Prediction Caption:', ' '.join(result))
plot_attention(image, result, attention_plot)


