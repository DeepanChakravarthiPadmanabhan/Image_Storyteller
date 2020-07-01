import tensorflow as tf
import time
import matplotlib.pyplot as plt
import argparse
import numpy as np

from constants import top_k, embedding_dim, units, start_epoch, \
    checkpoint_path, attention_features_shape

from encoder import CNN_Encoder
from decoder import RNN_Decoder
from loss import loss_function
from optimizers import select_optimizer
from util import get_architecture, load_image
from metrics import compute_BLEU_score
import pandas as pd
from data_prep import DataLoader
from PIL import Image
import os

# Parser
parser = argparse.ArgumentParser("./main.py")
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
    '--num_epochs', '-n',
    type=int,
    default=20,
    help='Maximum epochs',
)
parser.add_argument(
    '--architecture', '-a',
    type=str,
    required=True,
    choices=['Inception', 'Inception-ResNet'],
    help='Architecture choice',
)
parser.add_argument(
    '--num_examples', '-ex',
    type=int,
    default=1000,
    help='Number of training examples',
)
parser.add_argument(
    '--learning_rate', '-lr',
    type=float,
    default=0.0001,
    help='Learning rate',
)
parser.add_argument(
    '--test_validation', '-tv',
    type=str,
    default=True,
    help='Evaluate all validation images to provide metric',
)
parser.add_argument(
    '--validation_set_size', '-v',
    type=int,
    default=5,
    help='Number of images to test in validation',
)
parser.add_argument(
    '--annotation_folder',
    type=str,
    default='/home/dpadma2s/obj_det/deepan/Image_Storyteller/src/image_captioning/annotations/',
    help='Annotation file for all images in data',
)
parser.add_argument(
    '--image_folder',
    type=str,
    default='/home/dpadma2s/obj_det/deepan/Image_Storyteller/src/image_captioning/train2014/',
    help='Path for all images in dataset',
)

FLAGS, unparsed = parser.parse_known_args()

optimizer_name = FLAGS.optimizer
BATCH_SIZE = FLAGS.batch_size
num_epochs = FLAGS.num_epochs
architecture = FLAGS.architecture
num_examples = FLAGS.num_examples
learning_rate = FLAGS.learning_rate
test_validation = FLAGS.test_validation
validation_set_size = FLAGS.validation_set_size
annotation_folder = FLAGS.annotation_folder
image_folder = FLAGS.image_folder

optimizer = select_optimizer(optimizer_name, learning_rate)

data_getter = DataLoader(annotation_folder, image_folder, num_examples)
data_getter.get_imgnames_captions()
max_length, cap_vector, tokenizer = data_getter.get_tokenizer()
dataset, dataset_val = data_getter.get_dataset(architecture, BATCH_SIZE)
img_name_train, img_name_val = data_getter.img_name_train, data_getter.img_name_val
cap_train, cap_val = data_getter.cap_train, data_getter.cap_val

vocab_size = top_k + 1
encoder = CNN_Encoder(embedding_dim)
decoder = RNN_Decoder(embedding_dim, units, vocab_size) # (256, 512, 5001)
EPOCHS = num_epochs
num_steps = len(img_name_train)// BATCH_SIZE
num_steps_val = len(img_name_val)//BATCH_SIZE

ckpt = tf.train.Checkpoint(encoder=encoder,
                           decoder=decoder,
                           optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

if ckpt_manager.latest_checkpoint:
    start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
    # restoring the latest checkpoint in checkpoint_path
    ckpt.restore(ckpt_manager.latest_checkpoint)

@tf.function
def train_step(img_tensor, target):
    loss = 0
    # initializing the hidden state for each batch
    # because the captions are not related from image to image.

    hidden = decoder.reset_state(batch_size=target.shape[0])  # shape: (bs, 512)
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * target.shape[0], 1)  # shape: (bs,1)

    # with tf.GradientTape() as tape:
    #     features = encoder(img_tensor)  # features shape: (bs, 64, 256)
    #
    #     for i in range(1, target.shape[1]):
    #         # passing the features through the decoder.
    #         predictions, hidden, _ = decoder(dec_input, features, hidden)
    #
    #         loss += loss_function(target[:, i], predictions)
    #
    #         # using teacher forcing
    #         dec_input = tf.expand_dims(target[:, i], 1)
    #
    # total_loss = (loss / int(target.shape[1]))
    # trainable_variables = encoder.trainable_variables + decoder.trainable_variables
    # gradients = tape.gradient(loss, trainable_variables)
    # optimizer.apply_gradients(zip(gradients, trainable_variables))

    features = encoder(img_tensor)  # features shape: (bs, 64, 256)

    for i in range(1, target.shape[1]):
        # passing the features through the decoder.
        predictions, hidden, _ = decoder(dec_input, features, hidden)

        loss += loss_function(target[:, i], predictions)
        # using teacher forcing
        dec_input = tf.expand_dims(target[:, i], 1)

    total_loss = (loss / int(target.shape[1]))
    trainable_variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tf.gradients(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))


    return loss, total_loss

@tf.function
def validation_step(img_tensor, target):
    loss = 0

    # initializing the hidden state for each batch.
    # because the captions are not related from image to image.

    hidden = decoder.reset_state(batch_size=target.shape[0])  # shape: (bs, 512)
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * target.shape[0], 1)  # shape: (bs,1)

    features = encoder(img_tensor)  # features shape: (bs, 64, 256)

    for i in range(1, target.shape[1]):
        # passing the features through the decoder.
        predictions, hidden, _ = decoder(dec_input, features, hidden)
        loss += loss_function(target[:, i], predictions)
        # using teacher forcing
        dec_input = tf.expand_dims(target[:, i], 1)

    total_loss = (loss / int(target.shape[1]))

    return loss, total_loss

loss_plot = []
loss_plot_val = []
early_stopping_threshold = 2
early_stopper = 0
epoch = 0
run = True

while run:

    start = time.time()
    total_loss = 0

    for (batch, (img_tensor, target)) in enumerate(dataset):
        batch_loss, t_loss = train_step(img_tensor, target)
        total_loss += t_loss

        if batch % 100 == 0:
            print('Train: Epoch {} Batch {} Loss {:.4f}'.format(
                epoch + 1, batch, batch_loss.numpy() / int(target.shape[1])))

    # storing the epoch end loss value to plot later.
    loss_plot.append(total_loss / num_steps)

    if epoch % 5 == 0:
        ckpt_manager.save()

    print('Train: Epoch {} Loss {:.6f}'.format(epoch + 1, total_loss / num_steps))
    print('Train: Time taken for 1 train epoch {} second'.format(time.time() - start))


    ############### VALIDATION #######################
    if epoch%2==0:
        total_loss_val = 0

        for (batch_val, (img_tensor_val, target_val)) in enumerate(dataset_val):
           batch_loss_val, t_loss_val = validation_step(img_tensor_val, target_val)
           total_loss_val += t_loss_val

           # if batch_val % 100 == 0:
           #     print('Validation: Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch_val, batch_loss_val.numpy() / int(target_val.shape[1])))

        # storing the epoch end loss value to plot later.
        loss_plot_val.append(total_loss_val / num_steps_val)
        print('Validation: Epoch {} Loss {:.6f}'.format(epoch + 1, total_loss_val / num_steps_val))
        print('Validation: Time taken for 1 epoch {} second \n'.format(time.time() - start))
        print('***** Loss difference: {} *****'.format(loss_plot_val[-1].numpy()- loss_plot[-1].numpy()))

        # Early stopping
        if (loss_plot_val[-1].numpy() - loss_plot[-1].numpy() < 1) and (loss_plot[-1].numpy() < 0.5):
            early_stopper +=1
            if early_stopper == early_stopping_threshold:
                run = False
        else:
            early_stopper = 0

    if epoch >= EPOCHS:
        run = False

    epoch += 1

# print('Train loss curve:', loss_plot)
# print('Validation loss curve:', loss_plot_val)

fig_name = plt.figure(figsize=(12,10))
plt.plot(loss_plot, label='Train loss')
plt.plot(loss_plot_val, label='Validation loss')
plt.legend()
plt.grid()
# plt.show()
fig_name.savefig('trainval_curve.pdf', bbox_inches='tight')


def get_all_predictions(image):

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

        attention_plot[i] = tf.reshape(attention_weights, (-1,)).numpy()

        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        result.append(tokenizer.index_word[predicted_id])

        if tokenizer.index_word[predicted_id] == '<end>':
            return result, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot

def plot_attention(image, result, attention_plot, rid):
    temp_image = np.array(Image.open(image))
    fig_raw = plt.figure(figsize=(10,10))
    plt.imshow(temp_image)
    # plt.show()
    fig_raw.savefig('results/raw_images_' +str(rid)+'.png', bbox_inches='tight')
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
    # plt.show()
    fig.savefig('results/evaluation_result_'+str(rid)+'.pdf', bbox_inches='tight')

if test_validation=='True':
    print('Using Validation for validation')
    images_validation = img_name_val
    captions_validation = cap_val
else:
    # For development purpose to test whether training proceeded properly
    print('Using Train for validation')
    images_validation = img_name_train
    captions_validation = cap_train

if not os.path.exists('results'):
    os.makedirs('results')

BLEU = []
for rid, image in enumerate(images_validation[:validation_set_size]):
    real_caption = ' '.join([tokenizer.index_word[i] for i in captions_validation[rid] if i not in [0]])
    result, attention_plot = get_all_predictions(image)
    reference =  [tokenizer.index_word[i] for i in captions_validation[rid] if i not in [0]]
    BLEU.append(list(compute_BLEU_score(reference, result)))
    print('RID:',rid)
    print ('Real Caption:', real_caption)
    print ('Prediction Caption:', ' '.join(result))
    plot_attention(image, result, attention_plot, rid)

BLEU_mean = np.round(np.mean(np.array(BLEU), axis=0),4)
metrics = {'Val_images': [len(images_validation)],'BLEU-1': [BLEU_mean[0]], 'BLEU-2': [BLEU_mean[1]],
           'BLEU-3': [BLEU_mean[2]], 'BLEU-4': [BLEU_mean[3]]}
metrics_df = pd.DataFrame(metrics)
metrics_filename = 'metrics.csv'
metrics_df.to_csv(metrics_filename, mode='w+', header=True)

# encoder.save('encoder_full')
# ToDo: Save decoder feature
# call_output = decoder.call.get_concrete_function([tf.TensorSpec(shape=[None, 1], dtype=tf.int32, name='x'), tf.TensorSpec(shape=[None, 64, 256], dtype=tf.float32, name="features"), tf.TensorSpec(shape=[None, 512], dtype=tf.float32, name="hidden")])
# module_output_path = 'module_with_output_name'
# tf.saved_model.save(decoder, module_output_path, signatures={'serving_default': call_output})
# https://github.com/tensorflow/tensorflow/issues/31962




