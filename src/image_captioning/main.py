import tensorflow as tf
import time
import matplotlib.pyplot as plt
import argparse

from constants import top_k, embedding_dim, units, start_epoch, \
    checkpoint_path

from encoder import CNN_Encoder
from decoder import RNN_Decoder
from loss import loss_function
from optimizers import select_optimizer

from data_prep import DataLoader

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
    help='Batch size',
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
BATCH_SIZE = FLAGS.batch_size
num_epochs = FLAGS.num_epochs
architecture = FLAGS.architecture

optimizer = select_optimizer(optimizer_name)

data_getter = DataLoader()
data_getter.get_imgnames_captions()
max_length, cap_vector, tokenizer = data_getter.get_tokenizer('train')
dataset = data_getter.get_dataset('train', optimizer_name, architecture, BATCH_SIZE)
dataset_val = data_getter.get_dataset('validation', optimizer_name, architecture, BATCH_SIZE)
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
def train_step(img_tensor, target, module='train'):
    loss = 0
    # initializing the hidden state for each batch
    # because the captions are not related from image to image.

    hidden = decoder.reset_state(batch_size=target.shape[0])  # shape: (bs, 512)
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * target.shape[0], 1)  # shape: (bs,1)

    with tf.GradientTape() as tape:
        features = encoder(img_tensor)  # features shape: (bs, 64, 256)

        for i in range(1, target.shape[1]):
            # passing the features through the decoder.
            predictions, hidden, _ = decoder(dec_input, features, hidden)

            loss += loss_function(target[:, i], predictions)

            # using teacher forcing
            dec_input = tf.expand_dims(target[:, i], 1)

    total_loss = (loss / int(target.shape[1]))
    print('Propagating weights in train')
    trainable_variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))

    return loss, total_loss

loss_plot = []
loss_plot_val = []

for epoch in range(start_epoch, EPOCHS):

    start = time.time()
    total_loss = 0

    for (batch, (img_tensor, target)) in enumerate(dataset):
        batch_loss, t_loss = train_step(img_tensor, target)
        total_loss += t_loss

        if batch % 100 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(
                epoch + 1, batch, batch_loss.numpy() / int(target.shape[1])))

    # storing the epoch end loss value to plot later.
    loss_plot.append(total_loss / num_steps)

    if epoch % 5 == 0:
        ckpt_manager.save()

    print('Epoch {} Loss {:.6f}'.format(epoch + 1, total_loss / num_steps))
    print('Time taken for 1 epoch {} second \n'.format(time.time() - start))

    ############### VALIDATION #######################

    # total_loss_val = 0
    #
    # for (batch_val, (img_tensor_val, target_val)) in enumerate(dataset_val):
    #     batch_loss_val, t_loss_val = train_step(img_tensor_val, target_val, 'validation')
    #     total_loss_val += t_loss_val
    #
    #     if batch_val % 100 == 0:
    #         print('Validation: Epoch {} Batch {} Loss {:.4f}'.format(
    #             epoch + 1, batch_val, batch_loss_val.numpy() / int(target_val.shape[1])))
    #
    # # storing the epoch end loss value to plot later.
    # loss_plot_val.append(total_loss_val / num_steps_val)
    # print('Validation: Epoch {} Loss {:.6f}'.format(epoch + 1, total_loss_val / num_steps_val))
    # print('Validation: Time taken for 1 epoch {} second \n'.format(time.time() - start))

fig_name = plt.figure(figsize=(12,10))
plt.plot(loss_plot, label='Train loss')
plt.plot(loss_plot_val, label='Validation loss')
plt.legend()
plt.grid()
plt.show()
fig_name.savefig('trainval_curve.pdf', bbox_inches='tight')




