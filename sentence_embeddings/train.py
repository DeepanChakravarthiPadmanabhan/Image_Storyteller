import os
import time
import pickle
import datetime
import warnings
from config import *
from tqdm import tqdm
import tensorflow as tf
from model import layers
from model import lr_schedule
from data_loader import DataLoader
import nltk
import json
from numpy import array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
warnings.filterwarnings('ignore')
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'

# from absl import flags
# FLAGS = flags.FLAGS


def train():
    def compute_loss(labels, predictions):
        loss_fn = tf.keras.losses.sparse_categorical_crossentropy
        per_example_loss = loss_fn(labels, predictions, from_logits=True)
        return tf.reduce_sum(per_example_loss) * (1/global_batch_size)

    def train_step(inputs, optimizer):
        sentences, lengths = inputs
        lengths=tf.cast(lengths,tf.int32)
        with tf.GradientTape() as tape:
            masked_prev_pred, masked_next_pred = model(sentences, lengths)
            # print(sentences[:-1, :], masked_prev_pred)
            prev_loss = compute_loss(sentences[:-1, :], masked_prev_pred)
            next_loss = compute_loss(sentences[1:, :], masked_next_pred)
            losses = prev_loss + next_loss
            # print(masked_next_pred,masked_prev_pred,prev_loss,next_loss,"train step")
        grads = tape.gradient(losses, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        return losses

    def val_step(inputs):
        sentences, lengths = inputs
        masked_prev_pred, masked_next_pred = model(sentences, lengths)

        prev_loss = compute_loss(sentences[:-1, :], masked_prev_pred)
        next_loss = compute_loss(sentences[1:, :], masked_next_pred)
        losses = prev_loss + next_loss

        return losses

    # @tf.function
    # def distributed_train_step(dataset_inputs):
    #     per_replica_losses = strategy.experimental_run_v2(train_step, args=(dataset_inputs, optimizer))
    #     return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
    #
    # @tf.function
    # def distributed_test_step(dataset_inputs):
    #     return strategy.experimental_run_v2(val_step, args=(dataset_inputs,))

    for epoch in range(epochs):
        start = time.time()
        total_loss = 0.0
        num_batches = 0
        total_test_loss = 0.0
        num_test_batches = 0
        total_training_steps = (total_sent - val_size) // global_batch_size
        total_testing_steps = val_size // global_batch_size

        for step, x in tqdm(enumerate(train_dataset), total=total_training_steps):
            # print("start")
            # print("inside train---------------------------",step,max_length)
            current_lr = 0.001
            # print("111")
            current_loss = train_step(x,optimizer)/max_length
            # print("222")
            total_loss += current_loss
            num_batches += 1
            # print("1")
            # if step % 10 == 0:
            #     # print("2")
            #     with train_summary_writer.as_default():
            #         tf.summary.scalar('loss', current_loss, step=step)
            #         tf.summary.scalar('lr', current_lr ,step=step)
            #     template = ("Epoch {}, Step {}, Loss: {}, lr: {:.6f}")
            #     print(template.format(epoch + 1, step, current_loss, current_lr))
            # print("4")
            # if step % 100 == 0:
            #     checkpoint.save(checkpoint_prefix)
        try:
            # print("3")
            for step, x in tqdm(enumerate(val_dataset), total=total_testing_steps):
                # print("inside val---------------------------",step)

                current_test_loss = val_step(x)/max_length
                total_test_loss += current_test_loss
                num_test_batches += 1

                # if step % 100 == 0:
                #     with test_summary_writer.as_default():
                #         tf.summary.scalar('val_loss', current_test_loss, step=step)
                #     template = ("Epoch {}, Step {}, Val Loss: {}")
                #
                #     print(template.format(epoch + 1, step, current_test_loss))
        except:
            pass
        # print("5")
        total_train_loss = total_loss / num_batches
        total_test_loss = total_test_loss/num_test_batches
        if epoch % 1 == 0:
            checkpoint.save(checkpoint_prefix)
            template = ("Epoch {}, Total Train Loss: {}, Total Test Loss: {}")
            print(template.format(epoch + 1, total_train_loss, total_test_loss))
            print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("command",
                        metavar="<command>",
                        help="train from scratch or continue from last ckpt")
    parser.add_argument('--gpu',
                        required=False,
                        type=str,
                        metavar="choose which gpu to train on")
    parser.add_argument('--ckpt',
                        required=False,
                        type=str,
                        default='latest',
                        metavar='specific ckpt to continue')

    args = parser.parse_args()
    print("Command: ", args.command)
    print("Visible GPUs: ", args.gpu)
    print("Resume model from:", args.ckpt)

    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Create Distributed training strategy
    strategy = tf.distribute.MirroredStrategy()
    print('Number of GPUs: {}'.format(strategy.num_replicas_in_sync))
    global_batch_size = 128
    print("Global batch size: {}".format(global_batch_size))

    # with strategy.scope():
    model = layers.skip_thoughts(thought_size=thought_size, word_size=embed_dim, vocab_size=vocab_size,
                                 max_length=max_length)

    # custom lr schedules
    # lr = lr_schedule.CustomSchedule(1,warmu)
    lr = 0.1
    optimizer = tf.optimizers.Adam(learning_rate=lr, clipnorm=clip_gradient_norm)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

    if args.command == 'continue':
        if args.ckpt == 'latest':
            ckpt_path = tf.train.latest_checkpoint(checkpoint_dir)

        else:
            ckpt_path = os.path.join(checkpoint_dir, args.ckpt)
        print("Loading model from ", ckpt_path)
        checkpoint.restore(ckpt_path)

    elif args.command == 'train':
        print("Training model from scratch")
        pass

    else:
        print("Please enter 'train' or 'continue'!")
        print("Now Training model From scratch")
        pass

    try:
        model.summary()
    except:
        pass

    # load data, and train test split them.
    # sftp://smuthi2s@wr0.wr.inf.h-brs.de
    num_lines = 2000
    print("2000 lines dataset 240 embed size")
    with open('/scratch/smuthi2s/NLP_data/books/books_large_p1.txt', 'r') as in_file:
    # with open('/home/smuthi2s/perl5/NLP/Image_Storyteller/tf2-skip-thoughts/data.txt', 'r') as in_file:
        total_sentences = in_file.read().splitlines()[:num_lines]
        # total_sentences = nltk.sent_tokenize(text)
    top_k = 10000
    tokenizer = Tokenizer(num_words=top_k,oov_token="<unk>",filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
    tokenizer.fit_on_texts(total_sentences)
    with open('new_tokenizer_'+str(num_lines)+'.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(tokenizer.to_json(), ensure_ascii=False))
    encoded_data = tokenizer.texts_to_sequences(total_sentences)
    vocab_size = len(tokenizer.word_counts)+1
    # with open('./data/total_sentences', 'rb') as t:
    #     total_sentences = pickle.load(t)
    lengths = [int(len(i)) for i in encoded_data]
    # print(lengths)
    # with open('./data/lengths', 'rb') as l:
    #     lengths = pickle.load(l)

    print("Creating Tensorflow 2.0 datasets & distributed training strategy")
    # train_batches = tf.data.Dataset.from_generator(lambda: iter(zip(encoded_data, lengths)), output_types=(tf.int64, tf.int64))
    # print(train_batches.take(3).as_numpy_iterator(),"--------------")
    max_length = max(lengths)
    print(max_length)
    # print(encoded_data,vocab_size,max_length)
    sequences = pad_sequences(encoded_data, maxlen=max_length, padding='pre')
    sequences = array(sequences)
    dataset = tf.data.Dataset.from_tensor_slices((sequences, lengths))
    val_size = int(validation_size * len(encoded_data))
    print('Validation size: {}'.format(val_size))

    if val_size > 0:
        val_dataset = dataset.take(val_size).batch(global_batch_size)
        train_dataset = dataset.skip(val_size).batch(global_batch_size)

        # train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
        # val_dist_dataset = strategy.experimental_distribute_dataset(val_dataset)
    else:
        dataset = dataset.batch(global_batch_size)
        train_dist_dataset = strategy.experimental_distribute_dataset(dataset)

    current_time = datetime.datetime.now().strftime("%m%d-%H%M")
    train_log_dir = './logs/gradient_tape/' + current_time + '/train'
    test_log_dir = './logs/gradient_tape/' + current_time + '/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    # Start training
    # with strategy.scope():
    train()
