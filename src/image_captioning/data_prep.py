import tensorflow as tf
import os

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import json
import tqdm
import numpy as np
import pickle

from constants import annotation_folder, image_folder, num_examples, top_k, \
    BUFFER_SIZE, BATCH_SIZE
from util import get_Inception, calc_max_length, load_image, map_func

class DataLoader():
    def __init__(self):
        self.annotation_file = os.path.abspath('.') + annotation_folder + 'captions_train2014.json'
        self.PATH = os.path.abspath('.') + image_folder
        self.caption_filename = os.path.abspath('.') + "/train_captions.txt"


    def get_imgnames_captions(self):
        with open(self.annotation_file, 'r') as f:
            annotations = json.load(f)

        all_captions = []
        all_img_name_vector = []

        # annotations['annotations'] - list
        # annotations - dict
        for annot in annotations['annotations']:
            caption = '<start> ' + annot['caption'] + '<end>'
            image_id = annot['image_id']
            full_coco_image_path = self.PATH + 'COCO_train2014_' + '%012d.jpg' % (image_id)
            all_img_name_vector.append(full_coco_image_path)
            all_captions.append(caption)

        self.train_captions, self.img_name_vector = shuffle(all_captions, all_img_name_vector,
                                                  random_state=1)
        self.train_captions = self.train_captions[:num_examples]
        self.img_name_vector = self.img_name_vector[:num_examples]
        print("Num captions:",len(self.train_captions))
        self.save_captions()

    def save_captions(self, captions=None):
        if captions==None:
            captions = self.train_captions
        try:
            os.remove(self.caption_filename)
        except OSError:
            pass

        with open(self.caption_filename, 'wb') as fp:
            pickle.dump(captions, fp)

    def read_captions(self, filename):
        if filename == None:
            filename = self.caption_filename
        with open(filename, 'rb') as fp:
            captions = pickle.load(fp)
        return captions


    def get_tokenizer(self):
        train_captions = self.read_captions(self.caption_filename)
        # print('Caption after reading:',len(train_captions))
        tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                          oov_token="<unk>",
                                                          filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
        tokenizer.fit_on_texts(train_captions)
        # print(tokenizer.word_counts) # OrderedDict of word and count
        # print(tokenizer.document_count)  # Number of captions
        # print(tokenizer.word_index) # Dict of word and respective index value encoded
        # print(tokenizer.word_docs)
        tokenizer.word_index['<pad>'] = 0
        tokenizer.index_word[0] = '<pad>'
        train_seqs = tokenizer.texts_to_sequences(train_captions)
        # Pad each vector to the max_length of the captions.
        # Since max length is not given pad_sequences calculates automatically.
        cap_vector = tf.keras.preprocessing.sequence.pad_sequences(
            train_seqs, padding='post')
        max_length = calc_max_length(train_seqs)

        return max_length, cap_vector, tokenizer


    def get_dataset(self):
        # Get unique images.
        encode_train = sorted(set(self.img_name_vector))
        # Get InceptionV3
        image_features_extract_model = get_Inception()
        # batch_size fixed depending on the system configuration.
        # Source data from the input data.
        # Iteration happens in a streaming manner and
        # dataset is not fit in the memory.
        image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
        image_dataset = image_dataset.map(
            load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(16)

        for img, path in tqdm.tqdm(image_dataset):
            batch_features = image_features_extract_model(img)
            #     print('Batch features extracted:',batch_features.shape)
            batch_features = tf.reshape(batch_features,
                                        (batch_features.shape[0], -1, batch_features.shape[3]))
            #     print('Reshaped batch features:',batch_features.shape)
            for bf, p in zip(batch_features, path):
                path_of_feature = p.numpy().decode('utf-8')
                np.save(path_of_feature, bf.numpy())

        # Split training and testing data.
        # Create training and validation sets using 80-20 split.
        # image file name for training and validation
        # caption for training and validation

        self.get_imgnames_captions()
        max_length, cap_vector, tokenizer = self.get_tokenizer()

        self.img_name_train, self.img_name_val, self.cap_train, self.cap_val = train_test_split(self.img_name_vector,
                                                                            cap_vector,
                                                                            test_size=0.2,
                                                                            random_state=0)
        print(len(self.img_name_train), len(self.cap_train), len(self.img_name_val), len(self.cap_val))

        dataset = tf.data.Dataset.from_tensor_slices((self.img_name_train, self.cap_train))
        dataset = dataset.map(lambda item1, item2: tf.numpy_function(
            map_func, [item1, item2], [tf.float32, tf.int32]),
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)

        dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        dataset_val = tf.data.Dataset.from_tensor_slices((self.img_name_val, self.cap_val))
        dataset_val = dataset_val.map(lambda item1, item2: tf.numpy_function(
            map_func, [item1, item2], [tf.float32, tf.int32]),
                                      num_parallel_calls=tf.data.experimental.AUTOTUNE)

        dataset_val = dataset_val.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
        dataset_val = dataset_val.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


        return dataset, dataset_val