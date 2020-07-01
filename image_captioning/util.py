import tensorflow as tf
import numpy as np
import pickle

from constants import cap_val_file, img_name_val_file


def get_architecture(architecture = 'Inception'):
    if architecture == 'Inception':
    # Initialize Inception V3 and load the pretrained ImageNet weights

        image_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet') 
    elif architecture == 'Inception-Resnet':

        image_model = tf.keras.applications.InceptionResNetV2(include_top=False, weights='imagenet')
    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output
    image_features_extract_model = tf.keras.Model(new_input, hidden_layer)
    return image_features_extract_model

def calc_max_length(tensor):
    return max(len(t) for t in tensor)

def load_image(image_path):
    # Read image in the image path.
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    # Resize img 299,299 as per Inception V3 training details.
    # img = tf.image.resize(img, (299,299))
    img = tf.compat.v1.image.resize_image_with_crop_or_pad(img, 299, 299)
    img = tf.cast(img, tf.float32) / 255.0
    # Normalize pixel values between -1 and 1 as per Inception v3 training details.
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path


def map_func(img_name, cap):
    # Load the numpy files.
    img_tensor = np.load(img_name.decode('utf-8')+'.npy')
    return img_tensor, cap

def validation_requirements():
    with open('tokenizer.pkl', 'rb') as fp:
        tokenizer = pickle.load(fp)
    with open('max_length.pkl', 'rb') as fp:
        max_length = pickle.load(fp)
    with open('cap_vector.pkl', 'rb') as fp:
        cap_vector = pickle.load(fp)
    with open(cap_val_file + '.pkl', 'rb') as fp:
        cap_val = pickle.load(fp)
    with open( img_name_val_file +'.pkl', 'rb') as fp:
        img_name_val = pickle.load(fp)

    return tokenizer, max_length, cap_val, img_name_val

