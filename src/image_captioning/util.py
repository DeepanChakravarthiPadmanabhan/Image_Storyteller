import tensorflow as tf
import numpy as np


def get_Inception():
    # Initialize Inception V3 and load the pretrained ImageNet weights
    # print('Reading InceptionNet')
    # image_model = tf.keras.applications.InceptionV3(include_top=False, # Whether to include the FCN at the last layer
    #                                              weights='imagenet') # Use pre-trained imagenet weights.
    # print('Reading RESNET')
    # image_model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet')
    print('Reading INCEPTION RESNET')
    image_model = tf.keras.applications.InceptionResNetV2(include_top=False, weights='imagenet')
    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output
    image_features_extract_model = tf.keras.Model(new_input, hidden_layer)
    return image_features_extract_model

def get_ResNet():

    pass

def calc_max_length(tensor):
    return max(len(t) for t in tensor)

def load_image(image_path):
    # Read image in the image path.
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    # Resize img 299,299 as per Inception V3 training details.
    img = tf.image.resize(img, (299,299))
    # Normalize pixel values between -1 and 1 as per Inception v3 training details.
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path


def map_func(img_name, cap):
    # Load the numpy files.
    img_tensor = np.load(img_name.decode('utf-8')+'.npy')
    return img_tensor, cap

