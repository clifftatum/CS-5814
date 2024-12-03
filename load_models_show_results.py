import tensorflow as tf
from keras_preprocessing.image import load_img, img_to_array
from matplotlib import pyplot as plt
import os
import keras
import tensorflow as tf
import numpy as np
import os
from sklearn.model_selection import train_test_split
from models.DCNN import *
from train_DCNN import create_dataset
EPOCHS = 50
BATCH_SIZE = 32
VERBOSE = 1
OPTIMIZER = tf.keras.optimizers.SGD()
VALIDATION_SPLIT = 0.90

lenet_path = os.path.join(os.getcwd(), "models", "trained", "LeNet")
vgg16_path = os.path.join(os.getcwd(), "models", "trained", "VGG_16")

loaded_model_LeNet = tf.saved_model.load(lenet_path)
loaded_model_VGG_16 = tf.saved_model.load(vgg16_path)

print(loaded_model_LeNet)

image_dir = os.path.join(os.getcwd(), "reconstructed_images")
# TF dataset
target_size = (168, 168)
train_dataset, test_dataset = create_dataset(image_dir, target_size, BATCH_SIZE)
score = loaded_model_LeNet.evaluate(test_dataset, verbose=VERBOSE)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
print('And other metrics/plots')