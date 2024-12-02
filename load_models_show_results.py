import tensorflow as tf
from keras_preprocessing.image import load_img, img_to_array
from matplotlib import pyplot as plt
import os
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


loaded_model_LeNet = tf.keras.models.load_model(r"C:\Users\cft5385\Documents\Learning\GradSchool\Repos\CS-5814\models\trained\LeNet")
loaded_model_VGG_16 = tf.keras.models.load_model(r"C:\Users\cft5385\Documents\Learning\GradSchool\Repos\CS-5814\models\trained\VGG_16")

image_dir = os.path.join(os.getcwd(), "training_images\images")
# TF dataset
target_size = (168, 168)
train_dataset, test_dataset = create_dataset(image_dir, target_size, BATCH_SIZE)
score = loaded_model_LeNet.evaluate(test_dataset, verbose=VERBOSE)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
print('And other metrics/plots')