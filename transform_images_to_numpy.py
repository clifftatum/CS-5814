import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import datasets
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import os

if __name__ == '__main__':

    # (x_train,y_train),(x_test,y_test) = datasets.cfar100.load_data()
    # (x_train, y_train), (x_test, y_test) = datasets.cifar100.load_data()
    # (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data(label_mode='coarse')
    # plt.imshow(x_train[0])
    # plt.title(f"Label: {y_train[0][0]}")
    # plt.show()

    image_dir = r'C:\Users\cft5385\Documents\Learning\GradSchool\Repos\CS-5814\training_images'
    target_size = (255, 255)
    images = []
    labels = []
    len_dir =62232

    for dir in os.listdir(image_dir):
        path = os.path.join(image_dir, dir)
        if os.path.isdir(path):
            for i,img_name in enumerate(os.listdir(path)):
                img_path = os.path.join(path, img_name)
                img = load_img(img_path,target_size=target_size)
                img_array = img_to_array(img)/255
                images.append(img_array)
                if img_name.split('_')[0]=='coherent':
                    labels.append(0)
                elif img_name.split('_')[0]=='non-coherent':
                    labels.append(1)
                print(str(i) +'/'+str(len_dir))

    num_chunks = 20
    chunk_size = len(images) // num_chunks


    for i in range(num_chunks):
        start = i * chunk_size
        end = start + chunk_size if i < num_chunks - 1 else len(
            images)

        images_chunk = np.array(images[start:end], dtype="float32")
        labels_chunk = np.array(labels[start:end])


        data_chunk = {'images': images_chunk, 'labels': labels_chunk}
        np.save(
            rf'C:\Users\cft5385\Documents\Learning\GradSchool\Repos\CS-5814\training_images\caf_dataset_{i + 1}_of_20.npy',
            data_chunk)