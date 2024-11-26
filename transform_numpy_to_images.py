import numpy as np
import os
from keras_preprocessing.image import array_to_img

# Define the directory where your .npy files are saved
npy_dir = os.path.join(os.getcwd(), "training_images")

# Directory to save the reconstructed images
output_dir = os.path.join(os.getcwd(), "reconstructed_images")
os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist


for j,npy_file in enumerate(os.listdir(npy_dir)):
    if npy_file.endswith('.npy'):
        npy_path = os.path.join(npy_dir, npy_file)
        print(f"Processing {npy_file}")

        data_chunk = np.load(npy_path, allow_pickle=True).item()
        images = data_chunk['images']
        labels = data_chunk['labels']

        for i, (img_array, label) in enumerate(zip(images, labels)):

            img = array_to_img(img_array)
            if label == 0:
                label_name = 'coherent'
            elif label == 1:
                label_name = 'non-coherent'

            img_filename = f"{label_name}_{j}_{i}.png"
            img_path = os.path.join(output_dir, img_filename)
            img.save(img_path)
            print(f"Saved: {img_path}")
