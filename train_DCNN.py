import tensorflow as tf
from keras_preprocessing.image import load_img, img_to_array
from matplotlib import pyplot as plt
import os
import tensorflow as tf
import numpy as np
import os
from sklearn.model_selection import train_test_split

gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)


# Training Parameters
EPOCHS = 10
BATCH_SIZE = 32
VERBOSE = 1
OPTIMIZER = tf.keras.optimizers.Adam()
VALIDATION_SPLIT = 0.90

class LeNet(tf.keras.Model):
    def __init__(self, input_shape, classes, **kwargs):
        super(LeNet, self).__init__(**kwargs)

        self.conv1 = tf.keras.layers.Conv2D(filters=20, kernel_size=(5, 5), strides=(1, 1),
                                            padding='valid', activation='relu')
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

        self.conv2 = tf.keras.layers.Conv2D(filters=50, kernel_size=(5, 5), activation='relu')
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(units=150, activation='relu')
        self.fc2 = tf.keras.layers.Dense(units=classes, activation='softmax')

    def call(self, inputs):
        # Define forward pass using the layers defined in __init__
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return self.fc2(x)  # The output layer with softmax for predictions

def preprocess_image(img_path, label, target_size):
    try:
        img = tf.io.read_file(img_path)
        img = tf.image.decode_image(img, channels=3)
        img_array = tf.image.resize(img, target_size) / 255.0
        label = tf.cast(label, tf.int32)
        return img_array, label
    except Exception as e:
        print(f"Error processing file {img_path}: {e}")
        return None, None

def create_dataset(image_dir, target_size, batch_size, split_ratio=0.75):
    image_paths = []
    labels = []

    for img_name in os.listdir(image_dir):
        if img_name.endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(image_dir, img_name)
            image_paths.append(img_path)
            if img_name.startswith('coherent'):
                labels.append(0)  # Label for 'coherent'
            elif img_name.startswith('non-coherent'):
                labels.append(1)  # Label for 'non-coherent'

    # Split into training and test sets
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        image_paths, labels, train_size=split_ratio, random_state=42
    )

    # Create tf.data datasets for train and test
    train_dataset = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    train_dataset = train_dataset.map(
        lambda x, y: tf.numpy_function(preprocess_image, [x, y, target_size], [tf.float32, tf.int32]),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    # Enforce static shapes
    train_dataset = train_dataset.map(
    lambda x, y: (tf.ensure_shape(x, [target_size[0], target_size[1], 3]),
                  tf.ensure_shape(y, []))
    )
    train_dataset = train_dataset.shuffle(buffer_size=1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    test_dataset = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))
    test_dataset = test_dataset.map(
        lambda x, y: tf.numpy_function(preprocess_image, [x, y, target_size], [tf.float32, tf.int32]),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    # Enforce static shapes
    test_dataset = test_dataset.map(
        lambda x, y: (tf.ensure_shape(x, [target_size[0], target_size[1], 3]), 
                      tf.ensure_shape(y, []))
    )
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_dataset, test_dataset


if __name__ == '__main__':

    # Disabling GPUs with this command, not enough mem, CPU works but takes a feq hours
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disables GPU
    image_dir = os.path.join(os.getcwd(), "reconstructed_images")
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    target_size = (168, 168)

    train_dataset, test_dataset = create_dataset(image_dir, target_size, BATCH_SIZE)

    # List GPUs available
    if len(tf.config.experimental.list_physical_devices('GPU')) > 0:
        for device in tf.config.experimental.list_physical_devices('GPU'):
            print(f"Device: {device}")
    else:
        print("No GPU detected or GPU is not being utilized by TensorFlow.")

    # Input image parameters
    IMG_ROW,IMG_COL = target_size[0],target_size[1]
    IMG_CHANNELS = 3 # R,G,B
    INPUT_SHAPE = (IMG_ROW, IMG_COL, IMG_CHANNELS)
    NB_CLASSES = 2 # coherent and non-coherent

    # Build the CNN (LeNet)
    model = LeNet(input_shape=INPUT_SHAPE, classes=NB_CLASSES)

    inputs = tf.keras.Input(shape=INPUT_SHAPE)
    outputs = model.call(inputs)
    model = tf.keras.Model(inputs, outputs)

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer=OPTIMIZER,
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )
    model.summary()

    # Fit the model
    history = model.fit(train_dataset,
                        epochs=EPOCHS,
                        verbose=VERBOSE,
                        validation_data=test_dataset)
    score = model.evaluate(test_dataset,
                           verbose=VERBOSE)
    print(rf"Test Score {score[0]}")
    print(rf"Test Accuracy {score[1]}")

    plt.plot(history.history['accuracy'], label='train accuracy')
    plt.plot(history.history['val_accuracy'], label='val accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Plot training & validation loss
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
