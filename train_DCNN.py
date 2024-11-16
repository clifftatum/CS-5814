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
EPOCHS = 100
BATCH_SIZE = 50
VERBOSE = 1
OPTIMIZER = tf.keras.optimizers.Adam()
VALIDATION_SPLIT = 0.90

class LeNet(tf.keras.Model):
    def __init__(self, input_shape, classes, **kwargs):
        super(LeNet, self).__init__(**kwargs)

        self.conv1 = tf.keras.layers.Conv2D(filters=20, kernel_size=(5, 5), strides=(1, 1),
                                            padding='valid', activation='relu', input_shape=input_shape)
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

        self.conv2 = tf.keras.layers.Conv2D(filters=50, kernel_size=(5, 5), activation='relu')
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(units=500, activation='relu')
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




if __name__ == '__main__':

    # Disabling GPUs with this command, not enough mem, CPU works but takes a feq hours
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disables GPU
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    # List GPUs available
    if len(tf.config.experimental.list_physical_devices('GPU')) > 0:
        for device in tf.config.experimental.list_physical_devices('GPU'):
            print(f"Device: {device}")
    else:
        print("No GPU detected or GPU is not being utilized by TensorFlow.")

    import os




    image_dir = os.path.join(os.getcwd(), "training_images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    target_size = (168, 168)
    images = []
    labels = []
    len_dir = 62232


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


    input_data = np.array(images)
    input_labels = np.array(labels)
    del images
    del labels
    import gc
    gc.collect()


    # Input image parameters
    IMG_ROW,IMG_COL = target_size[0],target_size[1]
    IMG_CHANNELS = 3 # R,G,B
    INPUT_SHAPE = (IMG_ROW,IMG_COL,IMG_CHANNELS)
    NB_CLASSES = 2 # coherent and non-coherent

    # Split the data by train,test~ 85%
    x_train, x_test, y_train, y_test = train_test_split(input_data,input_labels,
                                                        test_size=0.15,
                                                        shuffle=True,
                                                        random_state=42)

    # One hot encode the labels
    y_train = tf.keras.utils.to_categorical(y_train, NB_CLASSES)
    y_test = tf.keras.utils.to_categorical(y_test, NB_CLASSES)

    # Build the CNN (LeNet)
    model = LeNet(input_shape=INPUT_SHAPE,classes=NB_CLASSES)
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=[tf.keras.metrics.Accuracy()]
    )
    model.build((None, *INPUT_SHAPE))
    model.summary()

    # Fit the model

    history = model.fit(x_train, y_train,
                        batch_size=BATCH_SIZE,
                        epochs = EPOCHS,
                        verbose=VERBOSE,
                        validation_split=VALIDATION_SPLIT)
    score = model.evaluate(x = x_test,
                           y = y_test,verbose=VERBOSE)
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




































