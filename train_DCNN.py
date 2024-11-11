import tensorflow as tf
from keras_preprocessing.image import load_img, img_to_array
from matplotlib import pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import numpy as np
# from tensorflow.keras.preprocessing.image import img_to_array, load_img
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

    # (x_train,y_train),(x_test,y_test) = datasets.mnist.load_data()
    # y_train = tf.keras.utils.to_categorical(y_train,10)
    # plt.imshow(x_train[0])
    # plt.title(f"Label: {y_train[0][0]}")
    # plt.show()
    # Check if TensorFlow can see GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disables GPU
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    # List GPUs available
    if len(tf.config.experimental.list_physical_devices('GPU')) > 0:
        for device in tf.config.experimental.list_physical_devices('GPU'):
            print(f"Device: {device}")
    else:
        print("No GPU detected or GPU is not being utilized by TensorFlow.")

    import os

    # dataset_dir = r'C:\Users\cft5385\Documents\Learning\GradSchool\Repos\CS-5814\training_images'
    # dataset = []
    # class_labels = []
    # dir = os.listdir(dataset_dir)
    # # Loop through your images
    # for i in np.arange(1,3):  # Assuming each class has its own directory
    #     path = os.path.join(dataset_dir,rf'caf_dataset_{i}_of_20.npy')
    #     data = np.load(path,allow_pickle=True)
    #     images = data.item().get('images')
    #     labels = data.item().get('labels')
    #     dataset.append(images)
    #     class_labels.append(labels)
    #     print(path)
    #
    # input_data = np.vstack(dataset)
    # input_labels = np.hstack(class_labels)

    # Define your image directory and target size
    image_dir = r'C:\Users\cft5385\Documents\Learning\GradSchool\Repos\CS-5814\training_images'
    target_size = (168, 168)  # You may use another size, just be consistent across your data
    images = []
    labels = []  # Use integers or class names as labels
    len_dir =62232

    # Loop through your images
    for dir in os.listdir(image_dir):  # Assuming each class has its own directory
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
    # train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    # train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
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




































