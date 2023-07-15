import os  # provides functions for interacting with the operating system
from glob import \
    glob  # import glob helps to filter through large datasets and pull out only files that are of interest, so this part group images from same class in files
import datetime
from tqdm import tqdm
import numpy as np  # required to read images because OpenCV uses it in the background
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import matplotlib.image as mpimg
import cv2  # required to read images
from sklearn.model_selection import train_test_split
from keras.utils import np_utils  # required to convert a class vector (integers) to binary class matrix.
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.applications.vgg16 import VGG16

os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# lOAD DATASET
dataset = pd.read_csv(
    r"C:\Users\loule\Desktop\New folder\Project B\distracted-driving-detector-main\state-farm-distracted-driver-detection\driver_imgs_list.csv")
dataset.head(5)
NUMBER_CLASSES = 10  # 10 classes


# Read image from path with opencv (Open Source Computer Vision) and return it with the right dimensions
def get_cv2_image(path, img_rows, img_cols, color_type=3):
    if color_type == 1:  # Loading image as Grayscale image Using cv2.imread() method
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    elif color_type == 3:  # Loading as color image
        img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (img_rows, img_cols))  # Compress Image based on specified size
    return img


# Loading Training dataset
def load_train(img_rows, img_cols, color_type=3):
    train_images = []  # list of training images
    train_labels = []  # list for labels
    # Loop over the training folder
    for Class in tqdm(range(NUMBER_CLASSES)):  # tqdm show progress during code run
        print('Loading directory c{}'.format(Class))
        files = glob(
            os.path.join(
                r"C:\Users\loule\Desktop\New folder\Project B\distracted-driving-detector-main\state-farm-distracted-driver-detection\imgs\train\c" + str(
                    Class), '*.jpg'))
        for file in files:  # for same class files, we take each image and read it using the get_cv2_image function
            img = get_cv2_image(file, img_rows, img_cols, color_type)
            train_images.append(img)  # add the image to the training images list
            train_labels.append(Class)  # add the label to the list
    return train_images, train_labels


# Data normalization ensures that each pixel has a similar data distribution.
# This makes convergence faster while training the network.
def read_and_normalize_train_data(img_rows, img_cols, color_type):
    # store previous function output, X is the train_images and labels is train_labels
    X, labels = load_train(img_rows, img_cols, color_type)

    # Converts a class vector (integers 0-10) to binary class matrix.
    y = np_utils.to_categorical(labels, 10)
    # split the dataset into training and testing 80:20
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    x_train = np.array(x_train, dtype=np.uint8).reshape(-1, img_rows, img_cols, color_type)
    x_test = np.array(x_test, dtype=np.uint8).reshape(-1, img_rows, img_cols, color_type)

    return x_train, x_test, y_train, y_test


# Loading validation dataset with imgs dimension of 64 X 64 colored
def load_test(size=200000, img_rows=64, img_cols=64, color_type=3):
    files = sorted(glob(os.path.join(
        r"C:\Users\loule\Desktop\New folder\Project B\distracted-driving-detector-main\state-farm-distracted-driver-detection\imgs\test",
        '*.jpg')))
    X_test, X_test_id = [], []
    total = 0
    files_size = len(files)
    for file in tqdm(files):
        if total >= size or total >= files_size:
            break
        # basename is the file name
        file_base = os.path.basename(file)
        img = get_cv2_image(file, img_rows, img_cols, color_type)
        # add the image to X_test list
        X_test.append(img)
        # add the image name to X_test_id list
        X_test_id.append(file_base)
        total += 1
    return X_test, X_test_id


# normalize validation dataset
def read_and_normalize_sampled_test_data(size, img_rows, img_cols, color_type=3):
    test_data, test_ids = load_test(size, img_rows, img_cols, color_type)
    test_data = np.array(test_data, dtype=np.uint8).reshape(-1, img_rows, img_cols, color_type)
    return test_data, test_ids


# Define Model Architecture
def create_model():
    model = Sequential()

    # CNN 1
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_rows, img_cols, color_type)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.3))

    # CNN 2
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.3))

    # CNN 3
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.5))

    # Output
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(10, activation='softmax'))

    return model


# Plot the validation accuracy and validation loss over epochs
def plot_train_history(history):
    # Summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # Summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


# Function that tests or model on test images and show the results
def plot_test_class(model, test_files, image_number, color_type=1):
    img_brute = test_files[image_number]
    img_brute = cv2.resize(img_brute, (img_rows, img_cols))
    plt.imshow(img_brute, cmap='gray')

    new_img = img_brute.reshape(-1, img_rows, img_cols, color_type)

    y_prediction = model.predict(new_img, batch_size=batch_size, verbose=1)
    print('Y prediction: {}'.format(y_prediction))
    print('Predicted: {}'.format(activity_map.get('c{}'.format(np.argmax(y_prediction)))))

    plt.show()


img_rows = 64  # dimension of images
img_cols = 64
color_type = 1  # greyscale
nb_test_samples = 200

# loading normalized train images
x_train, x_test, y_train, y_test = read_and_normalize_train_data(img_rows, img_cols, color_type)

# loading normalized validation images
test_files, test_targets = read_and_normalize_sampled_test_data(nb_test_samples, img_rows, img_cols, color_type)

activity_map = {'c0': 'Safe driving',
                'c1': 'Texting - right',
                'c2': 'Talking on the phone - right',
                'c3': 'Texting - left',
                'c4': 'Talking on the phone - left',
                'c5': 'Operating the radio',
                'c6': 'Drinking',
                'c7': 'Reaching behind',
                'c8': 'Hair and makeup',
                'c9': 'Talking to passenger'}

# plot an image for each of the 10 classes
plt.figure(figsize=(12, 20))
image_count = 1
BASE_URL = r'C:\Users\loule\Desktop\New folder\Project B\distracted-driving-detector-main\state-farm-distracted-driver-detection\imgs\train/'
for directory in os.listdir(BASE_URL):
    if directory[0] != '.':
        for i, file in enumerate(os.listdir(BASE_URL + directory)):
            if i == 1:
                break
            else:
                fig = plt.subplot(5, 2, image_count)
                image_count += 1
                image = mpimg.imread(BASE_URL + directory + '/' + file)
                plt.imshow(image)
                plt.title(activity_map[directory])

# Number of batch size and epochs
batch_size = 40
nb_epoch = 6
models_dir = "saved_models"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# ModelCheckpoint used in conjunction with training using model.fit() to save a model or weights (in a checkpoint file) at some interval,
# so the model or weights can be loaded later to continue the training from the state saved.
checkpointer = ModelCheckpoint(filepath='saved_models/weights_best_vanilla.hdf5',
                               monitor='val_loss', mode='min',
                               verbose=1, save_best_only=True)

# Stop training when a monitored metric has stopped improving
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)
model = create_model()
# More details about the layers
model.summary()

# Compiling the model with Root Mean Square Propogation algorithm
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# First Model that contains original training and testing sets and previous Network architecture  with epoch = 6 and batch_size = 40
history = model.fit(x_train, y_train,
                    validation_data=(x_test, y_test),
                    epochs=nb_epoch, batch_size=batch_size, verbose=1)
print('History of the training', history.history)
plot_train_history(history)

score1 = model.evaluate(x_test, y_test, verbose=1)

print('Loss: ', score1[0])
print('Accuracy: ', score1[1] * 100, ' %')
for i in range(10):
    plot_test_class(model, test_files, i)

# Using ImageDataGenerator from keras
train_datagen = ImageDataGenerator(rescale=1.0 / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   validation_split=0.2)

test_datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

nb_train_samples = x_train.shape[0]
nb_validation_samples = x_test.shape[0]
training_generator = train_datagen.flow(x_train, y_train, batch_size=batch_size)
validation_generator = test_datagen.flow(x_test, y_test, batch_size=batch_size)

# checkpoint = ModelCheckpoint('saved_models/weights_best_vanilla.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
history_v2 = model.fit_generator(training_generator,
                                 steps_per_epoch=nb_train_samples // batch_size,
                                 epochs=nb_epoch,
                                 verbose=1,
                                 validation_data=validation_generator,
                                 validation_steps=nb_validation_samples // batch_size)

plot_train_history(history_v2)

# Evaluate and compare the performance of the new model
score2 = model.evaluate_generator(validation_generator, nb_validation_samples // batch_size)
print("Loss for model 1", score1[0])
print("Loss for model 2 (data augmentation):", score2[0])

print("Test accuracy for model 1", score1[1])
print("Test accuracy for model 2 (data augmentation):", score2[1])


#     Architecture and adaptation of the VGG16 for our project
def vgg_std16_model(img_rows, img_cols, color_type=3):

    nb_classes = 10
    # Remove fully connected layer and replace
    vgg16_model = VGG16(weights="imagenet", include_top=False)
    for layer in vgg16_model.layers:
        layer.trainable = False

    x = vgg16_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(nb_classes, activation='softmax')(x)  # add dense layer with 10 neurons and activation softmax
    model = Model(vgg16_model.input, predictions)
    return model


# Load the VGG16 network
print("Loading network...")
model_vgg16 = vgg_std16_model(img_rows, img_cols)
model_vgg16.summary()
model_vgg16.compile(loss='categorical_crossentropy',
                    optimizer='rmsprop',
                    metrics=['accuracy'])

training_generator = train_datagen.flow_from_directory(r'C:\Users\loule\Desktop\New folder\Project B\distracted-driving-detector-main\state-farm-distracted-driver-detection\imgs\train',
                                                       target_size=(img_rows, img_cols),
                                                       batch_size=batch_size,
                                                       shuffle=True,
                                                       class_mode='categorical', subset="training")

validation_generator = test_datagen.flow_from_directory(r'C:\Users\loule\Desktop\New folder\Project B\distracted-driving-detector-main\state-farm-distracted-driver-detection\imgs\test',
                                                        target_size=(img_rows, img_cols), batch_size=batch_size,
                                                        shuffle=False, class_mode='categorical', subset="validation")
nb_train_samples = 17943
nb_validation_samples = 4481

epoch = 6
# Training the new Model
history_v3 = model_vgg16.fit_generator(training_generator, steps_per_epoch=nb_train_samples // batch_size, epochs=epoch,
                                       verbose=1, validation_data=validation_generator,
                                       validation_steps=nb_validation_samples // batch_size)

# model_vgg16.load_weights('saved_models/weights_best_vgg16.hdf5')
plot_train_history(history_v3)

# Evaluate the performance of the new model with Transfer learning
score3 = model_vgg16.evaluate_generator(validation_generator, nb_validation_samples // batch_size, verbose=1)

print("Test Score with simple CNN:", score1[0])
print("Test Accuracy with simple CNN", score1[1])
print('--------------------------------------')
print("Test Score with Data Augmentation:", score2[0])
print("Test Accuracy with Data Augmentation:", score2[1])
print('--------------------------------------')
print("Test Score with Transfer Learning:", score3[0])
print("Test Accuracy with Transfer Learning:", score3[1])
