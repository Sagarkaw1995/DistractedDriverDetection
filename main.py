# Importing Necessary Libraries
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

# Data Preprocessing
data = pd.read_csv('/Volumes/Samsung/Datasets/state-farm-distracted-driver-detection/driver_imgs_list.csv')
img = cv2.imread('/Volumes/Samsung/Datasets/state-farm-distracted-driver-detection/imgs/train/c0/img_327.jpg')


def get_im_cv2(path, img_rows, img_cols):
    img = cv2.imread(path, 0)
    resized = cv2.resize(img, (img_cols, img_rows))
    return resized


X = data[['img']]
y = data[['classname']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

image_train_arr = []
for i in range(len(X_train)):
    path = "/Volumes/Samsung/Datasets/state-farm-distracted-driver-detection/imgs/train/{}/{}".format(
        y_train.iloc[i, 0],
        X_train.iloc[i, 0])
    resized = get_im_cv2(path, 32, 32)
    image_train_arr.append(resized)

image_test_arr = []
for i in range(len(X_test)):
    path = "/Volumes/Samsung/Datasets/state-farm-distracted-driver-detection/imgs/train/{}/{}".format(y_test.iloc[i, 0],
                                                                                                      X_test.iloc[i, 0])
    resized = get_im_cv2(path, 32, 32)
    image_test_arr.append(resized)

image_train_arr = np.array(image_train_arr)
image_test_arr = np.array(image_test_arr)

image_train_arr = image_train_arr / 255
image_test_arr = image_test_arr / 255

# Converting Classname from c0, c1 ... c9 to 0, 1 .... 9
labelencoder = LabelEncoder()
y_train['new-col'] = labelencoder.fit_transform(y_train)
y_test['new-col'] = labelencoder.fit_transform(y_test)

# CNN Layers
model = models.Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)))
model.add(BatchNormalization())
model.add(MaxPool2D(2, 2))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(2, 2))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(10))
model.summary()

# Training Model
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
history = model.fit(image_train_arr, y_train['new-col'], epochs=30, batch_size=100,
                    validation_data=(image_test_arr, y_test['new-col']))
print("\nModel evaluation after 30 epochs: ", model.evaluate(image_test_arr, y_test['new-col'], verbose=2))

#Saving Model
models.save_model(model, 'model/ddd.hdf5')

# Plotting Loss and Accuracy
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='lower right')
plt.show()

# Model Prediction
test_data = pd.read_csv('/Volumes/Samsung/Datasets/state-farm-distracted-driver-detection/sample_submission.csv')
test_arr = []
for i in range(len(test_data)):
    path = "/Volumes/Samsung/Datasets/state-farm-distracted-driver-detection/imgs/test/{}".format(test_data.iloc[i, 0])
    resized = get_im_cv2(path, 32, 32)
    test_arr.append(resized)
test_arr = np.array(test_arr)
test_arr_norm = test_arr / 255
test_output_arr = model.predict(test_arr_norm)

Classes = {'c0': 'Safe driving',
           'c1': 'Texting - right',
           'c2': 'Talking on the phone - right',
           'c3': 'Texting - left',
           'c4': 'Talking on the phone - left',
           'c5': 'Operating the radio',
           'c6': 'Drinking',
           'c7': 'Reaching behind',
           'c8': 'Hair and makeup',
           'c9': 'Talking to passenger'}

test_output_classlist = []

for i in range(len(test_output_arr)):
    test_output_classlist.append(Classes["c{}".format(np.argmax(test_output_arr[i]))])

test_output_class = pd.DataFrame(test_output_classlist, columns=['Class'])
test_data_variable = pd.DataFrame(test_data.iloc[:, 0])
result = pd.concat([test_data_variable, test_output_class], axis=1)
result.to_csv(r'result.csv', index=False)
