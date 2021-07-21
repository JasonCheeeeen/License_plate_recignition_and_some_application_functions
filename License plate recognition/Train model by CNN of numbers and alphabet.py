import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D , MaxPool2D , Flatten , Dropout , Dense, MaxPooling2D
from tensorflow.keras.utils import to_categorical
import numpy as np

def CNN_mnist():
    mnist = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images.reshape((60000, 28, 28, 1))
    train_images = train_images.astype('float32') / 255
    test_images = test_images.reshape((10000, 28, 28, 1))
    test_images = test_images.astype('float32') / 255

    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='rmsprop',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    history = model.fit(train_images, train_labels, 
                        epochs=10, 
                        batch_size=512,
                        validation_split=0.1)

    model.save('mnist_model1.h5')

def CNN_alphabet():
    dataset = np.loadtxt('A_Z Handwritten Data.csv', delimiter=',')
    X = dataset[:,0:784]
    Y = dataset[:,0]
    x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.25,random_state = 100)
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float')
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float')
    x_train = x_train / 255
    x_test = x_test / 255
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    classes = y_test.shape[1]

    model2 = Sequential()
    model2.add(Conv2D(32, (5,5), input_shape = (28,28,1), activation = 'relu'))
    model2.add(MaxPool2D(pool_size=(2, 2)))
    model2.add(Dropout(0.3))
    model2.add(Flatten())
    model2.add(Dense(64, activation = 'relu'))
    model2.add(Dense(26, activation='softmax'))

    model2.compile(loss = 'categorical_crossentropy',
                   optimizer = 'adam',
                   metrics = ['acc'])

    history = model2.fit(x_train,y_train,
                        validation_data = (x_test,y_test),
                        epochs = 10,
                        batch_size = 150)
    model2.save('alpha_model.h5')

CNN_alphabet()
CNN_mnist()