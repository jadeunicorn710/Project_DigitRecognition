import os
import cv2
import h5py
import numpy as np
from scipy.io import loadmat

import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Input, Flatten, Dense, Conv2D, MaxPool2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model, Sequential, model_from_json
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import callbacks
import json
import matplotlib.pyplot as plt


# ======================================================================================================================
# VGG16 Model without pretrained weights
# ======================================================================================================================
class vgg16ModelFromScratch:
    # initialize model and history variables
    model = None
    history = None
    # initialize variables to store the accuracy and loss plots
    accuracy_plot = None
    loss_plot = None

    # initialize the model
    def initialize_model(self):
        # define the accuracy and loss plots to be saved
        self.accuracy_plot = 'vgg16_scratch_accuracy.png'
        self.loss_plot = 'vgg16_scratch_loss.png'

        # load the vgg16 framework
        # change include_top to False to specify the input_shape
        # vgg16 model without pretrained weights
        vgg16 = VGG16(weights=None, include_top=False, input_shape=(32, 32, 3))

        # define the input layer
        input_layer = Input(shape=(32, 32, 3), name='image_input')
        vgg16_output = vgg16(input_layer)

        # define intermediate layers
        flatten_layer = Flatten(name='flatten')(vgg16_output)
        fc1 = Dense(4096, activation='relu', name='fc1')(flatten_layer)
        fc2 = Dense(4096, activation='relu', name='fc2')(fc1)

        # define the output layer, changed default number of classes from 1000 to 11
        output = Dense(11, activation='softmax', name='predictions')(fc2)

        # build and compile the model
        self.model = Model(input_layer, output)
        # self.model.compile(optimizer=SGD(learning_rate=0.0001, momentum=0.1, nesterov=False, name="SGD"),
        #                    loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.compile(optimizer=Adam(learning_rate=0.0001, amsgrad=True),
                           loss='categorical_crossentropy', metrics=['accuracy'])


    # train the model
    def train_model(self, X_train, y_train):
        # change label class to categorical
        y_train = to_categorical(y_train)
        # X_train = np.array(X_train)

        # convert the grayscale image to three channels for input
        X_train_converted = np.concatenate((X_train, X_train, X_train), axis=3)

        # define early stopping callback
        es_callback = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='min')

        # Train the model with the new callback
        self.history = self.model.fit(X_train_converted, y_train,
                                      shuffle=True,  # shuffle train data before each epoch
                                      validation_split=0.2, # fraction of training data to be used as validation data
                                      epochs=50,  # define epochs
                                      batch_size=128, # define batch size
                                      callbacks=[es_callback])  # Pass callbacks to training


    # evaluate the model
    def evaluate_model(self, X_test, y_test):
        # change label class to categorical
        y_test = to_categorical(y_test)
        X_test = np.array(X_test)

        # convert the grayscale image to three channels for input
        X_test_converted = np.concatenate((X_test, X_test, X_test), axis=3)

        loss, acc = self.model.evaluate(X_test_converted, y_test, verbose = 2)
        print("classification accuracy: {:0.2f}%".format(100 * acc))
        print("classification loss: {:0.4f}".format(loss))


    # save checkpoints during training
    def save_checkpoints(self, model_path, weights_path):
        # save weights
        self.model.save_weights(weights_path)
        # save model
        model_json = self.model.to_json()
        with open(model_path, "w+") as json_file:
            json_file.write(model_json)

    # plot accuracy and loss
    def plot_metrics(self):
        # plot the training curve for accuracy
        plt.plot(self.history.history['acc'])
        plt.plot(self.history.history['val_acc'])
        plt.title('vgg16-scratch accuracy plot')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend(['train', 'validation'], loc = 'upper left')
        plt.savefig(self.accuracy_plot)
        plt.clf()

        # plot the training curve for loss
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('vgg16-scratch loss plot')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc = 'upper left')
        plt.savefig(self.loss_plot)
        plt.clf()


    # define digit prediction function
    @staticmethod
    def digit_prediction(model_path, weights_path, image):
        # convert the grayscale image to three channels for input
        image_converted = np.concatenate((image, image, image), axis=3)
        # read in saved model json file
        with open(model_path, 'r') as json_file:
            model_json = json.load(json_file)
            # convert json file to model instance
            model_loaded = model_from_json(json.dumps(model_json))
            # load the saved weights into the model instance
            model_loaded.load_weights(weights_path)
            image_converted = np.array(image_converted)
            # predict digit in the input image
            label_predicted = model_loaded.predict(np.array(image_converted.reshape((image_converted.shape[0], image_converted.shape[1], image_converted.shape[2], 3))))
            return label_predicted


# ======================================================================================================================
# VGG16 Model with pretrained weights
# ======================================================================================================================
class vgg16ModelPretrained:
    # initialize model and history variables
    model = None
    history = None
    # initialize variables to store the accuracy and loss plots
    accuracy_plot = None
    loss_plot = None

    # initialize the model
    def initialize_model(self):
        # define the accuracy and loss plots to be saved
        self.accuracy_plot = 'vgg16_pretrained_accuracy.png'
        self.loss_plot = 'vgg16_pretrained_loss.png'

        # load the vgg16 framework
        # change include_top to False to specify the input_shape
        # vgg16 model with pretrained weights
        vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

        # define the input layer
        input_layer = Input(shape=(32, 32, 3), name='image_input')
        vgg16_output = vgg16(input_layer)

        # define intermediate layers
        flatten_layer = Flatten(name='flatten')(vgg16_output)
        fc1 = Dense(4096, activation='relu', name='fc1')(flatten_layer)
        fc2 = Dense(4096, activation='relu', name='fc2')(fc1)

        # define the output layer, changed default number of classes from 1000 to 11
        output = Dense(11, activation='softmax', name='predictions')(fc2)

        # build and compile the model
        self.model = Model(input_layer, output)
        # self.model.compile(optimizer=SGD(learning_rate=0.0001, momentum=0.1, nesterov=False, name="SGD"),
        #                    loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.compile(optimizer=Adam(learning_rate=0.0001, amsgrad=True),
                           loss='categorical_crossentropy', metrics=['accuracy'])


    # train the model
    def train_model(self, X_train, y_train):
        # change label class to categorical
        y_train = to_categorical(y_train)
        # X_train = np.array(X_train)

        # convert the grayscale image to three channels for input
        X_train_converted = np.concatenate((X_train, X_train, X_train), axis=3)

        # define early stopping callback
        es_callback = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='min')

        # Train the model with the new callback
        self.history = self.model.fit(X_train_converted, y_train,
                                      shuffle=True,  # shuffle train data before each epoch
                                      validation_split=0.2, # fraction of training data to be used as validation data
                                      epochs=50,  # define epochs
                                      batch_size=128, # define batch size
                                      callbacks=[es_callback])  # Pass callbacks to training


    # evaluate the model
    def evaluate_model(self, X_test, y_test):
        # change label class to categorical
        y_test = to_categorical(y_test)
        X_test = np.array(X_test)

        # convert the grayscale image to three channels for input
        X_test_converted = np.concatenate((X_test, X_test, X_test), axis=3)

        loss, acc = self.model.evaluate(X_test_converted, y_test, verbose = 2)
        print("classification accuracy: {:0.2f}%".format(100 * acc))
        print("classification loss: {:0.4f}".format(loss))


    # save checkpoints during training
    def save_checkpoints(self, model_path, weights_path):
        # save weights
        self.model.save_weights(weights_path)
        # save model
        model_json = self.model.to_json()
        with open(model_path, "w+") as json_file:
            json_file.write(model_json)

    # plot accuracy and loss
    def plot_metrics(self):
        # plot the training curve for accuracy
        plt.plot(self.history.history['acc'])
        plt.plot(self.history.history['val_acc'])
        plt.title('vgg16-pretrained accuracy plot')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend(['train', 'validation'], loc = 'upper left')
        plt.savefig(self.accuracy_plot)
        plt.clf()

        # plot the training curve for loss
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('vgg16-pretrained loss plot')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc = 'upper left')
        plt.savefig(self.loss_plot)
        plt.clf()


    # define digit prediction function
    @staticmethod
    def digit_prediction(model_path, weights_path, image):
        # convert the grayscale image to three channels for input
        image_converted = np.concatenate((image, image, image), axis=3)
        # read in saved model json file
        with open(model_path, 'r') as json_file:
            model_json = json.load(json_file)
            # convert json file to model instance
            model_loaded = model_from_json(json.dumps(model_json))
            # load the saved weights into the model instance
            model_loaded.load_weights(weights_path)
            image_converted = np.array(image_converted)
            # predict digit in the input image
            label_predicted = model_loaded.predict(np.array(image_converted.reshape((image_converted.shape[0], image_converted.shape[1], image_converted.shape[2], 3))))
            return label_predicted


# ======================================================================================================================
# Self-developed CNN model
# ======================================================================================================================
class designedCNNModel:
    # initialize model and history variables
    model = None
    history = None
    # initialize variables to store the accuracy and loss plots
    accuracy_plot = None
    loss_plot = None

    # initialize the model
    def initialize_model(self):
        # define the accuracy and loss plots to be saved
        self.accuracy_plot = 'designed_cnn_accuracy.png'
        self.loss_plot = 'designed_cnn_loss.png'

        self.model = Sequential()

        # add layers into the neural network

        # add the input layer with Relu activation
        self.model.add(Conv2D(32, 3, activation = 'relu', padding = 'valid', input_shape = (32, 32, 1)))

        # add the first dropout layer to avoid overfitting
        # self.model.add(Dropout(0.2))
        # add one batch normalization layer
        self.model.add(BatchNormalization())
        # add the first 2D max pooling layer
        self.model.add(MaxPool2D(pool_size = (2, 2), strides = 2, padding = 'valid'))

        # add the second 2D convolutional layer with Relu activation
        self.model.add(Conv2D(64, 3, activation = 'relu', padding = 'same'))
        # add the second dropout layer to avoid overfitting
        self.model.add(Dropout(0.2))
        # add the second 2D max pooling layer
        self.model.add(MaxPool2D(pool_size = (2, 2), strides = 2, padding = 'same'))

        # add the third 2D convolutional layer with Relu activation
        self.model.add(Conv2D(128, 5, activation = 'relu', padding = 'same'))
        # add the third dropout layer to avoid overfitting
        self.model.add(Dropout(0.2))
        # add the third 2D max pooling layer
        self.model.add(MaxPool2D(pool_size = (3, 3), strides = 2, padding = 'same'))

        # add the fourth 2D convolutional layer with Relu activation
        self.model.add(Conv2D(256, 5, activation = 'relu', padding = 'same'))
        # add the fourth dropout layer to avoid overfitting
        self.model.add(Dropout(0.2))
        # add the fourth 2D max pooling layer
        self.model.add(MaxPool2D(pool_size = (3, 3), strides = 2, padding = 'same'))


        # add the fifth 2D convolutional layer with Relu activation
        self.model.add(Conv2D(512, 2, activation = 'relu', padding = 'valid'))
        # add the fifth dropout layer to avoid overfitting
        self.model.add(Dropout(0.2))
        # add the fifth 2D max pooling layer
        self.model.add(MaxPool2D(pool_size = (3, 3), strides = 2, padding = 'same'))

        # add the flatten layer
        self.model.add(Flatten())

        # add the first fully connected layer
        self.model.add(Dense(256, kernel_initializer = 'uniform'))
        # add the second fully connected layer
        self.model.add(Dense(128))

        # add the output layer with softmax activation
        self.model.add(Dense(11, activation = 'softmax'))

        # build and compile the model
        self.model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9, nesterov=False, name="SGD"),
                           loss='categorical_crossentropy', metrics=['accuracy'])

    # train the model
    def train_model(self, X_train, y_train):
        # change label class to categorical
        y_train = to_categorical(y_train)
        # X_train = np.array(X_train)

        # define early stopping callback
        es_callback = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='min')

        # Train the model with the new callback
        self.history = self.model.fit(X_train, y_train,
                                      shuffle=True,  # shuffle train data before each epoch
                                      validation_split=0.2, # fraction of training data to be used as validation data
                                      epochs=50,  # define epochs
                                      batch_size=128, # define batch size
                                      callbacks=[es_callback])  # Pass callbacks to training


    # evaluate the model
    def evaluate_model(self, X_test, y_test):
        # change label class to categorical
        y_test = to_categorical(y_test)
        X_test = np.array(X_test)

        loss, acc = self.model.evaluate(X_test, y_test, verbose = 1)
        print("classification accuracy: {:0.2f}%".format(100 * acc))
        print("classification loss: {:0.4f}".format(loss))


    # save checkpoints during training
    def save_checkpoints(self, model_path, weights_path):
        # save weights
        self.model.save_weights(weights_path)
        # save model
        model_json = self.model.to_json()
        with open(model_path, "w+") as json_file:
            json_file.write(model_json)

    # plot accuracy and loss
    def plot_metrics(self):
        # plot the training curve for accuracy
        plt.plot(self.history.history['acc'])
        plt.plot(self.history.history['val_acc'])
        plt.title('designed-cnn accuracy plot')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend(['train', 'validation'], loc = 'upper left')
        plt.savefig(self.accuracy_plot)
        plt.clf()

        # plot the training curve for loss
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('designed-cnn loss plot')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc = 'upper left')
        plt.savefig(self.loss_plot)
        plt.clf()

    # define digit prediction function
    @staticmethod
    def digit_prediction(model_path, weights_path, image):
        # read in saved model json file
        with open(model_path, 'r') as json_file:
            model_json = json.load(json_file)
            # convert json file to model instance
            model_loaded = model_from_json(json.dumps(model_json))
            # load the saved weights into the model instance
            model_loaded.load_weights(weights_path)
            image = np.array(image)
            # predict digit in the input image
            label_predicted = model_loaded.predict(np.array(image.reshape((image.shape[0], image.shape[1], image.shape[2], 1))))
            return label_predicted


# ======================================================================================================================
# train & evaluate models using the preprocessed format 2 dataset
# ======================================================================================================================
def trainEvaluateModel(model_used, model_path, weights_path):

    # load the original format 2 datasets in grayscale
    # X_train, y_train, X_test, y_test = grayscale_image()
    # X_train, y_train, X_test, y_test = label_processing()

    # load the saved combined train dataset in grayscale
    X_train = np.load('./svhnData/Format2/X_train_combined.npy')
    y_train = np.load('./svhnData/Format2/y_train_combined.npy')
    print("train set size")
    print(X_train.shape, y_train.shape)

    # # load the saved test dataset in grayscale
    X_test = np.load('./svhnData/Format2/X_test_combined.npy')
    y_test = np.load('./svhnData/Format2/y_test_combined.npy')
    print("test set size")
    print(X_test.shape, y_test.shape)

    model = model_used

    # train the model
    print("------------initializing model-----------------")
    model.initialize_model()
    print("------------training model-----------------")
    model.train_model(X_train, y_train)
    print("------------plotting metrics-----------------")
    model.plot_metrics()
    print("------------saving checkpoints-----------------")
    model.save_checkpoints(model_path, weights_path)
    print("------------evaluating model-----------------")
    model.evaluate_model(X_test, y_test)

# ======================================================================================================================
# train & evaluate various models and save checkpoints (already done, no need to rerun)
# ======================================================================================================================
# trainEvaluateModel(vgg16ModelFromScratch(), "vgg16_scratch_model.json", "vgg16_scratch_weights.h5")
# trainEvaluateModel(vgg16ModelPretrained(), "vgg16_pretrained_model.json", "vgg16_pretrained_weights.h5")
# trainEvaluateModel(designedCNNModel(), "designed_cnn_model.json", "designed_cnn_weights.h5")
