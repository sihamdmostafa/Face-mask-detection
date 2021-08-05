import cv2 
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from pathlib import Path


def save_model(json_filename, weights_filename, model):

    model_json = model.to_json()
    with open(json_filename, 'w') as json_file:
        json_file.write(model_json)

    model.save_weights(weights_filename)

def load_png(specifier):

    dirpath = 'data/' + specifier
    p = Path(dirpath)
    dirs = p.glob('*')
    labels = []
    data = []
    label_dict = {'mask' : 0, 'no-mask': 1}
    for folder in dirs:
        label = str(folder).split('/')[-1]
        print(label)
        for image_path in folder.glob('*.png'):
            img = image.load_img(image_path, target_size = (224, 224))
            data.append(image.img_to_array(img))
            labels.append(label_dict[label])
    image_train_png = np.array(data)
    label_train_png = np.array(labels)
    return image_train_png, label_train_png

def load_jpg(specifier):

    dirpath = 'data/' + specifier
    p = Path(dirpath)
    dirs = p.glob('*')
    labels = []
    data = []
    label_dict = {'mask' : 0, 'no-mask': 1}
    for folder in dirs:
        label = str(folder).split('/')[-1]
        print(label)
        for image_path in folder.glob('*.jpg'):
            img = image.load_img(image_path, target_size = (224, 224))
            data.append(image.img_to_array(img))
            labels.append(label_dict[label])
    image_train_jpg = np.array(data)
    label_train_jpg = np.array(labels)
    return image_train_jpg, label_train_jpg

def load_jpeg(specifier):

    dirpath = 'data/' + specifier
    p = Path(dirpath)
    dirs = p.glob('*')
    labels = []
    data = []
    label_dict = {'mask' : 0, 'no-mask': 1}
    for folder in dirs:
        label = str(folder).split('/')[-1]
        print(label)
        for image_path in folder.glob('*.jpeg'):
            img = image.load_img(image_path, target_size = (224, 224))
            data.append(image.img_to_array(img))
            labels.append(label_dict[label])
    image_train_jpeg = np.array(data)
    label_train_jpeg = np.array(labels)
    return image_train_jpeg, label_train_jpeg


def load_ds_train():

    png_images, png_labels = load_png('train')
    jpg_images, jpg_labels = load_jpg('train')
    jpeg_images, jpeg_labels = load_jpeg('train')
    train_images = np.concatenate((png_images, jpg_images, jpeg_images))
    train_labels = np.concatenate((png_labels, jpg_labels, jpeg_labels))
    return train_images, train_labels

def load_ds_valid():

    png_images, png_labels = load_png('valid')
    jpg_images, jpg_labels = load_jpg('valid')
    jpeg_images, jpeg_labels = load_jpeg('valid')
    valid_images = np.concatenate((png_images, jpg_images, jpeg_images))
    valid_labels = np.concatenate((png_labels, jpg_labels, jpeg_labels))

    return valid_images, valid_labels

def load_ds_test():

    png_images, png_labels = load_png('test')
    jpg_images, jpg_labels = load_jpg('test')
    jpeg_images, jpeg_labels = load_jpeg('test')
    test_images = np.concatenate((png_images, jpg_images, jpeg_images))
    test_labels = np.concatenate((png_labels, jpg_labels, jpeg_labels))

    return test_images, test_labels

def preprocess_images(train_images, valid_images, test_images):

    size = train_images.shape[1]
    train_images = np.reshape(train_images, [-1, size, size, 3])
    valid_images = np.reshape(valid_images, [-1, size, size, 3])
    test_images = np.reshape(test_images, [-1, size, size, 3])

    # Normalize image
    train_image_mean = np.mean(train_images, axis = 0)
    train_images -= train_image_mean / 10
    valid_images -= train_image_mean / 10
    test_images -= train_image_mean / 10

    return train_images, valid_images, test_images

def preprocess_labels(train_labels, valid_labels, test_labels):

    num_labels = 2
    train_labels = to_categorical(train_labels)
    valid_labels = to_categorical(valid_labels)
    test_labels = to_categorical(test_labels)

    return train_labels, valid_labels, test_labels


def plot_metrics(history):

    history_frame = pd.DataFrame(history.history)
    history_frame.loc[:, ['loss', 'val_loss']].plot()
    history_frame.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot()

def train(model):

    train_images, train_labels = load_ds_train()
    valid_images, valid_labels = load_ds_valid()
    test_images, test_labels = load_ds_test() 

    print('Train Data', train_images.shape, train_labels.shape)
    print('Validation Data', valid_images.shape, valid_labels.shape)
    print('Test Data', test_images.shape, test_labels.shape)

    train_images, valid_images, test_images = preprocess_images(train_images, valid_images, test_images)
    train_labels, valid_labels, test_labels = preprocess_labels(train_labels, valid_labels, test_labels)


    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001, decay = 6e-5)
    model.compile(
            optimizer = optimizer,
            loss = 'binary_crossentropy',
            metrics = ['binary_accuracy'],
            )

    history = model.fit(
            train_images, 
            train_labels,
            batch_size = 10,
            epochs = 50, 
            validation_data = (valid_images, valid_labels))

    plot_metrics(history)
    

def make_model():

    pretrained_base = tf.keras.applications.ResNet50(input_shape = (224, 224, 3)) # Need to check input_shape
    pretrained_base.trainable = False

    model = keras.Sequential([
        pretrained_base,
        layers.Flatten(),
        layers.Dense(16, activation = 'relu'),
        layers.Dense(16, activation = 'relu'),
        layers.Dense(2, activation = 'softmax'),
        ])
    return model



model = make_model()
train(model)
save_model('mask_detection.json', 'mask_detection.h5', model)
