import pandas as pd
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD

from MLP import MLP
from draw import draw_graphs, draw_confusion_matrix, plot_predictions

import math
from sklearn.metrics import confusion_matrix
import random


def train_preprocess(image):
    # Resize
    image = tf.image.resize(image, [224, 224])

    # Random horizontal flip
    image = tf.image.random_flip_left_right(image)

    # Random rotation
    image = tf.image.rot90(image, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))

    # Normalize
    mean = [0.4851, 0.4405, 0.3614]
    sd = [0.2213, 0.2092, 0.2036]
    image = (image - mean) / sd

    return image


def valid_preprocess(image):
    # Resize
    image = tf.image.resize(image, [224, 224])

    # Normalize
    mean = [0.5009, 0.4567, 0.3786]
    sd = [0.2143, 0.2058, 0.2002]
    image = (image - mean) / sd

    return image


def test_preprocess(image):
    # Resize
    image = tf.image.resize(image, [224, 224])

    # Normalize
    mean = [0.5002, 0.4565, 0.3743]
    sd = [0.1999, 0.1948, 0.1928]
    image = (image - mean) / sd

    return image


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, '..', 'data')

    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    test_dir = os.path.join(data_dir, 'test')

    train_labels = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    class_labels = train_labels['label'].unique()
    num_classes = len(class_labels)

    input_shape = (224, 224)

    batch_size = 64
    epochs = 150

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        preprocessing_function=train_preprocess,
        horizontal_flip=True
    )

    valid_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        preprocessing_function=valid_preprocess,
    )

    test_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        preprocessing_function=test_preprocess,
    )

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='categorical'
    )

    valid_generator = valid_datagen.flow_from_directory(
        valid_dir,
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='categorical'
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    input_size = 224 * 224 * 3  # Input size for the MLP model
    hidden_sizes = [256, 256, 256, 256]
    output_size = num_classes
    input_shape = (224, 224, 3)
    model = MLP(input_shape, hidden_sizes, output_size)
    lr_range = (0.001, 0.1)
    weight_decay_range = (0.0001, 0.01)
    lr = random.uniform(*lr_range)
    print('lr ', lr)
    weight_decay = random.uniform(*weight_decay_range)
    print('wd ', weight_decay)
    # lr = 0.005
    momentum = 0.8
    # weight_decay = 0.0005

    optimizer = SGD(learning_rate=lr, momentum=momentum, weight_decay=weight_decay)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=epochs,
        validation_data=valid_generator,
        validation_steps=valid_generator.samples // batch_size
    )

    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')

    draw_graphs(history)

    test_predictions = model.predict(test_generator)
    test_pred_labels = np.argmax(test_predictions, axis=1)
    draw_confusion_matrix(test_generator.labels, test_pred_labels, class_labels)

    print("Generating predictions for random images...")

    subset_indices = np.random.choice(len(test_generator.filenames), size=30, replace=False)
    subset_images = []
    subset_true_labels = []
    subset_pred_labels = []

    for i in subset_indices:
        image_path = os.path.join(test_dir, test_generator.filenames[i])
        image = tf.keras.preprocessing.image.load_img(image_path, target_size=input_shape[:2])
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = image / 255.0  # Normalize image
        subset_images.append(image)

        true_label = test_generator.labels[i]
        pred = model.predict(np.expand_dims(image, axis=0))
        pred_label = np.argmax(pred)

        subset_true_labels.append(true_label)
        subset_pred_labels.append(pred_label)

    plot_predictions(subset_images, subset_true_labels, subset_pred_labels, class_labels)


if __name__ == '__main__':
    main()
