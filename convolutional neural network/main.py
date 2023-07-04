import pandas as pd
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from classifier import CatClassifier
from draw import draw_graphs, draw_confusion_matrix, plot_predictions

import math
from sklearn.metrics import confusion_matrix


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, '..', 'data')

    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    test_dir = os.path.join(data_dir, 'test')

    train_labels = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    class_labels = train_labels['label'].unique()
    num_classes = len(class_labels)

    input_shape = (224, 224, 3)

    batch_size = 64
    epochs = 20

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='categorical'
    )

    valid_generator = test_datagen.flow_from_directory(
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

    model = CatClassifier(num_classes, input_shape)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

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
