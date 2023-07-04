import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
from sklearn.metrics import confusion_matrix


def draw_graphs(history, export_dir=None):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(history.history['loss']) + 1), history.history['loss'], label='Train Loss')
    plt.plot(range(1, len(history.history['val_loss']) + 1), history.history['val_loss'], label='Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    if export_dir:
        plt.savefig(f'{export_dir}/loss.png')

    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(history.history['accuracy']) + 1), history.history['accuracy'], label='Train Accuracy')
    plt.plot(range(1, len(history.history['val_accuracy']) + 1), history.history['val_accuracy'], label='Valid Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    if export_dir:
        plt.savefig(f'{export_dir}/accuracy.png')

    plt.show()


def draw_confusion_matrix(true_labels, pred_labels, class_labels, export_dir=None):
    confusion_mat = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted Type')
    plt.ylabel('True Type')
    plt.xticks(rotation=30)
    plt.yticks(rotation=0)

    # Add explicit labels
    ax = plt.gca()
    ax.xaxis.set_label_coords(0.5, -0.1)
    ax.yaxis.set_label_coords(-0.1, 0.5)
    plt.xlabel('Predicted Type', fontsize=12, fontweight='bold')
    plt.ylabel('True Type', fontsize=12, fontweight='bold')

    if export_dir:
        plt.savefig(f'{export_dir}/confusion.png')

    plt.show()


def plot_predictions(images, true_labels, pred_labels, class_labels, export_dir=None):
    num_images = len(images)
    num_rows = math.ceil(num_images / 5)
    num_cols = min(num_images, 5)

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(16, 12))
    fig.subplots_adjust(hspace=0.5)

    for i in range(num_rows):
        for j in range(num_cols):
            index = i * num_cols + j
            if index >= num_images:
                break

            img = images[index]

            axs[i, j].imshow(img)
            axs[i, j].axis('off')

            true_label = class_labels[true_labels[index]]
            pred_label = class_labels[pred_labels[index]]

            color = 'green' if true_label == pred_label else 'red'

            axs[i, j].set_title(f'True: {true_label}\nPred: {pred_label}', color=color)

    if export_dir:
        plt.savefig(f'{export_dir}/predictions.png')

    plt.show()
