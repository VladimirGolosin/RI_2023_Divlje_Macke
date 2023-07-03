import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import math

def draw_graphs(num_epochs, train_losses, valid_losses, accuracies):

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), valid_losses, label='Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), accuracies)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()

def draw_confusion_matrix(true_labels, pred_labels, class_labels):

    confusion_mat = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted Type')
    plt.ylabel('True Type')
    plt.show()


def plot_predictions(images, true_labels, pred_labels, class_labels):
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

            img = images[index].cpu().numpy().transpose((1, 2, 0))
            img = img * 0.5 + 0.5  # Denormalize image

            axs[i, j].imshow(img)
            axs[i, j].axis('off')

            true_label = class_labels[true_labels[index]]
            pred_label = class_labels[pred_labels[index]]

            color = 'green' if true_label == pred_label else 'red'

            axs[i, j].set_title(f'True: {true_label}\nPred: {pred_label}', color=color)

    plt.show()