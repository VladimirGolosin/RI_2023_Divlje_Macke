import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import pandas as pd
import os
from classifier import CatClassifier
from draw import *
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, '..', 'data')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
valid_dataset = ImageFolder(os.path.join(data_dir, 'valid'), transform=transform)
test_dataset = ImageFolder(os.path.join(data_dir, 'test'), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

train_labels = pd.read_csv(os.path.join(data_dir, 'train.csv'))
class_labels = train_labels['label'].unique()
num_classes = len(class_labels)

model = CatClassifier(num_classes).to(device)

criterion = nn.CrossEntropyLoss()

learning_rate = 0.0001
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

num_epochs = 10

train_losses = []
valid_losses = []
accuracies = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    model.eval()
    valid_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            valid_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    train_loss /= len(train_loader)
    valid_loss /= len(valid_loader)
    accuracy = 100.0 * correct / total

    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    accuracies.append(accuracy)

    print(
        f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Accuracy: {accuracy:.2f}%')

model.eval()
test_loss = 0.0
correct = 0
total = 0
pred_labels = []
true_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        pred_labels.extend(predicted.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

test_loss /= len(test_loader)
accuracy = 100.0 * correct / total

print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.2f}%')

print("Drawing graphs...")

subset_indices = np.random.choice(len(test_dataset), size=30, replace=False)
subset_loader = DataLoader(test_dataset, batch_size=30, sampler=SubsetRandomSampler(subset_indices))

subset_images = []
subset_true_labels = []
subset_pred_labels = []

with torch.no_grad():
    for images, labels in subset_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        subset_images.extend(images.cpu())
        subset_true_labels.extend(labels.cpu().numpy())
        subset_pred_labels.extend(predicted.cpu().numpy())

plot_predictions(subset_images, subset_true_labels, subset_pred_labels, class_labels)
draw_graphs(num_epochs, train_losses, valid_losses, accuracies)
draw_confusion_matrix(true_labels, pred_labels, class_labels)

