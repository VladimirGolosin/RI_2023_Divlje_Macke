import os
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import torch.cuda
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image
from MLPT import MLP
import random
import itertools
import torch.nn.functional as F
# import torchvision.transforms.functional as F

batch_size = 32


def get_files():
    current_directory = os.getcwd()
    # print(current_directory)

    parent_directory = os.path.dirname(current_directory)
    os.chdir(parent_directory)
    # print(parent_directory)

    data_dir = os.path.join(parent_directory, 'data')
    # print(data_dir)

    files_in_dir = os.listdir(data_dir)
    train, valid, test, csv = "", "", "", []
    for file in files_in_dir:
        # print(file)
        file_path = os.path.join(data_dir, file)
        if file == "train":
            train = file_path
        elif file == "valid":
            valid = file_path
        elif file == "test":
            test = file_path
        else:
            csv.append(file_path)
    return train, valid, test, csv


def get_mean_and_sd(train_path):
    training_transforms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    train_dataset = torchvision.datasets.ImageFolder(root=train_path, transform=training_transforms)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
    mean, sd, total_count = 0, 0, 0
    for images, _ in train_loader:
        image_count_in_batch = images.size(0)
        # print(images.shape)
        images = images.view(image_count_in_batch, images.size(1), -1)
        # print(images.shape)
        mean += images.mean(2).sum(0)
        sd += images.std(2).sum(0)
        total_count += image_count_in_batch

    mean /= total_count
    sd /= total_count

    return mean, sd


# CW ruzno
def show_transformed_images(dataset):
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=6, shuffle=True)
    batch = next(iter(loader))
    images, labels = batch
    grid = torchvision.utils.make_grid(images, nrow=3)
    plt.figure(figsize=(11, 11))
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    plt.show()


def set_device():
    dev = "cpu"
    if torch.cuda.is_available():
        dev = "cuda"
        print("jo")
    return torch.device(dev)


def save_model(model, epoch, optimizer, best):
    state = {
        'epoch': epoch + 1,
        'model': model.state_dict(),
        'best_accuracy': best,
        'opitimizer': optimizer.state_dict()
    }
    torch.save(state, 'best_model.pth.tar')


def train_nn(model, train_loader, valid_loader, test_loader, criteria, optimizer, n_epochs):
    device = set_device()
    best_result = 0
    best_model = model
    epoch_best = 1

    for epoch in range(n_epochs):
        print("Epoch no %d " % (epoch + 1))
        model = best_model

        # Training
        model.train()
        train_loss, valid_loss, train_correct, valid_correct, train_total, valid_total = 0.0, 0.0, 0.0, 0.0, 0, 0
        for data in train_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            train_total += labels.size(0)

            optimizer.zero_grad()

            outputs = model.forward(images)

            _, predicted = torch.max(outputs.data, 1)

            loss = criteria(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_correct += (labels == predicted).sum().item()

        train_loss = train_loss / len(train_loader)
        train_accuracy = 100.00 * train_correct / train_total
        print("     -Training. Got %d / %d images correctly (%.3f%%). Train loss: %.3f"
              % (train_correct, train_total, train_accuracy, train_loss))

        # Validating
        model.eval()
        with torch.no_grad():
            for images, labels in valid_loader:
                images = images.to(device)
                labels = labels.to(device)
                valid_total += labels.size(0)

                outputs = model.forward(images)
                _, predicted = torch.max(outputs.data, 1)
                loss = criteria(outputs, labels)
                valid_loss += loss.item()

                valid_correct += (predicted == labels).sum().item()
        valid_loss = valid_loss / len(valid_loader)
        valid_accuracy = 100.00 * valid_correct / valid_total
        print("     -Validating. Got %d / %d images correctly (%.3f%%). Valid loss: %.3f"
              % (valid_correct, valid_total, valid_accuracy, valid_loss))

        if valid_accuracy > best_result:
            best_result = valid_accuracy
            # save_model(model, epoch, optimizer, best_result)
            best_model = model
            epoch_best = epoch
    save_model(best_model, epoch_best, optimizer, best_result)
    evaluate_model_on_test_set(best_model, test_loader)
    print("Finished")
    return best_model


def evaluate_model_on_test_set(model, test_loader):
    model.eval()
    predicted_correctly_on_epoch = 0
    total = 0
    device = set_device()
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            total += labels.size(0)

            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)

            predicted_correctly_on_epoch += (predicted == labels).sum().item()

    epoch_acc = 100.00 * predicted_correctly_on_epoch / total
    print("\nTesting. Got %d / %d images correctly (%.3f%%)."
          % (predicted_correctly_on_epoch, total, epoch_acc))
    return epoch_acc


def classify(model, image_transforms, image_path, classes, real='unknown'):
    model = model.eval()
    image = Image.open(image_path + '.jpg')
    image = image_transforms(image).float()
    image = image.unsqueeze(0)

    output = model.forward(image)
    _, predicted = torch.max(output.data, 1)

    print("Prediction: " + classes[predicted.item()] + ", real: " + real)



def set_up_nn(train, valid, test, csv):
    mean = [0.4851, 0.4405, 0.3614]
    sd = [0.2213, 0.2092, 0.2036]
    train_transforms_crop = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomCrop(224),
        transforms.RandomGrayscale(0.4),
        transforms.ToTensor(),
        transforms.Normalize(torch.Tensor(mean), torch.Tensor(sd))
    ])

    train_transforms_color = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.2), ratio=(0.75, 1.333)),
        transforms.ColorJitter(brightness=0.4, contrast=0.6, saturation=0.4, hue=0.3),
        transforms.ToTensor(),
        transforms.Normalize(torch.Tensor(mean), torch.Tensor(sd))
    ])


    train_transforms_blur = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(224),
        transforms.RandomVerticalFlip(),
        transforms.GaussianBlur(kernel_size=3),
        transforms.RandomPerspective(distortion_scale=0.7, p=0.7),
        transforms.ToTensor(),
        transforms.Normalize(torch.Tensor(mean), torch.Tensor(sd))
    ])


    normal_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomAffine(degrees=0, shear=20),
        transforms.ToTensor(),
        transforms.Normalize(torch.Tensor(mean), torch.Tensor(sd))
    ])

    mean = [0.5009, 0.4567, 0.3786]
    sd = [0.2143, 0.2058, 0.2002]
    valid_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(torch.Tensor(mean), torch.Tensor(sd))
    ])

    mean = [0.5002, 0.4565, 0.3743]
    sd = [0.1999, 0.1948, 0.1928]
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(torch.Tensor(mean), torch.Tensor(sd))
    ])

    train_dataset_blur = torchvision.datasets.ImageFolder(root=train, transform=train_transforms_blur)
    train_dataset_color = torchvision.datasets.ImageFolder(root=train, transform=train_transforms_color)
    train_dataset_crop = torchvision.datasets.ImageFolder(root=train, transform=train_transforms_crop)
    normal_dataset = torchvision.datasets.ImageFolder(root=train, transform=normal_transforms)

    show_transformed_images(train_dataset_crop)
    show_transformed_images(train_dataset_color)
    show_transformed_images(train_dataset_blur)
    show_transformed_images(normal_dataset)

    concatenated_dataset = torch.utils.data.ConcatDataset([normal_dataset, train_dataset_blur, train_dataset_color, train_dataset_crop])
    valid_dataset = torchvision.datasets.ImageFolder(root=valid, transform=valid_transforms)
    test_dataset = torchvision.datasets.ImageFolder(root=test, transform=test_transforms)

    train_loader = torch.utils.data.DataLoader(dataset=concatenated_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    lr_range = (0.001, 0.1)
    weight_decay_range = (0.0001, 0.01)
    dropout_range = (0.1, 0.9)
    best_accuracy = 0.0
    best_model = None
    best_lr = None
    best_weight_decay = None
    num_trials = 20

    input_size = 224 * 224 * 3
    # hidden_sizes = [32, 64, 128, 256]
    hidden_sizes = 32
    output_size = 10

    criterion = nn.CrossEntropyLoss()
    dropout_rate = 0.4

    # for n in range(num_trials):
    lr = random.uniform(*lr_range)
    weight_decay = random.uniform(*weight_decay_range)
    # dropout_rate = random.uniform(*dropout_range)

    # print(f"Trial number {n+1}, training with lr={lr}, weight_decay={weight_decay}, dropout={dropout_rate}")

    model = MLP(input_size, hidden_sizes, output_size, dropout_rate)
    device = set_device()
    model = model.to(device)
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.8, weight_decay=weight_decay)
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=0.0005)

    trained_model = train_nn(model, train_loader, valid_loader, test_loader, criterion, optimizer, 400)

    accuracy = evaluate_model_on_test_set(trained_model, valid_loader)

    # if accuracy > best_accuracy:
    #     best_accuracy = accuracy
    #     best_model = trained_model
    #     best_lr = lr
    #     best_weight_decay = weight_decay

    checkpoint = torch.load('best_model.pth.tar')
    print(checkpoint['epoch'])
    print(checkpoint['best_accuracy'])

    model.load_state_dict(checkpoint['model'])

    torch.save(model, 'best_model.pth')


if __name__ == '__main__':
    train, valid, test, csv = get_files()
    set_up_nn(train, valid, test, csv)
