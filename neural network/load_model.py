import os
import torch
import torchvision
import torchvision.transforms as transforms
import PIL.Image as Image

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

def set_device():
    dev = "cpu"
    if torch.cuda.is_available():
        dev = "cuda"
        print("jo")
    return torch.device(dev)

def classify(model, image_transforms, image_path, classes, real=''):
    model = model.eval()
    image = Image.open(image_path)
    image = image_transforms(image).float()
    image = image.unsqueeze(0)

    output = model.forward(image)
    _, predicted = torch.max(output.data, 1)

    print("Prediction: " + classes[predicted.item()] + ", real: " + real)

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

if __name__ == '__main__':
    train, valid, test, csv = get_files()
    current_directory = os.getcwd()
    print(current_directory)
    # parent_directory = os.path.dirname(current_directory)
    # os.chdir(parent_directory)
    classes = ['AFRICAN LEOPARD',
               'CARACAL',
               'CHEETAH',
               'CLOUDED LEOPARD',
               'JAGUAR',
               'LION',
               'OCELOT',
               'PUMA',
               'SNOW LEOPARD',
               'TIGER']
    model = torch.load('best_model.pth')
    mean = [0.4851, 0.4405, 0.3614]
    sd = [0.2213, 0.2092, 0.2036]
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(torch.Tensor(mean), torch.Tensor(sd))
    ])
    test_dataset = torchvision.datasets.ImageFolder(root=test, transform=image_transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)
    evaluate_model_on_test_set(model, test_loader)
    classify(model, image_transform, 'african_leopard_8.jpg', classes, 'african_leopard')
    classify(model, image_transform, 'caracal_6.jpg', classes, 'caracal')
    classify(model, image_transform, 'cheetah_7.jpg', classes, 'cheetah')
    classify(model, image_transform, 'clouded_leopard_9.jpg', classes, 'clouded_leopard')
    classify(model, image_transform, 'jaguar_5.jpg', classes, 'jaguar')
    classify(model, image_transform, 'lion_6.jpg', classes, 'lion')
    classify(model, image_transform, 'ocelot_13.jpg', classes, 'ocelot')
    classify(model, image_transform, 'puma_20.jpg', classes, 'puma')
    classify(model, image_transform, 'snow_leopard_3.jpg', classes, 'snow_leopard')
    classify(model, image_transform, 'tiger_3.jpg', classes, 'tiger')