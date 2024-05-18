import os

import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import torch
import torchvision
from tqdm import tqdm
from torchvision import transforms
import numpy as np


class ResNet18(nn.Module):
    def __init__(self, pretrained=False, probing=False):
        super(ResNet18, self).__init__()
        if pretrained:
            weights = ResNet18_Weights.IMAGENET1K_V1
            self.transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
            self.resnet18 = resnet18(weights=weights)
        else:
            self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
            self.resnet18 = resnet18()
        in_features_dim = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Identity()
        if probing:
            for name, param in self.resnet18.named_parameters():
                param.requires_grad = False
        self.logistic_regression = nn.Linear(in_features_dim, 1)

    def forward(self, x):
        features = self.resnet18(x)
        output = self.logistic_regression(features)
        return output


def get_loaders(path, transform, batch_size):
    """
    Get the data loaders for the train, validation and test sets.
    :param path: The path to the 'whichfaceisreal' directory.
    :param transform: The transform to apply to the images.
    :param batch_size: The batch size.
    :return: The train, validation and test data loaders.
    """
    train_set = torchvision.datasets.ImageFolder(root=os.path.join(path, 'train'), transform=transform)
    val_set = torchvision.datasets.ImageFolder(root=os.path.join(path, 'val'), transform=transform)
    test_set = torchvision.datasets.ImageFolder(root=os.path.join(path, 'test'), transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def compute_accuracy(model, data_loader, device):
    """
    Compute the accuracy of the model on the data in data_loader
    :param model: The model to evaluate.
    :param data_loader: The data loader.
    :param device: The device to run the evaluation on.
    :return: The accuracy of the model on the data in data_loader
    """
    model.eval()
    with torch.no_grad():
        correct = 0
        for inputs, labels in data_loader:
            # perform an evaluation iteration
            # move the inputs and labels to the device
            inputs, labels = inputs.to(device), labels.to(device)
            # forward pass
            outputs = model(inputs)
            # Apply sigmoid
            outputs = torch.sigmoid(outputs)
            preds = (outputs >= 0.5).float()
            preds = preds.squeeze(dim=1)
            correct += torch.sum(preds == labels).item()

    return correct / len(data_loader.dataset)


def run_training_epoch(model, criterion, optimizer, train_loader, device):
    """
    Run a single training epoch
    :param model: The model to train
    :param criterion: The loss function
    :param optimizer: The optimizer
    :param train_loader: The data loader
    :param device: The device to run the training on
    :return: The average loss for the epoch.
    """
    model.train()
    epoch_loss = 0.0

    for (imgs, labels) in tqdm(train_loader, total=len(train_loader)):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs.squeeze(), labels.float())
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(train_loader)


def resnet_train(batch_size=32, num_of_epochs=1, learning_rate=0.1, type=[False, False]):
    # Set the random seed for reproducibility
    torch.manual_seed(42)

    ### UNCOMMENT THE FOLLOWING LINES TO TRAIN THE MODEL ###
    # From Scratch
    model = ResNet18(pretrained=type[0], probing=type[1])
    # Linear probing
    # model = ResNet18(pretrained=True, probing=True)
    # Fine-tuning
    # model = ResNet18(pretrained=True, probing=False)

    transform = model.transform

    path = '/content/drive/MyDrive/Year 2/Semester A/Machine Learning Methods/excercise 4/whichfaceisreal'  # For example '/cs/usr/username/whichfaceisreal/'
    train_loader, val_loader, test_loader = get_loaders(path, transform, batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    ### Define the loss function and the optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    ### Train the model

    # Train the model
    for epoch in range(num_of_epochs):
        # Run a training epoch
        loss = run_training_epoch(model, criterion, optimizer, train_loader, device)
        # Compute the accuracy
        train_acc = compute_accuracy(model, train_loader, device)
        # Compute the validation accuracy
        val_acc = compute_accuracy(model, val_loader, device)
        print(f'Epoch {epoch + 1}/{num_of_epochs}, Loss: {loss:.4f}, Val accuracy: {val_acc:.4f}')
        # Stopping condition
        ### YOUR CODE HERE ###

    # Compute the test accuracy
    test_acc = compute_accuracy(model, test_loader, device)
    return test_acc


def compute_accuracy_and_get_preds(model, data_loader, device):
    """
    Compute the accuracy of the model on the data in data_loader
    :param model: The model to evaluate.
    :param data_loader: The data loader.
    :param device: The device to run the evaluation on.
    :return: The accuracy of the model on the data in data_loader
    """
    model.eval()
    predictions = []

    with torch.no_grad():
        correct = 0
        for inputs, labels in data_loader:
            # perform an evaluation iteration
            # move the inputs and labels to the device
            inputs, labels = inputs.to(device), labels.to(device)
            # forward pass
            outputs = model(inputs)
            # Apply sigmoid
            outputs = torch.sigmoid(outputs)
            preds = (outputs >= 0.5).float()
            preds = preds.squeeze(dim=1)
            correct += torch.sum(preds == labels).item()
            predictions.append(preds.cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)

    return correct / len(data_loader.dataset), predictions


def get_models_preds(lr, type):
    """
    This function is used to get the predictions of the best and worst models
    :param lr:  learning rate
    :param type:  type of model
    :return:predictions
    """
    torch.manual_seed(42)

    model = ResNet18(pretrained=type[0], probing=type[1])

    transform = model.transform
    batch_size = 32
    num_of_epochs = 1
    path = '/content/drive/MyDrive/Year 2/Semester A/Machine Learning Methods/excercise 4/whichfaceisreal'  # For example '/cs/usr/username/whichfaceisreal/'
    train_loader, val_loader, test_loader = get_loaders(path, transform, batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    ### Define the loss function and the optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ### Train the model

    # Train the model
    for epoch in range(num_of_epochs):
        # Run a training epoch
        loss = run_training_epoch(model, criterion, optimizer, train_loader, device)
        # Compute the accuracy
        train_acc = compute_accuracy(model, train_loader, device)
        # Compute the validation accuracy
        val_acc, preds = compute_accuracy_and_get_preds(model, val_loader, device)
        print(f'Epoch {epoch + 1}/{num_of_epochs}, Loss: {loss:.4f}, Val accuracy: {val_acc:.4f}')
        # Stopping condition
        ### YOUR CODE HERE ###

    # Compute the test accuracy
    test_acc = compute_accuracy(model, test_loader, device)

    return preds


def create_models():
    """
    This function is used to create the models and train them
    """
    learning_rates = [0.00001, 0.0001, 0.001, 0.01, 0.1]
    all_models_accs = []
    # From Scratch
    for lr in learning_rates:
        acc = resnet_train(32, 1, lr, [False, False])
        all_models_accs.append(acc)

    # Linear probing
    for lr in learning_rates:
        acc = resnet_train(32, 1, lr, [True, True])
        all_models_accs.append(acc)

    # Fine-tuning
    for lr in learning_rates:
        acc = resnet_train(32, 1, lr, [True, False])
        all_models_accs.append(acc)

    # Print all accuracies
    for i, acc in enumerate(all_models_accs):
        # Determine the model type and learning rate corresponding to the current accuracy
        lr_index = i % len(learning_rates)  # Index of learning rate in learning_rates list
        model_index = i // len(learning_rates)  # Index of model configuration

        # Determine model type
        if model_index == 0:
            model_type = "From Scratch"
        elif model_index == 1:
            model_type = "Linear Probing"
        else:
            model_type = "Fine-tuning"

        # Print model type, learning rate, and accuracy
        print(f"Model Type: {model_type}, Learning Rate: {learning_rates[lr_index]}, Accuracy: {acc}")


def last_que():
    """
    This function is used to answer the last question in the exercise
    """
    # best model
    best_model_preds = get_models_preds(0.0001, [True, False])
    worst_model_preds = get_models_preds(0.1, [False, False])
    test_loader = get_loaders(os.path.join(os.getcwd(),
                                           '/content/drive/MyDrive/Year 2/Semester A/Machine Learning Methods/excercise 4/whichfaceisreal'),
                              transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]), 32)[2]
    img_cnt = 0
    for i in range(len(best_model_preds)):
        label = test_loader.dataset[i][1]
        # check if the best model succeeded and the worst model did not
        if best_model_preds[i] == label and worst_model_preds[i] != label:
            print(i)
            plt.imshow(test_loader.dataset[i][0].permute(1, 2, 0))
            plt.show()
            img_cnt += 1
            if img_cnt == 5:
                break


# if __name__ == '__main__':
# Set the random seed for reproducibility
torch.manual_seed(42)

### UNCOMMENT THE FOLLOWING LINES TO TRAIN THE MODEL ###
# From Scratch
# model = ResNet18(pretrained=False, probing=False)
# Linear probing
# model = ResNet18(pretrained=True, probing=True)
# Fine-tuning
model = ResNet18(pretrained=True, probing=False)

transform = model.transform
batch_size = 32
num_of_epochs = 1
learning_rate = 0.001
path = 'G:\My Drive\Year 2\Semester A\Machine Learning Methods\excercise 4\whichfaceisreal'  # For example '/cs/usr/username/whichfaceisreal/'
train_loader, val_loader, test_loader = get_loaders(path, transform, batch_size)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
### Define the loss function and the optimizer
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
### Train the model

# Train the model
for epoch in range(num_of_epochs):
    # Run a training epoch
    loss = run_training_epoch(model, criterion, optimizer, train_loader, device)
    # Compute the accuracy
    train_acc = compute_accuracy(model, train_loader, device)
    # Compute the validation accuracy
    val_acc = compute_accuracy(model, val_loader, device)
    print(f'Epoch {epoch + 1}/{num_of_epochs}, Loss: {loss:.4f}, Val accuracy: {val_acc:.4f}')
    # Stopping condition
    ### YOUR CODE HERE ###

# Compute the test accuracy
test_acc = compute_accuracy(model, test_loader, device)
