import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from helpers import *


def train_model(train_data, val_data, test_data, model, lr=0.001, epochs=50, batch_size=256):
    """
    Train the model on the train data and evaluate it on the validation and test data.
    :param train_data: The train data.
    :param val_data: The validation data.
    :param test_data: The test data.
    :param model: The model to train.
    :param lr: The learning rate.
    :param epochs: The number of epochs.
    :param batch_size: The batch size.
    :return: The trained model, train accuracies, validation accuracies, test accuracies, train losses, validation losses, test losses.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print('Using device:', device)

    trainset = torch.utils.data.TensorDataset(torch.tensor(train_data[['long', 'lat']].values).float(),
                                              torch.tensor(train_data['country'].values).long())
    valset = torch.utils.data.TensorDataset(torch.tensor(val_data[['long', 'lat']].values).float(),
                                            torch.tensor(val_data['country'].values).long())
    testset = torch.utils.data.TensorDataset(torch.tensor(test_data[['long', 'lat']].values).float(),
                                             torch.tensor(test_data['country'].values).long())

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    valloader = torch.utils.data.DataLoader(valset, batch_size=1024, shuffle=False, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1024, shuffle=False, num_workers=0)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_accs = []
    val_accs = []
    test_accs = []
    train_losses = []
    val_losses = []
    test_losses = []

    for ep in range(epochs):
        model.train()
        pred_correct = 0
        ep_loss = 0.
        for i, (inputs, labels) in enumerate(tqdm(trainloader)):
            #### YOUR CODE HERE ####

            # perform a training iteration

            # move the inputs and labels to the device
            inputs, labels = inputs.to(device, dtype=torch.float), labels.to(device)

            # zero the gradients
            optimizer.zero_grad()

            # forward pass
            outputs = model(inputs)

            # calculate the loss
            loss = criterion(outputs, labels)

            # backward pass
            loss.backward()

            # update the weights
            optimizer.step()

            #### END OF YOUR CODE ####

            pred_correct += (torch.argmax(outputs, dim=1) == labels).sum().item()
            ep_loss += loss.item()

        train_accs.append(pred_correct / len(trainset))
        train_losses.append(ep_loss / len(trainloader))

        model.eval()
        with torch.no_grad():
            for loader, accs, losses in zip([valloader, testloader], [val_accs, test_accs], [val_losses, test_losses]):
                correct = 0
                total = 0
                ep_loss = 0.
                for inputs, labels in loader:
                    #### YOUR CODE HERE ####

                    # perform an evaluation iteration

                    # move the inputs and labels to the device
                    inputs, labels = inputs.to(device, dtype=torch.float), labels.to(device)

                    # forward pass
                    outputs = model(inputs)

                    # calculate the loss
                    loss = criterion(outputs, labels)

                    #### END OF YOUR CODE ####

                    ep_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                accs.append(correct / total)
                losses.append(ep_loss / len(loader))

        print('Epoch {:}, Train Acc: {:.3f}, Val Acc: {:.3f}, Test Acc: {:.3f}'.format(ep, train_accs[-1], val_accs[-1],
                                                                                       test_accs[-1]))

    return model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses


def que_6_1_2_1():
    """
    Train the model with different learning rates and plot the validation losses.
    """
    learning_rates = [1., 0.01, 0.001, 0.00001]
    lr_val_losses = []
    output_dim = len(train_data['country'].unique())

    for lr in learning_rates:
        model = [nn.Linear(2, 16), nn.ReLU(),  # hidden layer 1
                 nn.Linear(16, 16), nn.ReLU(),  # hidden layer 2
                 nn.Linear(16, 16), nn.ReLU(),  # hidden layer 3
                 nn.Linear(16, 16), nn.ReLU(),  # hidden layer 4
                 nn.Linear(16, 16), nn.ReLU(),  # hidden layer 5
                 nn.Linear(16, 16), nn.ReLU(),  # hidden layer 6
                 nn.Linear(16, output_dim)  # output layer
                 ]
        model = nn.Sequential(*model)
        model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses = train_model(train_data,
                                                                                                    val_data, test_data,
                                                                                                    model, lr,
                                                                                                    epochs=50,
                                                                                                    batch_size=256)
        lr_val_losses.append(val_losses)

    plt.figure()
    for i, lr in enumerate(learning_rates):
        plt.plot(np.arange(1, 51), lr_val_losses[i], label=f'LR={lr}')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Loss')
    plt.title('Epochs vs validation loss for different learning rates')
    plt.legend()
    plt.show()


def que_6_1_2_2():
    """
    Train the model with different batch sizes and plot the test accuracies.
    """
    epochs_values = [1, 5, 10, 20, 50, 100]
    val_losses_specific_epochs = []
    output_dim = len(train_data['country'].unique())

    model = [nn.Linear(2, 16), nn.ReLU(),  # hidden layer 1
             nn.Linear(16, 16), nn.ReLU(),  # hidden layer 2
             nn.Linear(16, 16), nn.ReLU(),  # hidden layer 3
             nn.Linear(16, 16), nn.ReLU(),  # hidden layer 4
             nn.Linear(16, 16), nn.ReLU(),  # hidden layer 5
             nn.Linear(16, 16), nn.ReLU(),  # hidden layer 6
             nn.Linear(16, output_dim)  # output layer
             ]
    model = nn.Sequential(*model)
    model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses = train_model(train_data, val_data,
                                                                                                test_data, model,
                                                                                                lr=0.001, epochs=100,
                                                                                                batch_size=256)
    for epoch in epochs_values:
        val_losses_specific_epochs.append(val_losses[epoch - 1])

    plt.figure()
    plt.plot(epochs_values, val_losses_specific_epochs, marker='o', linestyle='-')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Loss')
    plt.title('Validation loss at specific epochs')
    plt.legend()
    plt.show()


def que_6_1_2_3():
    """
    Train the model with and without batch normalization and plot the validation losses at specific epochs.
    """
    epochs_values = [1, 5, 10, 20, 50, 100]
    val_losses_specific_epochs_without_norm = []
    val_losses_specific_epochs_with_norm = []
    output_dim = len(train_data['country'].unique())

    model_without_norm = [nn.Linear(2, 16), nn.ReLU(),  # hidden layer 1
                          nn.Linear(16, 16), nn.ReLU(),  # hidden layer 2
                          nn.Linear(16, 16), nn.ReLU(),  # hidden layer 3
                          nn.Linear(16, 16), nn.ReLU(),  # hidden layer 4
                          nn.Linear(16, 16), nn.ReLU(),  # hidden layer 5
                          nn.Linear(16, 16), nn.ReLU(),  # hidden layer 6
                          nn.Linear(16, output_dim)  # output layer
                          ]
    model_without_norm = nn.Sequential(*model_without_norm)
    model_without_norm, train_accs_without_norm, val_accs_without_norm, test_accs_without_norm, train_losses_without_norm, val_losses_without_norm, test_losses_without_norm = train_model(
        train_data, val_data, test_data, model_without_norm, lr=0.001, epochs=100, batch_size=256)

    model_with_norm = [nn.Linear(2, 16), nn.ReLU(),  # hidden layer 1
                       nn.BatchNorm1d(16),
                       nn.Linear(16, 16), nn.ReLU(),  # hidden layer 2
                       nn.BatchNorm1d(16),
                       nn.Linear(16, 16), nn.ReLU(),  # hidden layer 3
                       nn.BatchNorm1d(16),
                       nn.Linear(16, 16), nn.ReLU(),  # hidden layer 4
                       nn.BatchNorm1d(16),
                       nn.Linear(16, 16), nn.ReLU(),  # hidden layer 5
                       nn.BatchNorm1d(16),
                       nn.Linear(16, 16), nn.ReLU(),  # hidden layer 6
                       nn.BatchNorm1d(16),
                       nn.Linear(16, output_dim)  # output layer
                       ]
    model_with_norm = nn.Sequential(*model_with_norm)
    model_with_norm, train_accs_with_norm, val_accs_with_norm, test_accs_with_norm, train_losses_with_norm, val_losses_with_norm, test_losses_with_norm = train_model(
        train_data, val_data, test_data, model_with_norm, lr=0.001, epochs=100, batch_size=256)
    for epoch in epochs_values:
        val_losses_specific_epochs_without_norm.append(val_losses_without_norm[epoch - 1])
        val_losses_specific_epochs_with_norm.append(val_losses_with_norm[epoch - 1])

    plt.figure()
    plt.plot(epochs_values, val_losses_specific_epochs_without_norm, marker='o', linestyle='-',
             label="Without batch norm")
    plt.plot(epochs_values, val_losses_specific_epochs_with_norm, marker='o', linestyle='-', label="With batch norm")

    plt.xlabel('Epochs')
    plt.ylabel('Validation Loss')
    plt.title('Validation loss at specific epochs')
    plt.legend()
    plt.show()


def que_6_1_2_4():
    """
    Train the model with different batch sizes and plot the test accuracies.
    """
    batch_values = [1, 16, 128, 1024]
    epochs_values = [1, 10, 50, 50]
    test_accuracies_of_all_models = []
    # train_losses_of_all_models=[]
    # test_losses_of_all_models=[]
    val_losses_of_all_models = []
    output_dim = len(train_data['country'].unique())

    for i in range(4):
        model = [nn.Linear(2, 16), nn.ReLU(),  # hidden layer 1
                 nn.Linear(16, 16), nn.ReLU(),  # hidden layer 2
                 nn.Linear(16, 16), nn.ReLU(),  # hidden layer 3
                 nn.Linear(16, 16), nn.ReLU(),  # hidden layer 4
                 nn.Linear(16, 16), nn.ReLU(),  # hidden layer 5
                 nn.Linear(16, 16), nn.ReLU(),  # hidden layer 6
                 nn.Linear(16, output_dim)  # output layer
                 ]
        model = nn.Sequential(*model)

        model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses = train_model(train_data,
                                                                                                    val_data, test_data,
                                                                                                    model, lr=0.001,
                                                                                                    epochs=
                                                                                                    epochs_values[i],
                                                                                                    batch_size=
                                                                                                    batch_values[i])

        test_accuracies_of_all_models.append(test_accs)
        # train_losses_of_all_models.append(train_losses)
        # test_losses_of_all_models.append(test_losses)
        val_losses_of_all_models.append(val_losses)

    # accuracy plot
    plt.figure()
    for i in range(4):
        plt.plot(np.arange(1, epochs_values[i] + 1), test_accuracies_of_all_models[i], marker='o', linestyle='-',
                 label=f"Batch size= {batch_values[i]}")
    plt.xlabel('Epochs')
    plt.ylabel('Test accuracy')
    plt.title('Test accuracies vs epochs with different batch sizes')
    plt.legend()
    plt.show()

    # loss plot
    plt.figure()
    for i in range(4):
        plt.plot(np.arange(1, epochs_values[i] + 1), val_losses_of_all_models[i], marker='o', linestyle='-',
                 label=f"Batch size= {batch_values[i]}")
    plt.xlabel('Epochs')
    plt.ylabel('Validation Loss')
    plt.title('Epochs vs validation loss for different batch sizes')
    plt.legend()
    plt.show()


def que_6_2_1():
    """
    Train different models and compare them.
    :return: A dictionary of the models and their data.
    """
    models_dict = {}
    output_dim = len(train_data['country'].unique())

    # Model 1: Depth=1, Width=16
    model_1 = [nn.Linear(2, 16), nn.ReLU(),
               nn.Linear(16, output_dim)]
    model_1 = nn.Sequential(*model_1)
    model_1, train_accs_1, val_accs_1, test_accs_1, train_losses_1, val_losses_1, test_losses_1 = train_model(
        train_data, val_data, test_data, model_1, lr=0.001, epochs=10, batch_size=16)
    # Add to dictionary
    models_dict[model_1] = (
        model_1, 1, 16, train_accs_1, val_accs_1, test_accs_1, train_losses_1, val_losses_1, test_losses_1)
    print(f"Added a model with depth: 1, width: 16, epochs: 10, batch size: 16")

    # Model 2: Depth=2, Width=16
    model_2 = [nn.Linear(2, 16), nn.ReLU(),
               nn.Linear(16, 16), nn.ReLU(),
               nn.Linear(16, output_dim)]
    model_2 = nn.Sequential(*model_2)
    model_2, train_accs_2, val_accs_2, test_accs_2, train_losses_2, val_losses_2, test_losses_2 = train_model(
        train_data, val_data, test_data, model_2, lr=0.001, epochs=10, batch_size=16)
    # Add to dictionary
    models_dict[model_2] = (
        model_2, 2, 16, train_accs_2, val_accs_2, test_accs_2, train_losses_2, val_losses_2, test_losses_2)
    print(f"Added a model with depth: 2, width: 16, epochs: 10, batch size: 16")

    # Model 3: Depth=6, Width=16
    model_3 = [nn.Linear(2, 16), nn.ReLU(),  # hidden layer 1
               nn.Linear(16, 16), nn.ReLU(),  # hidden layer 2
               nn.Linear(16, 16), nn.ReLU(),  # hidden layer 3
               nn.Linear(16, 16), nn.ReLU(),  # hidden layer 4
               nn.Linear(16, 16), nn.ReLU(),  # hidden layer 5
               nn.Linear(16, 16), nn.ReLU(),  # hidden layer 6
               nn.Linear(16, output_dim)  # output layer
               ]
    model_3 = nn.Sequential(*model_3)
    model_3, train_accs_3, val_accs_3, test_accs_3, train_losses_3, val_losses_3, test_losses_3 = train_model(
        train_data, val_data, test_data, model_3, lr=0.001, epochs=50, batch_size=128)
    # Add to dictionary
    models_dict[model_3] = (
        model_3, 6, 16, train_accs_3, val_accs_3, test_accs_3, train_losses_3, val_losses_3, test_losses_3)
    print(f"Added a model with depth: 6, width: 16, epochs: 50, batch size: 128")

    # Model 4: Depth=10, Width=16
    model_4 = [nn.Linear(2, 16), nn.ReLU(),  # hidden layer 1
               nn.Linear(16, 16), nn.ReLU(),  # hidden layer 2
               nn.Linear(16, 16), nn.ReLU(),  # hidden layer 3
               nn.Linear(16, 16), nn.ReLU(),  # hidden layer 4
               nn.Linear(16, 16), nn.ReLU(),  # hidden layer 5
               nn.Linear(16, 16), nn.ReLU(),  # hidden layer 6
               nn.Linear(16, 16), nn.ReLU(),  # hidden layer 7
               nn.Linear(16, 16), nn.ReLU(),  # hidden layer 8
               nn.Linear(16, 16), nn.ReLU(),  # hidden layer 9
               nn.Linear(16, 16), nn.ReLU(),  # hidden layer 10
               nn.Linear(16, output_dim)  # output layer
               ]
    model_4 = nn.Sequential(*model_4)
    model_4, train_accs_4, val_accs_4, test_accs_4, train_losses_4, val_losses_4, test_losses_4 = train_model(
        train_data, val_data, test_data, model_4, lr=0.001, epochs=50, batch_size=128)
    # Add to dictionary
    models_dict[model_4] = (
        model_4, 10, 16, train_accs_4, val_accs_4, test_accs_4, train_losses_4, val_losses_4, test_losses_4)
    print(f"Added a model with depth: 10, width: 16, epochs: 50, batch size: 128")

    # Model 5: Depth=6, Width=8
    model_5 = [nn.Linear(2, 8), nn.ReLU(),  # hidden layer 1
               nn.Linear(8, 8), nn.ReLU(),  # hidden layer 2
               nn.Linear(8, 8), nn.ReLU(),  # hidden layer 3
               nn.Linear(8, 8), nn.ReLU(),  # hidden layer 4
               nn.Linear(8, 8), nn.ReLU(),  # hidden layer 5
               nn.Linear(8, 8), nn.ReLU(),  # hidden layer 6
               nn.Linear(8, output_dim)  # output layer
               ]
    model_5 = nn.Sequential(*model_5)
    model_5, train_accs_5, val_accs_5, test_accs_5, train_losses_5, val_losses_5, test_losses_5 = train_model(
        train_data, val_data, test_data, model_5, lr=0.001, epochs=50, batch_size=128)
    # Add to dictionary
    models_dict[model_5] = (
        model_5, 6, 8, train_accs_5, val_accs_5, test_accs_5, train_losses_5, val_losses_5, test_losses_5)
    print(f"Added a model with depth: 6, width: 8, epochs: 50, batch size: 128")

    # Model 6: Depth=6, Width=32
    model_6 = [nn.Linear(2, 32), nn.ReLU(),  # hidden layer 1
               nn.Linear(32, 32), nn.ReLU(),  # hidden layer 2
               nn.Linear(32, 32), nn.ReLU(),  # hidden layer 3
               nn.Linear(32, 32), nn.ReLU(),  # hidden layer 4
               nn.Linear(32, 32), nn.ReLU(),  # hidden layer 5
               nn.Linear(32, 32), nn.ReLU(),  # hidden layer 6
               nn.Linear(32, output_dim)  # output layer
               ]
    model_6 = nn.Sequential(*model_6)
    model_6, train_accs_6, val_accs_6, test_accs_6, train_losses_6, val_losses_6, test_losses_6 = train_model(
        train_data, val_data, test_data, model_6, lr=0.001, epochs=50, batch_size=128)
    # Add to dictionary
    models_dict[model_6] = (
        model_6, 6, 32, train_accs_6, val_accs_6, test_accs_6, train_losses_6, val_losses_6, test_losses_6)
    print(f"Added a model with depth: 6, width: 32, epochs: 50, batch size: 128")

    # Model 7: Depth=6, Width=64
    model_7 = [nn.Linear(2, 64), nn.ReLU(),  # hidden layer 1
               nn.Linear(64, 64), nn.ReLU(),  # hidden layer 2
               nn.Linear(64, 64), nn.ReLU(),  # hidden layer 3
               nn.Linear(64, 64), nn.ReLU(),  # hidden layer 4
               nn.Linear(64, 64), nn.ReLU(),  # hidden layer 5
               nn.Linear(64, 64), nn.ReLU(),  # hidden layer 6
               nn.Linear(64, output_dim)  # output layer
               ]
    model_7 = nn.Sequential(*model_7)
    model_7, train_accs_7, val_accs_7, test_accs_7, train_losses_7, val_losses_7, test_losses_7 = train_model(
        train_data, val_data, test_data, model_7, lr=0.001, epochs=50, batch_size=128)
    # Add to dictionary
    models_dict[model_7] = (
        model_7, 6, 64, train_accs_7, val_accs_7, test_accs_7, train_losses_7, val_losses_7, test_losses_7)
    print(f"Added a model with depth: 6, width: 64, epochs: 50, batch size: 128")

    return models_dict


def que_6_2_1_1(models_dict):
    """
    Compare the models and find the best one.
    :param models_dict: A dictionary of the models and their data.
    :return: The best model and its data.
    """
    # Find the best model
    best_model_name = None
    best_val_acc = 0.0

    for model_name, (_, _, _, _, val_accs, _, _, _, _) in models_dict.items():
        max_val_acc = np.mean(val_accs)
        if max_val_acc > best_val_acc:
            best_val_acc = max_val_acc
            best_model_name = model_name

    best_model_data = models_dict[best_model_name]
    best_model = best_model_data[0]  # Best model
    best_model_depth = best_model_data[1]
    best_model_width = best_model_data[2]
    best_val_acc = best_model_data[4]  # Validation accuracy
    best_train_losses = best_model_data[6]  # Training losses
    best_val_losses = best_model_data[7]  # Validation losses
    best_test_losses = best_model_data[8]  # Test losses

    # Compute the mean validation accuracy
    print("The best model's mean validation accuracy is: ", np.mean(best_val_acc))

    # Plot its training, validation and test losses
    plt.figure()
    plt.plot(range(1, len(best_train_losses) + 1), best_train_losses, label='Train Loss')
    plt.plot(range(1, len(best_val_losses) + 1), best_val_losses, label='Validation Loss')
    plt.plot(range(1, len(best_test_losses) + 1), best_test_losses, label='Test Loss')
    plt.title(f'Training, Validation, and Test Losses of the best model- ({best_model_depth}, {best_model_width})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Plot the prediction space of the model
    plot_decision_boundaries(best_model, test_data[['long', 'lat']].values, test_data['country'].values,
                             f'Decision Boundaries of the best model ({best_model_depth}, {best_model_width})',
                             implicit_repr=False)


def que_6_2_1_2(models_dict):
    """
    Compare the models and find the worst one.
    :param models_dict: A dictionary of the models and their data.
    :return: The worst model and its data.
    """
    # Find the worst model
    worst_model_name = None
    worst_val_acc = 1.0  # Initialize with a high value for finding the minimum

    for model_name, (_, _, _, _, val_accs, _, _, _, _) in models_dict.items():
        min_val_acc = np.mean(val_accs)
        if min_val_acc < worst_val_acc:
            worst_val_acc = min_val_acc
            worst_model_name = model_name

    worst_model_data = models_dict[worst_model_name]
    worst_model = worst_model_data[0]  # Worst model
    worst_model_depth = worst_model_data[1]
    worst_model_width = worst_model_data[2]
    worst_val_acc = worst_model_data[4]  # Validation accuracy
    worst_train_losses = worst_model_data[6]  # Training losses
    worst_val_losses = worst_model_data[7]  # Validation losses
    worst_test_losses = worst_model_data[8]  # Test losses

    # Compute the mean validation accuracy
    print("The worst model's mean validation accuracy is: ", np.mean(worst_val_acc))

    # Plot its training, validation and test losses
    plt.figure()
    plt.plot(range(1, len(worst_train_losses) + 1), worst_train_losses, label='Train Loss')
    plt.plot(range(1, len(worst_val_losses) + 1), worst_val_losses, label='Validation Loss')
    plt.plot(range(1, len(worst_test_losses) + 1), worst_test_losses, label='Test Loss')
    plt.title(f'Training, Validation, and Test Losses of the worst model- ({worst_model_depth}, {worst_model_width})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Plot the prediction space of the model
    plot_decision_boundaries(worst_model, test_data[['long', 'lat']].values, test_data['country'].values,
                             f'Decision Boundaries of the worst model ({worst_model_depth}, {worst_model_width})',
                             implicit_repr=False)


def que_6_2_1_3(models_width_16):
    """
    Plot accuracies vs. depth for models with width 16.
    :param models_width_16: A dictionary of the models and their data.
    """
    depths = []
    train_accs = []
    val_accs = []
    test_accs = []

    # Extract accuracies for each depth
    for model_name, info in models_width_16.items():
        print(model_name)
        depths.append(info[1])
        train_accs.append(np.mean(info[3]))
        val_accs.append(np.mean(info[4]))
        test_accs.append(np.mean(info[5]))

    print(train_accs)
    print(val_accs)
    print(test_accs)

    # Plot accuracies vs. depth
    plt.scatter(depths, train_accs, label='Train Accuracy')
    plt.scatter(depths, val_accs, label='Validation Accuracy')
    plt.scatter(depths, test_accs, label='Test Accuracy')
    plt.xlabel('Depth')
    plt.ylabel('Accuracy')
    plt.title('Accuracies vs. depth for models with width 16')
    plt.legend()
    plt.show()


def que_6_2_1_4(models_depth_6):
    """
    Plot accuracies vs. width for models with depth 6.
    :param models_depth_6: A dictionary of the models and their data.
    """
    widths = []
    train_accs = []
    val_accs = []
    test_accs = []

    # Extract accuracies for each width
    for model_name, info in models_depth_6.items():
        print(model_name)
        widths.append(info[2])
        train_accs.append(np.mean(info[3]))
        val_accs.append(np.mean(info[4]))
        test_accs.append(np.mean(info[5]))

    print(train_accs)
    print(val_accs)
    print(test_accs)

    # Plot accuracies vs. width
    plt.scatter(widths, train_accs, label='Train Accuracy')
    plt.scatter(widths, val_accs, label='Validation Accuracy')
    plt.scatter(widths, test_accs, label='Test Accuracy')
    plt.xlabel('Depth')
    plt.ylabel('Accuracy')
    plt.title('Accuracies vs. width for models with depth 6')
    plt.legend()
    plt.show()


def compute_avg_grad_magnitude(layer):
    """
    Compute the average gradient magnitude of the layer.
    :param layer: The layer.
    :return: The average gradient magnitude.
    """
    sum = 0.0
    cnt = 0

    for param in layer.parameters():
        if param.grad is not None:
            sum += torch.norm(param.grad).item() ** 2
            cnt += 1

    if cnt == 0:
        return 0

    return sum / cnt


def train_model_grad_magnitudes(train_data, val_data, test_data, model, lr=0.001, epochs=50, batch_size=256):
    """
    Train the model and get the gradients magnitudes.
    :param train_data: The train data.
    :param val_data: The validation data.
    :param test_data: The test data.
    :param model: The model to train.
    :param lr: The learning rate.
    :param epochs: The number of epochs.
    :param batch_size: The batch size.
    :return: The trained model and the gradients magnitudes.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print('Using device:', device)

    trainset = torch.utils.data.TensorDataset(torch.tensor(train_data[['long', 'lat']].values).float(),
                                              torch.tensor(train_data['country'].values).long())
    valset = torch.utils.data.TensorDataset(torch.tensor(val_data[['long', 'lat']].values).float(),
                                            torch.tensor(val_data['country'].values).long())
    testset = torch.utils.data.TensorDataset(torch.tensor(test_data[['long', 'lat']].values).float(),
                                             torch.tensor(test_data['country'].values).long())

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    valloader = torch.utils.data.DataLoader(valset, batch_size=1024, shuffle=False, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1024, shuffle=False, num_workers=0)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_accs = []
    val_accs = []
    test_accs = []
    train_losses = []
    val_losses = []
    test_losses = []

    grad_magnitudes = []

    for ep in range(epochs):
        layers = model.modules()
        model.train()
        pred_correct = 0
        ep_loss = 0.

        for i, (inputs, labels) in enumerate(tqdm(trainloader)):
            #### YOUR CODE HERE ####

            # perform a training iteration

            # move the inputs and labels to the device
            inputs, labels = inputs.to(device, dtype=torch.float), labels.to(device)

            # zero the gradients
            optimizer.zero_grad()

            # forward pass
            outputs = model(inputs)

            # calculate the loss
            loss = criterion(outputs, labels)

            # backward pass
            loss.backward()

            # update the weights
            optimizer.step()

            #### END OF YOUR CODE ####

            pred_correct += (torch.argmax(outputs, dim=1) == labels).sum().item()
            ep_loss += loss.item()

        avg_grad_magnitude = []
        for layer in layers:
            if isinstance(layer, nn.Linear):
                avg_grad_magnitude.append(compute_avg_grad_magnitude(layer))

        grad_magnitudes.append(avg_grad_magnitude)

        train_accs.append(pred_correct / len(trainset))
        train_losses.append(ep_loss / len(trainloader))

        model.eval()
        with torch.no_grad():
            for loader, accs, losses in zip([valloader, testloader], [val_accs, test_accs], [val_losses, test_losses]):
                correct = 0
                total = 0
                ep_loss = 0.
                for inputs, labels in loader:
                    #### YOUR CODE HERE ####

                    # perform an evaluation iteration

                    # move the inputs and labels to the device
                    inputs, labels = inputs.to(device, dtype=torch.float), labels.to(device)

                    # forward pass
                    outputs = model(inputs)

                    # calculate the loss
                    loss = criterion(outputs, labels)

                    #### END OF YOUR CODE ####

                    ep_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                accs.append(correct / total)
                losses.append(ep_loss / len(loader))

        print('Epoch {:}, Train Acc: {:.3f}, Val Acc: {:.3f}, Test Acc: {:.3f}'.format(ep, train_accs[-1], val_accs[-1],
                                                                                       test_accs[-1]))

    return model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses, grad_magnitudes


def que_6_1_2_5():
    """
    Train the model and plot the gradients magnitudes for selected layers.
    """
    output_dim = len(train_data['country'].unique())
    # Initial layers
    model = [nn.Linear(2, 4), nn.ReLU()]

    # Add 100 hidden layers
    for i in range(99):
        model.append(nn.Linear(4, 4))
        model.append(nn.ReLU())

    # Output layer
    model.append(nn.Linear(4, output_dim))

    # Create the sequential model
    model = nn.Sequential(*model)

    layers_to_plot = [0, 30, 60, 90, 95, 99]

    # Train the model and get the gradients magnitudes
    _, _, _, _, _, _, _, grad_magnitudes = train_model_grad_magnitudes(train_data, val_data, test_data, model, 0.001,
                                                                       10, 256)
    # Plot gradients magnitudes for selected layers
    plt.figure()
    for layer_index in layers_to_plot:
        plt.plot(grad_magnitudes[layer_index], label=f'Layer {layer_index}')

    plt.xlabel('Epochs')
    plt.ylabel('Gradient Magnitude')
    plt.title('Gradient Magnitude vs. Epochs for selected layers')
    plt.legend()
    plt.show()


def que_6_2_1_7(train_data, val_data, test_data):
    """
    Train the model with implicit representation and plot the decision boundaries.
    :param train_data: the train data
    :param val_data:  the validation data
    :param test_data: the test data
    :return:
    """
    output_dim = len(train_data['country'].unique())
    model = [nn.Linear(2, 16), nn.ReLU(),  # hidden layer 1
             nn.BatchNorm1d(16),
             nn.Linear(16, 16), nn.ReLU(),  # hidden layer 2
             nn.BatchNorm1d(16),
             nn.Linear(16, 16), nn.ReLU(),  # hidden layer 3
             nn.BatchNorm1d(16),
             nn.Linear(16, 16), nn.ReLU(),  # hidden layer 4
             nn.BatchNorm1d(16),
             nn.Linear(16, 16), nn.ReLU(),  # hidden layer 5
             nn.BatchNorm1d(16),
             nn.Linear(16, 16), nn.ReLU(),  # hidden layer 6
             nn.BatchNorm1d(16),
             nn.Linear(16, output_dim)  # output layer
             ]

    model = nn.Sequential(*model)
    update_train_data = train_data.copy()
    update_val_data = val_data.copy()
    update_test_data = test_data.copy()

    for i, a in enumerate(np.arange(0.1, 1.1, 0.1), start=1):
        update_train_data[f'feature{i}'] = np.sin(a * train_data['long'].values)
        update_val_data[f'feature{i}'] = np.sin(a * val_data['long'].values)
        update_test_data[f'feature{i}'] = np.sin(a * test_data['long'].values)

    for i, a in enumerate(np.arange(0.1, 1.1, 0.1), start=10 + 1):
        update_train_data[f'feature{i}'] = np.sin(a * train_data['lat'].values)
        update_val_data[f'feature{i}'] = np.sin(a * val_data['lat'].values)
        update_test_data[f'feature{i}'] = np.sin(a * test_data['lat'].values)

    update_train_data.drop(columns=['long'], inplace=True)
    update_train_data.drop(columns=['lat'], inplace=True)
    update_train_data.drop(columns=['country'], inplace=True)
    update_train_data['country'] = train_data['country']

    update_val_data.drop(columns=['long'], inplace=True)
    update_val_data.drop(columns=['lat'], inplace=True)
    update_val_data.drop(columns=['country'], inplace=True)
    update_val_data['country'] = train_data['country']

    update_test_data.drop(columns=['long'], inplace=True)
    update_test_data.drop(columns=['lat'], inplace=True)
    update_test_data.drop(columns=['country'], inplace=True)
    update_test_data['country'] = train_data['country']

    model, train_accs, val_accs, test_accs_6, train_losses, val_losses, test_losses = train_model_implicit_repr(
        update_train_data, update_val_data, update_test_data, model,
        lr=0.001, epochs=50, batch_size=256)

    test_data_features = test_data[['long', 'lat']].values
    test_data_labels = test_data[['country']].values.flatten()

    plot_decision_boundaries(model, test_data_features, test_data_labels,
                             "Decision boundaries implicit representation", implicit_repr=True)


def train_model_implicit_repr(train_data, val_data, test_data, model, lr=0.001, epochs=50, batch_size=256):
    """
    Train the model with implicit representation.
    :param train_data: The train data.
    :param val_data: The validation data.
    :param test_data: The test data.
    :param model: The model to train.
    :param lr: The learning rate.
    :param epochs: The number of epochs.
    :param batch_size: The batch size.
    :return: The trained model and the losses.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print('Using device:', device)

    trainset = torch.utils.data.TensorDataset(
        torch.tensor(train_data[['feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6',
                                 'feature7', 'feature8', 'feature9', 'feature10',
                                 'feature11', 'feature12', 'feature13', 'feature14',
                                 'feature15', 'feature16', 'feature17', 'feature18', 'feature19',
                                 'feature20']].values).float(), torch.tensor(train_data['country'].values).long())

    valset = torch.utils.data.TensorDataset(
        torch.tensor(val_data[['feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6',
                               'feature7', 'feature8', 'feature9', 'feature10',
                               'feature11', 'feature12', 'feature13', 'feature14',
                               'feature15', 'feature16', 'feature17', 'feature18', 'feature19',
                               'feature20']].values).float(), torch.tensor(val_data['country'].values).long())

    testset = torch.utils.data.TensorDataset(
        torch.tensor(test_data[['feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6',
                                'feature7', 'feature8', 'feature9', 'feature10',
                                'feature11', 'feature12', 'feature13', 'feature14',
                                'feature15', 'feature16', 'feature17', 'feature18', 'feature19',
                                'feature20']].values).float(), torch.tensor(test_data['country'].values).long())

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    valloader = torch.utils.data.DataLoader(valset, batch_size=1024, shuffle=False, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1024, shuffle=False, num_workers=0)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_accs = []
    val_accs = []
    test_accs = []
    train_losses = []
    val_losses = []
    test_losses = []

    for ep in range(epochs):
        model.train()
        pred_correct = 0
        ep_loss = 0.
        for i, (inputs, labels) in enumerate(tqdm(trainloader)):
            #### YOUR CODE HERE ####

            # perform a training iteration

            # move the inputs and labels to the device
            inputs, labels = inputs.to(device, dtype=torch.float), labels.to(device)

            # zero the gradients
            optimizer.zero_grad()

            # forward pass
            outputs = model(inputs)

            # calculate the loss
            loss = criterion(outputs, labels)

            # backward pass
            loss.backward()

            # update the weights
            optimizer.step()

            #### END OF YOUR CODE ####

            pred_correct += (torch.argmax(outputs, dim=1) == labels).sum().item()
            ep_loss += loss.item()

        train_accs.append(pred_correct / len(trainset))
        train_losses.append(ep_loss / len(trainloader))

        model.eval()
        with torch.no_grad():
            for loader, accs, losses in zip([valloader, testloader], [val_accs, test_accs], [val_losses, test_losses]):
                correct = 0
                total = 0
                ep_loss = 0.
                for inputs, labels in loader:
                    #### YOUR CODE HERE ####

                    # perform an evaluation iteration

                    # move the inputs and labels to the device
                    inputs, labels = inputs.to(device, dtype=torch.float), labels.to(device)

                    # forward pass
                    outputs = model(inputs)

                    # calculate the loss
                    loss = criterion(outputs, labels)

                    #### END OF YOUR CODE ####

                    ep_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                accs.append(correct / total)
                losses.append(ep_loss / len(loader))

        print('Epoch {:}, Train Acc: {:.3f}, Val Acc: {:.3f}, Test Acc: {:.3f}'.format(ep, train_accs[-1], val_accs[-1],
                                                                                       test_accs[-1]))

    return model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses


if __name__ == '__main__':
    # seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    train_data = pd.read_csv('train.csv')
    val_data = pd.read_csv('validation.csv')
    test_data = pd.read_csv('test.csv')

    models_dict = que_6_2_1()
    que_6_2_1_1(models_dict)
    que_6_2_1_7(train_data, val_data, test_data)
