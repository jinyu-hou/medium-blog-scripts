from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

import numpy as np
import torch

import matplotlib.pyplot as plt

import math
import pandas as pd
import numpy as np
import csv
import os


def load_data(path):
    # A helper function to load the csv file.
    if not os.path.exists(path):
        raise Exception("The specified path {} does not exist.".format(path))
    data = pd.read_csv(path)    

    # some data manipulation
    data = data.dropna()
    data = data.drop("ID", axis=1)
    data['Gender'] = (data['Gender'] == "Male").astype(int) #convert Female and Male to 0 and 1
    data['Ever_Married'] = (data['Ever_Married'] == "Yes").astype(int)
    data['Graduated'] = (data['Graduated'] == "Yes").astype(int)
    profession = ['Healthcare', 'Entertainment', 'Executive', 'Lawyer', 'Artist', 'Engineer', 'Doctor', 'Marketing', 'Homemaker']
    spending = ['Low', 'Average', 'High']
    var_1 = ['Cat_1', 'Cat_2', 'Cat_3', 'Cat_4', 'Cat_5', 'Cat_6', 'Cat_7']
    segmentation = ['A', 'B', 'C', 'D']
    data['Profession'] = data['Profession'].apply(lambda x: profession.index(x))
    data['Spending_Score'] = data['Spending_Score'].apply(lambda x: spending.index(x))
    data['Var_1'] = data['Var_1'].apply(lambda x: var_1.index(x))
    data['Segmentation'] = data['Segmentation'].apply(lambda x: segmentation.index(x))

    # Split into training and test sets
    msk1 = np.random.rand(len(data)) < 0.8
    train = data[msk1]
    valid_test = data[~msk1]
    msk2 = np.random.rand(len(valid_test)) < 0.5
    valid = valid_test[msk2]
    test = valid_test[~msk2]

    return train, valid, test


class CustomerDataset(data.Dataset):
    def __init__(self, train_data, batch_size):
        self.labels = train_data['Segmentation'].values
        self.train = train_data.drop("Segmentation",axis=1).values
        self.batch_size = batch_size
        self.n_cases = train_data.shape[0]

    def __getitem__(self, index):
        begin, end = index*self.batch_size, min((index+1)*self.batch_size, self.n_cases)
        batch = torch.Tensor(self.train[begin:end].astype(float))
        targets = torch.Tensor(self.labels[begin:end]).to(torch.long)
        one_hot = torch.zeros(end - begin, 4)
        one_hot[torch.arange(end - begin), targets] = 1
        return Variable(batch), targets, Variable(one_hot)

    def __len__(self):
        return math.ceil(self.n_cases / self.batch_size)


class SegmentationNet(nn.Module):
    def __init__(self, feature, hidden1, hidden2, output):
        """ Initialize a class NeuralNet.

        :param batch_size: int
        :param hidden: int
        """
        super(SegmentationNet, self).__init__()

        # Define linear functions.
        self.layer1 = nn.Linear(feature, hidden1)
        self.layer2 = nn.Linear(hidden1, hidden2)
        self.layer3 = nn.Linear(hidden2, output)

    def get_weight_norm(self):
        """ Return ||W||

        :return: float
        """
        layer_1_w_norm = torch.norm(self.layer1.weight, 2)
        layer_2_w_norm = torch.norm(self.layer2.weight, 2)
        layer_3_w_norm = torch.norm(self.layer3.weight, 2)
        return layer_1_w_norm + layer_2_w_norm + layer_3_w_norm

    def forward(self, inputs):
        """ Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        out = inputs
        out = self.layer1(out)
        out = F.relu(out)
        out = self.layer2(out)
        out = F.relu(out)
        out = self.layer3(out)
        out = F.softmax(out, dim=1)
        return out


def train(model, lr, lamb, batch_size, train_data, valid_data, num_epoch):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param valid_data: 2D FloatTensor
    :param num_epoch: int
    :return: None
    """
    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Preprocess the training and validation data
    n_cases = float(train_data.shape[0])
    train_dataset = CustomerDataset(train_data, batch_size)
    valid_labels = torch.Tensor(valid_data['Segmentation'].values)
    valid_dataset = torch.Tensor(valid_data.drop("Segmentation",axis=1).values)

    # list to record change in learning objective
    train_accs = []
    valid_accs = []

    for epoch in range(0, num_epoch):
        train_loss = 0.
        correct_guess = 0

        for batch in range(len(train_dataset)):
            inputs, targets, one_hot_targets = train_dataset[batch]

            optimizer.zero_grad()
            output = model(inputs)

            loss = torch.sum((output - one_hot_targets) ** 2) + lamb * model.get_weight_norm() / 2
            loss.backward()

            optimizer.step()
            train_loss += loss.item()

            guess = torch.argmax(output, dim=1)
            correct_guess += torch.sum((guess == targets))

        train_acc = correct_guess / n_cases
        train_accs.append(train_acc)
        valid_acc = evaluate(model, valid_dataset, valid_labels)
        valid_accs.append(valid_acc)
        if epoch % 100 == 0:
            print("Epoch: {} \tTraining Cost: {:.6f}\t "
                "Train Acc: {}\t"
                "Valid Acc: {}".format(epoch, train_loss, train_acc, valid_acc))

    epochs = list(range(num_epoch))
    plt.plot(epochs, train_accs, label = "training accuracy")
    plt.plot(epochs, valid_accs, label = "validation accuracy")
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Training & Validation Accuracy vs. Epoch')
    plt.legend()
    plt.savefig("seg_nn_accuracy.png")


def evaluate(model, dataset, labels):
    """ Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = float(dataset.shape[0])
    correct = 0
    output = model(dataset)
    guess = torch.argmax(output, dim=1)
    correct_guess = torch.sum((guess == labels))

    return correct_guess / total


def main():
    train_data, valid_data, test_data = load_data("train.csv")
    n_cases, n_features = train_data.shape
    n_features -= 1

    # Set model hyperparameters.
    model = SegmentationNet(feature=n_features, hidden1=20, hidden2=20, output=4)

    # Set optimization hyperparameters.
    lr = 0.02
    num_epoch = 200
    lamb = 0.1
    batch_size = 512

    # train the model
    train(model, lr, lamb, batch_size, train_data, valid_data, num_epoch)
    
    # run model on test set
    test_labels = torch.Tensor(test_data['Segmentation'].values)
    test_dataset = torch.Tensor(test_data.drop("Segmentation",axis=1).values)
    test_acc = evaluate(model, test_dataset, test_labels)
    print(f"Test Acc: {test_acc}")



if __name__ == "__main__":
    main()
