from torch.optim.lr_scheduler import StepLR
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(12544, 128)  # Changed input from 9216 to 12544
        self.fc2 = nn.Linear(128, 6)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class Dilbert_Classifier:
    def __init__(self, device):
        self.model = ConvNet().to(device)
        self.optimizer = optim.Adadelta(self.model.parameters(), lr=1.0)
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=0.7)
        self.device = device

    def train(self, train_iter, epoch):
        self.model.train()
        for step, batch in enumerate(train_iter):
            data, target = batch['image'], batch['label']
            # print(data.shape)

            # Convert TensorFlow tensors to PyTorch tensors
            data = torch.from_numpy(data.numpy()).float().to(self.device)
            target = torch.from_numpy(target.numpy()).long().to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()
            if step % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, step * len(data), len(train_iter._dataset),
                    100. * step / len(train_iter), loss.item()))



    def test(self, eval_iter):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for step, batch in enumerate(eval_iter):
                data, target = batch['image'], batch['label']
                # print(data.shape)

                # Convert TensorFlow tensors to PyTorch tensors
                data = torch.from_numpy(data.numpy()).float().to(self.device)
                target = torch.from_numpy(target.numpy()).long().to(self.device)
                output = self.model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(eval_iter._dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(eval_iter._dataset),
            100. * correct / len(eval_iter._dataset)))




    def predict(self, input_tensor, threshold=0.7):
        self.model.eval()  # Set the model to evaluation mode
        input_tensor = input_tensor.to(self.device)  # Ensure the tensor is on the right device
        output = self.model(input_tensor)  # Pass the tensor through the model (raw output / logits)
        output_probabilities = torch.nn.functional.softmax(output, dim=1)
        _, preds = torch.max(output_probabilities, 1)  # Get the predicted labels
        confident_indices = output_probabilities.max(dim=1).values > threshold
        preds[~confident_indices] = -1  # Replace predictions below the threshold with -1 (or any invalid class label)
        return preds, output  # return raw output (logits) instead of probabilities

    def get_prediction_function(self):
        return self.predict