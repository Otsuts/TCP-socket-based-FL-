import torch
import dill
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.Sigmoid()
        )

    def forward(self, X):
        X = self.conv1(X)
        X = self.conv2(X)
        # print(X.flatten().shape)
        X = self.fc(X.view(X.shape[0],-1))
        return X
