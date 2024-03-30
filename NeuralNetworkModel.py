import pandas as pd
import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split


class NeuralNetworkModel(nn.Module):

    def __init__(self, input_size, output_size):
        super(NeuralNetworkModel, self).__init__()
        self.relu = nn.ReLU()
        self.l1 = nn.Linear(input_size, 512)
        self.l2 = nn.Linear(512, 128)
        self.l3 = nn.Linear(128, output_size)

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.l3(x)
        result = torch.softmax(x, dim=1)
        # print(result)
        return result
