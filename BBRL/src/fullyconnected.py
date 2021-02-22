import pickle
import time
import os

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

from behaviours import plot

class fullyconnected(nn.Module):
    def __init__(self):
        super(fullyconnected, self).__init__()
        self.fc1 = nn.Linear(128 * 4 * 4, 128)  # (6*4*4, 1)
        self.fc2 = nn.Linear(128, 6)

    def forward(self, x): # x is pixel motion from visualMotion
        x2 = x.view(x.size(0), -1)
        l1 = self.fc1(x2)
        output = self.fc2(l1)
        return output

    @staticmethod
    def loss_function(x_net, x, criterion=None):
        if criterion is None:
            # criterion = nn.BCEWithLogitsLoss(reduction='mean')#nn.CrossEntropyLoss()#nn.NLLLoss()#BCELoss()
            criterion = nn.MSELoss(reduction='mean')

        loss = criterion(x_net, x)
        return loss