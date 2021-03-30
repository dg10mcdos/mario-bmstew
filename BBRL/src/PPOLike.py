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


class PPOLike(nn.Module):

    def __init__(self, num_inputs, num_actions):
        super(PPOLike, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs * 128 * 4 * 4, 32, 3, stride=2, padding=1) # input 4 states channels, output channels
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.linear = nn.Linear(32 * 6 * 6, 512)
        self.critic_linear = nn.Linear(512, 1)
        self.soft = nn.Softmax(dim=1)
        self.actor_linear = nn.Linear(512, 6)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, nn.init.calculate_gain('relu'))
                # nn.init.xavier_uniform_(module.weight)
                # nn.init.kaiming_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, x): # x = curr_states, relu is activation function, if relu +ve, output the input x is 4,4,84,84

        # x [4,4,84,84]
        x = F.relu(self.conv1(x)) # input states, 32 output filters, convolution 3x3
        # x [4,32,42,42]
        x = F.relu(self.conv2(x)) # 32 input filters, 32 output filters. filters will learn to recognise "objects" in environment
        # x [4, 32, 21, 21]
        x = F.relu(self.conv3(x)) #
        # x [4,32,11,11]
        x = F.relu(self.conv4(x)) # x is 4, 32, 6, 6
        # x [4, 32, 6, 6]
        # x.view(x.size(0),-1) [4, 32 * 6 * 6] - [4,1152]
        x = self.linear(x.view(x.size(0), -1))
        # x [4, 512]
        # actor [4, 7] critic [4, 1]
        return self.soft(self.actor_linear(x)), self.critic_linear(x)