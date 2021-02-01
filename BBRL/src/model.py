"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""

import torch.nn as nn
import torch.nn.functional as F


class PPO(nn.Module):
    def __init__(self, num_inputs, num_actions): # num_states, num_actions (e.g. 4 & 7)
        super(PPO, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1) # input 4 states channels, output channels
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.linear = nn.Linear(32 * 6 * 6, 512)
        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, num_actions)
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
        return self.actor_linear(x), self.critic_linear(x)
