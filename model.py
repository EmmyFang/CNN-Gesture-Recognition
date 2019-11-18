'''
    Write a model for gesture classification.
'''

import torch.nn as nn
import torch.nn.functional as f
import torch
import numpy as np

class Net (nn.Module):
    def __init__ (self, input_size,kernel_num, kernel_size, hidden_layer_size, hidden_layer_num, conv_layer_num, pooling_size):
        super(Net, self).__init__()

        if (conv_layer_num == 2):
            self.conv1 = nn.Conv1d(input_size, kernel_num, kernel_size=kernel_size, padding=kernel_size // 2)
            self.conv2 = nn.Conv1d(kernel_num, kernel_num, kernel_size=kernel_size, padding=kernel_size // 2)

        elif (conv_layer_num == 3):
            self.conv1 = nn.Conv1d(input_size, kernel_num, kernel_size, stride=1, padding=kernel_size // 2)
            self.conv2 = nn.Conv1d(kernel_num, kernel_num, kernel_size, stride=1, padding=kernel_size // 2)
            self.conv3 = nn.Conv1d(kernel_num, kernel_num, kernel_size, stride=1, padding=kernel_size // 2)
        elif (conv_layer_num == 1):
            self.conv1 = nn.Conv1d(input_size, kernel_num, kernel_size, stride=1, padding=kernel_size // 2)

        self.conv_layer_num = conv_layer_num
        self.pool = nn.MaxPool1d(kernel_size=pooling_size, stride=pooling_size)

        self.hidden_layer_num = hidden_layer_num
        self.MLP_first = 64*6
        self.drop_out = nn.Dropout(p=0.1)
        self.batchnorm = nn.BatchNorm1d(self.MLP_first)

        if (hidden_layer_num == 1):
            self.fc1 = nn.Linear(self.MLP_first, hidden_layer_size)
            self.fc2 = nn.Linear(hidden_layer_size, 26)



        elif (hidden_layer_num ==2):
            self.fc1 = nn.Linear (self.MLP_first,self.MLP_first//2)
            self.fc2 = nn.Linear(self.MLP_first//2, self.MLP_first//4)
            self.fc3 = nn.Linear(self.MLP_first//4, 26)
        elif (hidden_layer_num == 0):
            self.fc1 = nn.Linear(self.MLP_first, 26)

    def forward(self,x):
        x = np.transpose(x, (0,2,1))
        x = x.float()

        if (self.conv_layer_num == 2):
            x = self.pool(f.relu(self.conv1(x)))
            x = self.pool(f.relu(self.conv2(x)))
        # elif (self.conv_layer_num == 3):
        #     x = self.pool(f.relu(self.conv1(x)))
        #     x = self.pool(f.relu(self.conv2(x)))
        #     x = self.pool(f.relu(self.conv3(x)))
        # elif (self.conv_layer_num == 1):
        #     x = self.pool(f.relu(self.conv1(x)))

        x = x.view(-1, self.MLP_first)

        x = self.batchnorm(x)
        x = self.drop_out(x)

        if (self.hidden_layer_num == 1):
            x = f.relu(self.fc1(x))
            x = self.fc2(x)


        elif (self.hidden_layer_num == 2):

            x = f.relu(self.fc1(x))
            x = f.relu(self.fc2(x))
            x = self.fc3(x)

        elif (self.hidden_layer_num == 0):
            x = self.fc1(x)


        return x