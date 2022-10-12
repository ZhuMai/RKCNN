import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn.functional as F
from torch.nn import init

def norm(dim):
    return nn.BatchNorm2d(dim, eps=0.001, momentum=0.01)


class attention(nn.Module):
    def __init__(self, input_channels):
        super(attention, self).__init__()

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(in_features = input_channels,out_features = input_channels // 2)
        self.fc2 = nn.Linear(in_features = input_channels // 2, out_features = input_channels)


    def forward(self, x):
        output = self.pool(x)
        output = output.view(output.size()[0], output.size()[1])
        output = self.fc1(output)
        output = F.relu(output, inplace=True)
        output = self.fc2(output)
        output = torch.sigmoid(output)
        output = output.view(output.size()[0],output.size()[1],1,1)
        output = torch.mul(x, output)
        return output


class transition(nn.Module):
    def __init__(self, if_att, current_size, input_channels, out_channels, keep_prob):
        super(transition, self).__init__()
        self.input_channels = input_channels
        self.out_channels = out_channels
        self.keep_prob = keep_prob
        self.bn = norm(self.input_channels)
        self.conv = nn.Conv2d(self.input_channels, self.out_channels, kernel_size = 1, bias = False)
        if self.keep_prob < 1:
            self.dropout = nn.Dropout(1 - self.keep_prob)
        self.pool = nn.AvgPool2d(kernel_size = 2)
        self.if_att = if_att
        if self.if_att == True:
            self.attention = attention(input_channels = self.out_channels)

    def forward(self, x):
        output = self.bn(x)
        output = F.relu(output, inplace=True)
        output = self.conv(output)
        if self.keep_prob < 1:
            output = self.dropout(output)
        if self.if_att==True:
            output = self.attention(output)
        output = self.pool(output)
        return output

class global_pool(nn.Module):
    def __init__(self, input_size, input_channels):
        super(global_pool, self).__init__()
        self.input_channels = input_channels
        self.bn = norm(self.input_channels)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        output = self.bn(x)
        output = F.relu(output, inplace=True)
        output = self.pool(output)
        return output
