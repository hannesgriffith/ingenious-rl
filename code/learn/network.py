import torch
import torch.nn as nn
import torch.nn.functional as F

def get_network(params):
    if params["network_type"] == "mlp":
        return MLP()
    elif params["network_type"] == "conv_v1":
        return ConvV1()
    elif params["network_type"] == "conv_v2":
        return ConvV2()
    else:
        raise ValueError("Incorrect network name.")

def num_input_channels():
    g = 8  # num grid input channels
    v = 29 # num vector input channels
    return g, v

class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        _, self.v = num_input_channels()
        self.h = 32

        self.mlp = nn.Sequential(
            nn.Linear(self.v, self.h),
            nn.ReLU(),
            nn.Linear(self.h, self.h),
            nn.ReLU(),
            nn.Linear(self.h, 1),
            nn.Sigmoid()
        )

    def forward(self, x_grid, x_vector):
        return self.mlp(x_vector)

class ResBlockV1(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(ResBlockV1, self).__init__()
        self.conv_1 = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, stride=1, bias=True)
        self.conv_2 = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, stride=1, bias=True)

    def forward(self, x_in):
        x = F.relu(self.conv_1(x_in))
        x = F.relu(self.conv_2(x) + x_in)
        return x

class ConvV1(torch.nn.Module):
    def __init__(self):
        super(ConvV1, self).__init__()
        self.g, self.v = num_input_channels()
        self.num_blocks = 2
        self.conv_h = 32

        blocks = [ResBlockV1(self.conv_h) for _ in range(self.num_blocks)]
        self.res_stack = nn.Sequential(*blocks)

        self.conv_in = nn.Conv2d(self.g + self.f, self.conv_h, 3, padding=1, stride=1, bias=True)
        self.avg_pool = nn.AvgPool2d((11, 11), stride=1, padding=0)
        self.max_pool = nn.MaxPool2d((11, 11), stride=1, padding=0)
        self.conv_1d = nn.Conv2d(self.conv_h, self.o, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def combine_inputs(self, x_grid, x_vector):
        b = x_grid.size()[0]
        x_vector = x_vector.view(b, self.f, 1, 1).repeat(1, 1, 11, 21)
        return torch.cat((x_grid, x_vector), dim=1)

    def forward(self, x_grid, x_vector):
        x_in = self.combine_inputs(x_grid, x_vector)
        x = F.relu(self.conv_in(x_in))
        x = self.res_stack(x)

        x_avg = self.avg_pool(x)
        x_max = self.max_pool(x)
        x = torch.cat((x_avg, x_max), dim=1)

        x = self.conv_1d(x).squeeze()
        x = self.sigmoid(x)
        return x

class ResBlockV2(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(ResBlockV1, self).__init__()
        self.conv_1 = nn.Conv2d(hidden_channels, hidden_channels, (3, 5), padding=(1, 2), stride=1, bias=True)
        self.conv_2 = nn.Conv2d(hidden_channels, hidden_channels, (3, 5), padding=(1, 2), stride=1, bias=True)

    def forward(self, x_in):
        x = F.relu(self.conv_1(x_in))
        x = F.relu(self.conv_2(x) + x_in)
        return x

class ConvV2(torch.nn.Module):
    def __init__(self):
        super(ConvV2, self).__init__()
        self.g, self.v = num_input_channels()
        self.num_blocks = 2
        self.conv_h = 32

        blocks = [ResBlockV2(self.conv_h) for _ in range(self.num_blocks)]
        self.res_stack = nn.Sequential(*blocks)

        self.conv_in = nn.Conv2d(self.g + self.f, self.conv_h, (3, 5), padding=(1, 2), stride=1, bias=True)
        self.avg_pool = nn.AvgPool2d((11, 21), stride=1, padding=0)
        self.max_pool = nn.MaxPool2d((11, 21), stride=1, padding=0)
        self.conv_1d = nn.Conv2d(self.conv_h, self.o, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def combine_inputs(self, x_grid, x_vector):
        b = x_grid.size()[0]
        x_vector = x_vector.view(b, self.f, 1, 1).repeat(1, 1, 11, 21)
        return torch.cat((x_grid, x_vector), dim=1)

    def forward(self, x_grid, x_vector):
        x_in = self.combine_inputs(x_grid, x_vector)
        x = F.relu(self.conv_in(x_in))
        x = self.res_stack(x)

        x_avg = self.avg_pool(x)
        x_max = self.max_pool(x)
        x = torch.cat((x_avg, x_max), dim=1)

        x = self.conv_1d(x).squeeze()
        x = self.sigmoid(x)
        return x
