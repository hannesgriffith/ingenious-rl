import torch
import torch.nn as nn
import torch.nn.functional as F

def get_network(params):
    if params["network_type"] == "mlp_v1":
        return MLPV1()
    elif params["network_type"] == "mlp_v2":
        return MLPV2()
    elif params["network_type"] == "mlp2_only":
        return MLP2Only()
    elif params["network_type"] == "conv_v1":
        return ConvV1()
    else:
        raise ValueError("Incorrect network name.")

def num_input_channels():
    g = 10  # num grid input channels
    v = 29  # num vector input channels
    return g, v

def mlp2(in_channels, h_channels, out_channels, p=0.5):
    return nn.Sequential(
        nn.Linear(in_channels, h_channels),
        nn.Dropout(p=p),
        nn.ReLU(),
        nn.Linear(h_channels, h_channels // 2),
        nn.Dropout(p=p),
        nn.ReLU(),
        nn.Linear(h_channels // 2, out_channels),
        nn.Sigmoid()
    )

class MLP2Only(torch.nn.Module):
    def __init__(self):
        super().__init__()
        _, self.f = num_input_channels()
        self.h = 64 # num linear hidden channels
        self.mlp2 = mlp2(self.f, self.h, 1, p=0.0)

    def forward(self, x_grid, x_vector):
        return self.mlp2(x_vector)

class ProcessInputs(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_grid, x_vector):
        b = x_vector.size()[0]
        num_unoccupied = (11 * 21 - torch.sum(x_grid[:, 6].view(b, -1), dim=1).view(b, 1)) / 231.
        num_available = torch.sum(x_grid[:, 7].view(b, -1), dim=1).view(b, 1) / 91.
        return torch.cat((x_vector, num_unoccupied, num_available), dim=1)

class MLPV1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        _, self.v = num_input_channels()
        self.h = 32

        self.process_inputs = ProcessInputs()
        self.mlp = nn.Sequential(
            nn.Linear(self.v + 2, self.h),
            nn.ReLU(),
            nn.Linear(self.h, 1),
            nn.Tanh()
        )

    def forward(self, x_grid, x_vector):
        x_in = self.process_inputs(x_grid, x_vector)
        return self.mlp(x_in)

class MLPV2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        _, self.v = num_input_channels()
        self.h = 32

        self.process_inputs = ProcessInputs()
        self.mlp = nn.Sequential(
            nn.Linear(self.v + 2, self.h),
            nn.ReLU(),
            nn.Linear(self.h, self.h),
            nn.ReLU(),
            nn.Linear(self.h, 1),
            nn.Tanh()
        )

    def forward(self, x_grid, x_vector):
        x_in = self.process_inputs(x_grid, x_vector)
        return self.mlp(x_in)

class CombineInputs(torch.nn.Module):
    def __init__(self, num_vec):
        super().__init__()
        self.v = num_vec

    def forward(self, x_grid, x_vector):
        b = x_grid.size()[0]
        x_vector = x_vector.view(b, self.v, 1, 1).repeat(1, 1, 11, 21)
        return torch.cat((x_grid, x_vector), dim=1)

class ResBlockV1(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv_1 = nn.Conv2d(hidden_channels, hidden_channels, (3, 5), padding=(1, 2), stride=1, bias=True)
        self.conv_2 = nn.Conv2d(hidden_channels, hidden_channels, (3, 5), padding=(1, 2), stride=1, bias=True)

    def forward(self, x_in):
        x = F.relu(self.conv_1(x_in))
        x = F.relu(self.conv_2(x) + x_in)
        return x

class ConvV1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.g, self.v = num_input_channels()
        self.num_blocks = 2
        self.conv_h = 32

        blocks = [ResBlockV1(self.conv_h) for _ in range(self.num_blocks)]
        self.res_stack = nn.Sequential(*blocks)

        self.conv_1d = nn.Conv2d(2 * self.conv_h, 1, 1, bias=True)
        self.conv_in = nn.Conv2d(self.g + self.v, self.conv_h, (3, 5), padding=(1, 2), stride=1, bias=True)

        self.combine_inputs = CombineInputs(self.v)
        self.avg_pool = nn.AvgPool2d((11, 21), stride=1, padding=0)
        self.max_pool = nn.MaxPool2d((11, 21), stride=1, padding=0)
        self.tanh = nn.Tanh()

    def forward(self, x_grid, x_vector):
        x_in = self.combine_inputs(x_grid, x_vector)
        x = F.relu(self.conv_in(x_in))
        x = self.res_stack(x)

        x_avg = self.avg_pool(x)
        x_max = self.max_pool(x)
        x = torch.cat((x_avg, x_max), dim=1)

        x = self.conv_1d(x).squeeze()
        x = self.tanh(x)
        return x

# Add conv v2, similar to above but with batch norm and more layers
