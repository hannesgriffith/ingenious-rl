import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.cudnn.benchmark = True
print("torch.backends.cudnn.benchmark =", torch.backends.cudnn.benchmark)

def get_network(params):
    if params["network_type"] == "mlp":
        return MLP()
    elif params["network_type"] == "mlp2":
        return MLP2()
    elif params["network_type"] == "conv":
        return Conv()
    else:
        raise ValueError("Incorrect network name.")

input_channels = {"grid": None, "vector": 109}

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_input_channels = input_channels["vector"]
        self.num_hidden_units = 128

        self.mlp = nn.Sequential(
            nn.Linear(self.num_input_channels, self.num_hidden_units),
            Swish(),
            nn.Linear(self.num_hidden_units, self.num_hidden_units),
            Swish(),
            nn.Linear(self.num_hidden_units, 1),
            nn.Tanh()
        )

    def forward(self, x_grid, x_grid_vector, x_vector):
        return self.mlp(x_vector)

class MLP2(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_input_channels = input_channels["vector"]
        self.num_hidden_units = 256

        self.mlp = nn.Sequential(
            nn.Linear(self.num_input_channels, self.num_hidden_units),
            Swish(),
            nn.Linear(self.num_hidden_units, self.num_hidden_units),
            Swish(),
            nn.Linear(self.num_hidden_units, self.num_hidden_units),
            Swish(),
            nn.Linear(self.num_hidden_units, self.num_hidden_units),
            Swish(),
            nn.Linear(self.num_hidden_units, 1),
            nn.Tanh()
        )

    def forward(self, x_grid, x_grid_vector, x_vector):
        return self.mlp(x_vector)

def get_hexagonal_kernel_mask():
    return torch.tensor([
                [0., 1., 0., 1., 0.],
                [1., 0., 1., 0., 1.],
                [0., 1., 0., 1., 0.],
            ], dtype=torch.float16, requires_grad=False).view(1, 1, 3, 5)

def get_hexagonal_activations_mask():
    return torch.tensor([
                [0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0,],
                [0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0,],
                [0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0,],
                [0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0,],
                [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,],
                [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,],
                [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,],
                [0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0,],
                [0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0,],
                [0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0,],
                [0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0,]
            ], dtype=torch.float16, requires_grad=False).view(1, 1, 11, 21)

def swish(x):
    return x * torch.sigmoid(x)

def hexagonal_2d_global_avg_pool(x, mask):
    return torch.sum(x * mask, dim=(2, 3), keepdim=True) / torch.sum(mask)

class Hexagonal3x3Conv2dNoBias(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_mask):
        super().__init__(in_channels, out_channels, (3, 5), stride=1, padding=(1, 2), bias=False)
        self.kernel_mask = kernel_mask.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    def forward(self, x_in):
        return F.conv2d(x_in, self.weight * self.kernel_mask, bias=None, stride=1, padding=(1, 2))

class HexagonalBlock(nn.Module):
    def __init__(self, hidden_channels, activations_mask, kernel_mask):
        super().__init__()
        self.activations_mask = activations_mask
        self.conv_1 = Hexagonal3x3Conv2dNoBias(hidden_channels, hidden_channels, kernel_mask)
        self.conv_2 = Hexagonal3x3Conv2dNoBias(hidden_channels, hidden_channels, kernel_mask)
        self.bn_1 = nn.BatchNorm2d(hidden_channels, affine=True)
        self.bn_2 = nn.BatchNorm2d(hidden_channels, affine=True)

    def forward(self, x_in):
        x = swish(self.bn_1(self.conv_1(x_in))) * self.activations_mask
        x = swish(self.bn_2(self.conv_2(x)) + x_in) * self.activations_mask
        return x

class Conv(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_input_channels = ...
        self.num_blocks = 8
        self.num_hidden_units = 32

        self.activations_mask = get_hexagonal_activations_mask()
        self.activations_mask = self.activations_mask.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

        self.kernel_mask = get_hexagonal_kernel_mask()
        self.kernel_mask = self.kernel_mask.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

        self.conv_in = Hexagonal3x3Conv2dNoBias(self.num_input_channels, self.num_hidden_units, self.kernel_mask)
        self.bn_in = nn.BatchNorm2d(self.num_hidden_units, affine=True)

        self.res_stack = nn.Sequential(*[
            HexagonalBlock(
                self.num_hidden_units,
                self.activations_mask,
                self.kernel_mask
                ) for _ in range(self.num_blocks)
            ])

        self.avg_pool = hexagonal_2d_global_avg_pool
        self.max_pool = nn.MaxPool2d((11, 21), stride=1, padding=0)
        self.fc_out = nn.Linear(2 * self.num_hidden_units, 1, bias=True)
        self.tanh = nn.Tanh()

    def forward(self, x_grid, x_grid_vector, x_vector):
        x = swish(self.bn_in(self.conv_in(x_grid))) * self.activations_mask
        x = self.res_stack(x) * self.activations_mask

        x_max = self.max_pool(x)
        x_avg = self.avg_pool(x, self.activations_mask)
        x = torch.cat((x_avg, x_max), dim=1).squeeze()

        x = self.fc_out(x)
        x = self.tanh(x)
        return x