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
    elif params["network_type"] == "conv_v2":
        return ConvV2()
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

class MLP2Only(nn.Module):
    def __init__(self):
        super().__init__()
        _, self.f = num_input_channels()
        self.h = 64 # num linear hidden channels
        self.mlp2 = mlp2(self.f, self.h, 1, p=0.0)

    def forward(self, x_grid, x_vector):
        return self.mlp2(x_vector)

class ProcessInputs(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_grid, x_vector):
        b = x_vector.size()[0]
        num_unoccupied = (11 * 21 - torch.sum(x_grid[:, 6].view(b, -1), dim=1).view(b, 1)) / 231.
        num_available = torch.sum(x_grid[:, 7].view(b, -1), dim=1).view(b, 1) / 91.
        return torch.cat((x_vector, num_unoccupied, num_available), dim=1)

class MLPV1(nn.Module):
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

class MLPV2(nn.Module):
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

class CombineInputs(nn.Module):
    def __init__(self, num_vec):
        super().__init__()
        self.v = num_vec

    def forward(self, x_grid, x_vector):
        b = x_grid.size()[0]
        x_vector = x_vector.view(b, self.v, 1, 1).repeat(1, 1, 11, 21)
        return torch.cat((x_grid, x_vector), dim=1)

class ResBlockV1(nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv_1 = nn.Conv2d(hidden_channels, hidden_channels, (3, 5), padding=(1, 2), stride=1, bias=True)
        self.conv_2 = nn.Conv2d(hidden_channels, hidden_channels, (3, 5), padding=(1, 2), stride=1, bias=True)

    def forward(self, x_in):
        x = F.relu(self.conv_1(x_in))
        x = F.relu(self.conv_2(x) + x_in)
        return x

class ConvV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.g, self.v = num_input_channels()
        self.num_blocks = 2
        self.conv_h = 32

        blocks = [ResBlockV1(self.conv_h) for _ in range(self.num_blocks)]
        self.res_stack = nn.Sequential(*blocks)

        self.conv_in = nn.Conv2d(self.g + self.v, self.conv_h, (3, 5), padding=(1, 2), stride=1, bias=True)
        self.conv_out = nn.Conv2d(2 * self.conv_h, 1, 1, bias=True)

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

        x = self.conv_out(x).squeeze()
        x = self.tanh(x)
        return x

class Hexagonal3x3Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels, (3, 5), stride=1, padding=(1, 2), bias=True)
        self.mask = torch.tensor([
                                    [0., 1., 0., 1., 0.],
                                    [1., 0., 1., 0., 1.],
                                    [0., 1., 0., 1., 0.],
                                ], dtype=torch.float32, requires_grad=False).view(1, 1, 3, 5)
        self.mask = self.mask.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    def forward(self, x_in):
        return F.conv2d(x_in, self.weight * self.mask, bias=self.bias, stride=1, padding=(1, 2))

class ResBlockV2(nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.hex_conv_1 = Hexagonal3x3Conv2d(hidden_channels, hidden_channels)
        self.hex_conv_2 = Hexagonal3x3Conv2d(hidden_channels, hidden_channels)

    def forward(self, x_in):
        x = F.relu(self.hex_conv_1(x_in))
        x = F.relu(self.hex_conv_2(x) + x_in)
        return x

class ConvV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.g, self.v = num_input_channels()
        self.num_blocks = 2
        self.conv_h = 32

        blocks = [ResBlockV2(self.conv_h) for _ in range(self.num_blocks)]
        self.res_stack = nn.Sequential(*blocks)

        self.conv_in = Hexagonal3x3Conv2d(self.g + self.v, self.conv_h)
        self.conv_out = nn.Conv2d(2 * self.conv_h, 1, 1, stride=1, padding=0, bias=True)

        self.combine_inputs = CombineInputs(self.v)
        self.avg_pool = nn.AvgPool2d((11, 21), stride=1, padding=0)
        self.max_pool = nn.MaxPool2d((11, 21), stride=1, padding=0)
        self.tanh = nn.Tanh()

        self.mask = torch.tensor([
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
            ], dtype=torch.float32, requires_grad=False).view(1, 1, 11, 21)
        self.mask = self.mask.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    def forward(self, x_grid, x_vector):
        x_in = self.combine_inputs(x_grid, x_vector) * self.mask
        x = F.relu(self.conv_in(x_in))
        x = self.res_stack(x)

        x *= self.mask
        x_avg = self.avg_pool(x)
        x_max = self.max_pool(x)
        x = torch.cat((x_avg, x_max), dim=1)

        x = self.conv_out(x).squeeze()
        x = self.tanh(x)
        return x

class Hexagonal5x5Conv2dConnected(nn.Conv2d):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels, (5, 9), stride=1, padding=(2, 4), bias=True)
        self.mask = torch.tensor([
                                    [0., 0., 1., 0., 0., 0., 1., 0., 0.],
                                    [0., 0., 0., 1., 0., 1., 0., 0., 0.],
                                    [1., 0., 1., 0., 1., 0., 1., 0., 1.],
                                    [0., 0., 0., 1., 0., 1., 0., 0., 0.],
                                    [0., 0., 1., 0., 0., 0., 1., 0., 0.],
                                ], dtype=torch.float32, requires_grad=False).view(1, 1, 5, 9)
        self.mask = self.mask.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    def forward(self, x_in):
        return F.conv2d(x_in, self.weight * self.mask, bias=self.bias, stride=1, padding=(2, 4))

class Hexagonal5x5Conv2dAll(nn.Conv2d):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels, (5, 9), stride=1, padding=(2, 4), bias=True)
        self.mask = torch.tensor([
                                    [1., 0., 1., 0., 1., 0., 1., 0., 1.],
                                    [0., 1., 0., 1., 0., 1., 0., 1., 0.],
                                    [1., 0., 1., 0., 1., 0., 1., 0., 1.],
                                    [0., 1., 0., 1., 0., 1., 0., 1., 0.],
                                    [1., 0., 1., 0., 1., 0., 1., 0., 1.],
                                ], dtype=torch.float32, requires_grad=False).view(1, 1, 5, 9)
        self.mask = self.mask.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    def forward(self, x_in):
        return F.conv2d(x_in, self.weight * self.mask, bias=self.bias, stride=1, padding=(2, 4))

class Hexagonal7x7Conv2dConnected(nn.Conv2d):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels, (7, 13), stride=1, padding=(3, 6), bias=True)
        self.mask = torch.tensor([
                                    [0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
                                    [0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0.],
                                    [1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1.],
                                    [0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0.],
                                    [0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
                                ], dtype=torch.float32, requires_grad=False).view(1, 1, 7, 13)
        self.mask = self.mask.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    def forward(self, x_in):
        return F.conv2d(x_in, self.weight * self.mask, bias=self.bias, stride=1, padding=(3, 6))

class Hexagonal7x7Conv2dAll(nn.Conv2d):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels, (7, 13), stride=1, padding=(3, 6), bias=True)
        self.mask = torch.tensor([
                                    [0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0.],
                                    [1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1.],
                                    [0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0.],
                                    [1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1.],
                                    [0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0.],
                                    [1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1.],
                                    [0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0.],
                                ], dtype=torch.float32, requires_grad=False).view(1, 1, 7, 13)
        self.mask = self.mask.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    def forward(self, x_in):
        return F.conv2d(x_in, self.weight * self.mask, bias=self.bias, stride=1, padding=(3, 6))

class IngeniousBlock(nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.h_outer = hidden_channels
        self.h_inner = hidden_channels // 8
        self.conv_1d = nn.Conv2d(self.h_outer, self.h_outer, 1, bias=True)
        self.conv_3x3 = Hexagonal3x3Conv2d(self.h_inner, self.h_inner)
        self.conv_5x5_connected = Hexagonal5x5Conv2dConnected(self.h_inner, self.h_inner)
        self.conv_5x5_all = Hexagonal5x5Conv2dAll(self.h_inner, self.h_inner)
        self.conv_7x7_connected = Hexagonal7x7Conv2dConnected(self.h_inner, self.h_inner)
        self.conv_7x7_all = Hexagonal7x7Conv2dAll(self.h_inner, self.h_inner)
        self.avg_pool = nn.AvgPool2d((11, 21), stride=1, padding=0)
        self.max_pool = nn.MaxPool2d((11, 21), stride=1, padding=0)

    def forward(self, x_in):
        b = x_vector.size()[0]
        x = F.relu(self.conv_1d(x_in))

        x1 = self.conv_1d(x[0 : self.h_inner])
        x2 = self.conv_3x3(x[self.h_inner : 2 * self.h_inner])
        x3 = self.conv_5x5_connected(x[2 * self.h_inner : 3 * self.h_inner])
        x4 = self.conv_5x5_all(x[3 * self.h_inner : 4 * self.h_inner])
        x5 = self.conv_7x7_connected(x[4 * self.h_inner : 5 * self.h_inner])
        x6 = self.conv_7x7_all(x[5 * self.h_inner : 6 * self.h_inner])

        x7 = self.avg_pool(x[6 * self.h_inner : 7 * self.h_inner])
        x7 = x_vector.view(b, x7, 1, 1).repeat(1, 1, 11, 21)

        x8 = self.max_pool(x[7 * self.h_inner : 8 * self.h_inner])
        x8 = x_vector.view(b, x8, 1, 1).repeat(1, 1, 11, 21)

        x = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8), dim=1)
        return F.relu(x + x_in)

class ConvV3(nn.Module):
    def __init__(self):
        super().__init__()
        self.g, self.v = num_input_channels()
        self.num_blocks = 4
        self.conv_h = 64

        blocks = [IngeniousBlock(self.conv_h) for _ in range(self.num_blocks)]
        self.ingenious_stack = nn.Sequential(*blocks)

        self.conv_in = nn.Conv2d(self.g + self.v, self.conv_h, 1, padding=0, bias=True)
        self.conv_out = nn.Conv2d(2 * self.conv_h, 1, 1, stride=1, padding=0, bias=True)

        self.combine_inputs = CombineInputs(self.v)
        self.avg_pool = nn.AvgPool2d((11, 21), stride=1, padding=0)
        self.max_pool = nn.MaxPool2d((11, 21), stride=1, padding=0)
        self.tanh = nn.Tanh()

        self.mask = torch.tensor([
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
            ], dtype=torch.float32, requires_grad=False).view(1, 1, 11, 21)
        self.mask = self.mask.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    def forward(self, x_grid, x_vector):
        x_in = self.combine_inputs(x_grid, x_vector) * self.mask
        x = F.relu(self.conv_in(x_in))
        x = self.ingenious_stack(x)

        x *= self.mask
        x_avg = self.avg_pool(x)
        x_max = self.max_pool(x)
        x = torch.cat((x_avg, x_max), dim=1)

        x = self.conv_out(x).squeeze()
        x = self.tanh(x)
        return x