import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.cudnn.benchmark = True
print("torch.backends.cudnn.benchmark =", torch.backends.cudnn.benchmark)

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
    elif params["network_type"] == "conv_v2_plus":
        return ConvV2Plus()
    elif params["network_type"] == "conv_v3":
        return ConvV3()
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

def get_hexagonal_kernel_mask():
    return torch.tensor([
                [0., 1., 0., 1., 0.],
                [1., 0., 1., 0., 1.],
                [0., 1., 0., 1., 0.],
            ], dtype=torch.float32, requires_grad=False).view(1, 1, 3, 5)

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
            ], dtype=torch.float32, requires_grad=False).view(1, 1, 11, 21)

class Hexagonal3x3Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, mask=None):
        super().__init__(in_channels, out_channels, (3, 5), stride=1, padding=(1, 2), bias=True)
        self.mask = mask if mask is not None else get_hexagonal_kernel_mask()
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

        self.mask = get_hexagonal_activations_mask()
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

class ConvV2Plus(nn.Module):
    def __init__(self):
        super().__init__()
        self.g, self.v = num_input_channels()
        self.num_blocks = 4
        self.conv_h = 48

        blocks = [ResBlockV2(self.conv_h) for _ in range(self.num_blocks)]
        self.res_stack = nn.Sequential(*blocks)

        self.conv_in = nn.Conv2d(self.g + self.v, self.conv_h, 1, stride=1, padding=0, bias=True)
        self.conv_out = nn.Conv2d(2 * self.conv_h, 1, 1, stride=1, padding=0, bias=True)

        self.combine_inputs = CombineInputs(self.v)
        self.avg_pool = nn.AvgPool2d((11, 21), stride=1, padding=0)
        self.max_pool = nn.MaxPool2d((11, 21), stride=1, padding=0)
        self.tanh = nn.Tanh()

        self.mask = get_hexagonal_activations_mask()
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

def swish(x):
    return x * torch.sigmoid(x)

def hexagonal_2d_global_avg_pool(x, mask):
    return torch.sum(x * mask, dim=(2, 3), keepdim=True) / torch.sum(mask)

class HexagonalDepthwise3x3Conv2d(nn.Conv2d):
    def __init__(self, num_channels, kernel_mask):
        super().__init__(num_channels, num_channels, (3, 5), stride=1, padding=(1, 2), groups=num_channels, bias=True)
        self.kernel_mask = kernel_mask
        self.num_channels = num_channels

    def forward(self, x_in):
        return F.conv2d(x_in, self.weight * self.kernel_mask, bias=self.bias, stride=1, padding=(1, 2), groups=self.num_channels)

class HexagonalSEBlock(nn.Module):
    def __init__(self, hidden_units, activations_mask):
        super().__init__()
        self.squeeze_ratio = 16
        self.hidden_units = hidden_units
        self.activations_mask = activations_mask

        self.avg_pool = hexagonal_2d_global_avg_pool
        self.conv1 = nn.Conv2d(self.hidden_units, self.hidden_units // self.squeeze_ratio, kernel_size=1)
        self.conv2 = nn.Conv2d(self.hidden_units // self.squeeze_ratio, self.hidden_units, kernel_size=1)

    def forward(self, x_in):
        x = self.avg_pool(x_in, self.activations_mask)
        x = swish(self.conv1(x))
        s = torch.sigmoid(self.conv2(x))
        return s * x_in

class HexagonalMBConvSEBlock(nn.Module):
    def __init__(self, hidden_units, activations_mask, kernel_mask):
        super().__init__()
        self.expansion_ratio = 6
        self.hidden_units = hidden_units
        self.activations_mask = activations_mask

        self.expansion_conv = nn.Conv2d(self.hidden_units, self.hidden_units * self.expansion_ratio, kernel_size=1)
        self.depthwise_conv = HexagonalDepthwise3x3Conv2d(self.hidden_units * self.expansion_ratio, kernel_mask)
        self.pointwise_conv = nn.Conv2d(self.hidden_units * self.expansion_ratio, self.hidden_units, kernel_size=1)
        self.se_block = HexagonalSEBlock(self.hidden_units, self.activations_mask)

        self.dropout_1 = nn.Dropout2d(0.2)
        self.dropout_2 = nn.Dropout2d(0.2)
        self.dropout_3 = nn.Dropout2d(0.2)

    def forward(self, x_in):
        x = swish(self.dropout_1(self.expansion_conv(x_in)) * self.activations_mask)
        x = swish(self.dropout_2(self.depthwise_conv(x)) * self.activations_mask)
        x = self.dropout_3(self.se_block(self.pointwise_conv(x)))
        return x_in + x

class ConvV3(nn.Module):
    def __init__(self):
        super().__init__()
        self.g, self.v = num_input_channels()
        self.num_blocks = 10
        self.num_hidden_units = 64

        self.activations_mask = get_hexagonal_activations_mask()
        self.activations_mask = self.activations_mask.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

        self.kernel_mask = get_hexagonal_kernel_mask()
        self.kernel_mask = self.kernel_mask.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

        self.mbconv_stack = nn.Sequential(*[
            HexagonalMBConvSEBlock(
                self.num_hidden_units,
                self.activations_mask,
                self.kernel_mask
            ) for _ in range(self.num_blocks)
        ])

        self.conv_in = Hexagonal3x3Conv2d(self.g + self.v, self.num_hidden_units, mask=self.kernel_mask)
        self.fc_out = nn.Linear(2 * self.num_hidden_units, 1, bias=True)

        self.combine_inputs = CombineInputs(self.v)
        self.avg_pool = hexagonal_2d_global_avg_pool
        self.max_pool = nn.MaxPool2d((11, 21), stride=1, padding=0)

        self.dropout_in = nn.Dropout2d(0.2)
        self.dropout_out = nn.Dropout(0.2)
        self.tanh = nn.Tanh()

    def forward(self, x_grid, x_vector):
        x_in = self.combine_inputs(x_grid, x_vector) * self.activations_mask
        x = swish(self.dropout_in(self.conv_in(x_in)) * self.activations_mask)
        x = self.mbconv_stack(x)

        x *= self.activations_mask
        x_max = self.max_pool(x)
        x_avg = self.avg_pool(x, self.activations_mask)
        x = torch.cat((x_avg, x_max), dim=1).squeeze()

        x = self.fc_out(self.dropout_out(x))
        x = self.tanh(x)
        return x