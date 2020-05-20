import torch
import torch.nn as nn
import torch.nn.functional as F

def get_network(params):
    if params["network_type"] == "mlp_only":
        return MLPOnly()
    elif params["network_type"] == "mlp2_only":
        return MLP2Only()
    elif params["network_type"] == "mlp2_only_extra_features":
        return MLP2OnlyExtraFeatures()
    elif params["network_type"] == "3_hidden_layer":
        return ThreeLayerNetwork()
    elif params["network_type"] == "5_hidden_layer":
        return FiveLayerNetwork()
    elif params["network_type"] == "5_block":
        return NBlockResNet(5)
    elif params["network_type"] == "10_block":
        return NBlockResNet(10)
    elif params["network_type"] == "resnet_variant":
        return ResNetVariant()
    elif params["network_type"] == "simple_convnet":
        return SimpleConvNet()
    elif params["network_type"] == "simple_convnet_fc":
        return SimpleConvNetFullyConvolutional()
    elif params["network_type"] == "debug":
        return Debug()
    elif params["network_type"] == "debug2":
        return Debug2()
    elif params["network_type"] == "debug3":
        return Debug3()
    elif params["network_type"] == "smaller":
        return Smaller()
    else:
        raise ValueError("Incorrect network name.")

def get_network_params():
    g = 8  # num grid input channels
    i = 11 # grid input width/height
    f = 29 # num vector input channels
    o = 1  # num outputs
    return g, i, f, o

def get_network_params_extra_features():
    g = 8  # num grid input channels
    i = 11 # grid input width/height
    f = 35 # num vector input channels
    o = 1  # num outputs
    return g, i, f, o

def conv2d_dropout_relu(in_channels, out_channels, k, padding=0, p=0.5):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, k, padding=padding),
        nn.Dropout(p=p),
        nn.ReLU()
        )

def conv2d_bn2d(in_channels, out_channels, k, padding=0, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, k, padding=padding, stride=stride),
        nn.BatchNorm2d(out_channels)
        )

def mlp(in_channels, h_channels, out_channels, p=0.5):
    return nn.Sequential(
        nn.Linear(in_channels, h_channels),
        nn.Dropout(p=p),
        nn.ReLU(),
        nn.Linear(h_channels, out_channels),
        nn.Sigmoid()
    )

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

def mlp3(in_channels, h1_channels, h2_channels, out_channels):
    return nn.Sequential(
        nn.Linear(in_channels, h1_channels),
        nn.ReLU(),
        nn.Linear(h1_channels, h2_channels),
        nn.ReLU(),
        nn.Linear(h2_channels, out_channels),
        nn.Sigmoid()
    )

class MLPOnly(torch.nn.Module):
    def __init__(self):
        super(MLPOnly, self).__init__()
        _, _, self.f, self.o = get_network_params()
        self.h = 64 # num linear hidden channels
        self.mlp = mlp(self.f, self.h, self.o, p=0.0)

    def forward(self, x_grid, x_vector):
        return self.mlp(x_vector)

class MLP2Only(torch.nn.Module):
    def __init__(self):
        super(MLP2Only, self).__init__()
        _, _, self.f, self.o = get_network_params()
        self.h = 64 # num linear hidden channels
        self.mlp2 = mlp2(self.f, self.h, self.o, p=0.0)

    def forward(self, x_grid, x_vector):
        return self.mlp2(x_vector)

class MLP2OnlyExtraFeatures(torch.nn.Module):
    def __init__(self):
        super(MLP2OnlyExtraFeatures, self).__init__()
        _, _, self.f, self.o = get_network_params_extra_features()
        self.h = 64 # num linear hidden channels
        self.mlp2 = mlp2(self.f, self.h, self.o, p=0.0)

    def forward(self, x_grid, x_vector):
        return self.mlp2(x_vector)

class ThreeLayerNetwork(torch.nn.Module):
    def __init__(self):
        super(ThreeLayerNetwork, self).__init__()
        self.g, self.i, self.f, self.o = get_network_params()
        self.h1 = 64 # num conv hidden channels
        self.h2 = 64 # num linear hidden channels

        self.conv_layers = nn.Sequential(
            nn.conv2d_dropout_relu(self.g, self.h1, 3, padding=1, p=0.5),
            nn.conv2d_dropout_relu(self.h1, self.h1, 3, padding=1, p=0.5),
            nn.conv2d_dropout_relu(self.h1, self.h1, 3, padding=1, p=0.5),
        )

        self.avg_pool = nn.AvgPool2d(self.i, stride=1, padding=0)
        self.mlp = mlp2(self.h1 + self.f, self.h2, self.o, p=0.0)

    def forward(self, x_grid, x_vector):
        x = self.self.conv_layers(x_grid)
        x = self.avg_pool(x)
        x = torch.cat((x.squeeze(), x_vector), dim=1)
        x = self.mlp(x)
        return x

class FiveLayerNetwork(torch.nn.Module):
    def __init__(self):
        super(FiveLayerNetwork, self).__init__()
        self.g, _, self.f, self.o = get_network_params()
        self.h1 = 16 # num conv hidden channels
        self.h2 = 128 # num linear hidden channels

        self.conv_layers = nn.Sequential(
            nn.conv2d_dropout_relu(self.g, self.h1, 3),
            nn.conv2d_dropout_relu(self.h1 * 1, self.h1 * 2, 3),
            nn.conv2d_dropout_relu(self.h1 * 2, self.h1 * 4, 3),
            nn.conv2d_dropout_relu(self.h1 * 4, self.h1 * 8, 3),
            nn.conv2d_dropout_relu(self.h1 * 8, self.h1 * 16, 3)
        )

        self.mlp = mlp2(self.h1 * 16 + self.f, self.h2, self.o, p=0.0)

    def forward(self, x_grid, x_vector):
        x = self.conv_layers(x_grid)
        x = torch.cat((x.squeeze(), x_vector), dim=1)
        x = self.mlp(x)
        return x

class ResNetBottleneckBlock(torch.nn.Module):
    def __init__(self, channels_outer, channels_inner):
        super(ResNetBottleneckBlock, self).__init__()

        self.bn_conv_1 = conv2d_bn2d(channels_outer, channels_inner, 1, padding=0)
        self.bn_conv_2 = conv2d_bn2d(channels_inner, channels_inner, 3, padding=1)
        self.bn_conv_3 = conv2d_bn2d(channels_inner, channels_outer, 1, padding=0)

    def forward(self, x_in):
        x = F.relu(self.bn_conv_1(x_in))
        x = F.relu(self.bn_conv_2(x))
        x = F.relu(self.bn_conv_3(x) + x_in)
        return x

class NBlockResNet():
    def __init__(self, N):
        super(NBlockResNet, self).__init__()
        self.N = N
        self.g, self.i, self.f, self.o = get_network_params()
        self.outer = 128    # num resnet block outer channels
        self.inner = 32     # num resnet block inner channels
        self.h = 128        # num units in mlp hidden layer

        blocks = [ResNetBottleneckBlock(self.outer, self.inner) for _ in range(self.N)]
        self.bottleneck_layers = nn.Sequential(*blocks)

        self.conv_bn = conv2d_bn2d(self.g, self.outer, 1, padding=0)
        self.mlp = mlp2(self.outer + self.f, self.h, self.o, p=0.0)
        self.avg_pool = nn.AvgPool2d(self.i, stride=1, padding=0)

    def forward(self, x_grid, x_vector):
        x = F.relu(self.conv_bn(x_grid))
        x = self.bottleneck_layers(x)
        x = self.avg_pool(x)
        x = torch.cat((x.squeeze(), x_vector), dim=1)
        x = self.mlp(x)
        return x

class ResNetBottleneckBlockDownsample(torch.nn.Module):
    def __init__(self, channels_outer_in, channels_inner, channels_outer_out):
        super(ResNetBottleneckBlockDownsample, self).__init__()

        self.bn_conv_1 = conv2d_bn2d(channels_outer_in, channels_inner, 1, padding=0)
        self.bn_conv_2 = conv2d_bn2d(channels_inner, channels_inner, 3, padding=0)
        self.bn_conv_3 = conv2d_bn2d(channels_inner, channels_outer_out, 1, padding=0)

        self.avg_pool = nn.AvgPool2d(3, stride=1, padding=0)
        self.bn_conv_4 = conv2d_bn2d(channels_outer_in, channels_outer_out, 1, padding=0)

    def forward(self, x_in):
        x_in_downscaled = self.bn_conv_4(self.avg_pool(x_in))
        x = F.relu(self.bn_conv_1(x_in))
        x = F.relu(self.bn_conv_2(x))
        x = F.relu(self.bn_conv_3(x) + x_in_downscaled)
        return x

class ResNetVariant(torch.nn.Module):
    def __init__(self):
        super(ResNetVariant, self).__init__()
        self.g, self.i, self.f, self.o = get_network_params()

        self.bottleneck_stack = nn.Sequential(
            ResNetBottleneckBlock(32, 8), # 11
            ResNetBottleneckBlock(32, 8), # 11
            ResNetBottleneckBlockDownsample(32, 8, 64), # 9
            ResNetBottleneckBlock(64, 16), # 9
            ResNetBottleneckBlock(64, 16), # 9
            ResNetBottleneckBlockDownsample(64, 16, 128), # 7
            ResNetBottleneckBlock(128, 32), # 7
            ResNetBottleneckBlock(128, 32), # 7
            ResNetBottleneckBlockDownsample(128, 32, 256), # 5
            ResNetBottleneckBlock(256, 64), # 5
            ResNetBottleneckBlock(256, 64), # 5
        )

        self.conv_bn = conv2d_bn2d(self.g, 32, 1, padding=0)
        self.avg_pool = nn.AvgPool2d(5, stride=1, padding=0)
        self.mlp = mlp3(256 + self.f, 184, 32, self.o)

        # self.bottleneck_stack = nn.Sequential(
        #     ResNetBottleneckBlock(64, 16), # 11
        #     ResNetBottleneckBlock(64, 16), # 11
        #     ResNetBottleneckBlock(64, 16), # 11
        #     ResNetBottleneckBlockDownsample(64, 16, 128), # 9
        #     ResNetBottleneckBlock(128, 32), # 9
        #     ResNetBottleneckBlock(128, 32), # 9
        #     ResNetBottleneckBlock(128, 32), # 9
        # )

        # self.conv_bn = conv2d_bn2d(self.g, 64, 1, padding=0)
        # self.avg_pool = nn.AvgPool2d(9, stride=1, padding=0)
        # self.mlp = mlp3(128 + self.f, 128, 32, self.o)

    def forward(self, x_grid, x_vector):
        x = F.relu(self.conv_bn(x_grid))
        x = self.bottleneck_stack(x)
        x = self.avg_pool(x)
        x = torch.cat((x.squeeze(), x_vector), dim=1)
        x = self.mlp(x)
        return x

class ResNetVariantFullyConnected(torch.nn.Module):
    def __init__(self):
        super(ResNetVariant, self).__init__()
        pass

class SimpleConvNet(torch.nn.Module):
    def __init__(self):
        super(SimpleConvNet, self).__init__()
        self.g, self.i, self.f, self.o = get_network_params()
        self.conv_h = 32

        # self.conv_stack = nn.Sequential(
        #     # nn.Conv2d(self.g, self.conv_h, 1, padding=0, stride=1), nn.ReLU(),
        #     nn.Conv2d(self.conv_h, self.conv_h, 3, padding=1, stride=1), nn.ReLU(),
        #     nn.Conv2d(self.conv_h, self.conv_h, 3, padding=1, stride=1), nn.ReLU(),
        #     nn.Conv2d(self.conv_h, self.conv_h, 3, padding=1, stride=1), nn.ReLU(),
        # )

        self.conv_1 = nn.Conv2d(self.g, self.conv_h, 3, padding=1, stride=1)
        self.conv_2 = nn.Conv2d(self.conv_h, self.conv_h, 3, padding=1, stride=1)
        self.conv_3 = nn.Conv2d(self.conv_h, self.conv_h, 3, padding=1, stride=1)
        self.conv_4 = nn.Conv2d(self.conv_h, self.conv_h, 3, padding=1, stride=1)
        self.conv_5 = nn.Conv2d(self.conv_h, self.conv_h, 3, padding=1, stride=1)

        self.avg_pool = nn.AvgPool2d(self.i, stride=1, padding=0)
        self.mlp = mlp3(self.conv_h + self.f, 64, 32, self.o)

    def forward(self, x_grid, x_vector):
        x0 = F.relu(self.conv_1(x_grid))
        x1 = F.relu(self.conv_2(x0))
        x2 = F.relu(self.conv_3(x1) + x0)
        x3 = F.relu(self.conv_4(x2))
        x4 = F.relu(self.conv_5(x3) + x2)

        x = self.avg_pool(x4)
        x = torch.cat((x.squeeze(), x_vector), dim=1)
        x = self.mlp(x)
        return x

class SimpleConvNetFullyConvolutional(torch.nn.Module):
    def __init__(self):
        super(SimpleConvNetFullyConvolutional, self).__init__()
        self.g, self.i, self.f, self.o = get_network_params()
        self.conv_h = 32

        self.conv_1 = nn.Conv2d(self.g + self.f, self.conv_h, 3, padding=1, stride=1)
        self.conv_2 = nn.Conv2d(self.conv_h, self.conv_h, 3, padding=1, stride=1)
        self.conv_3 = nn.Conv2d(self.conv_h, self.conv_h, 3, padding=1, stride=1)
        self.conv_4 = nn.Conv2d(self.conv_h, self.conv_h, 3, padding=1, stride=1)
        self.conv_5 = nn.Conv2d(self.conv_h, self.conv_h, 3, padding=1, stride=1)

        self.avg_pool = nn.AvgPool2d(self.i, stride=1, padding=0)
        self.conv_6 = nn.Conv2d(self.conv_h, self.o, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_grid, x_vector):
        b = x_grid.size()[0]
        x_vector = x_vector.view(b, self.f, 1, 1).repeat(1, 1, self.i, self.i)

        x_in = torch.cat((x_grid, x_vector), dim=1)
        x0 = F.relu(self.conv_1(x_in))
        x1 = F.relu(self.conv_2(x0))
        x2 = F.relu(self.conv_3(x1) + x0)
        x3 = F.relu(self.conv_4(x2))
        x4 = F.relu(self.conv_5(x3) + x2)

        x = self.avg_pool(x4)
        x = self.conv_6(x).squeeze()
        x = self.sigmoid(x)
        return x

class ResBlock1(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(ResBlock1, self).__init__()
        self.conv_1 = nn.Conv2d(hidden_channels, hidden_channels, (3, 5), padding=(1, 2), stride=1, bias=True)
        self.conv_2 = nn.Conv2d(hidden_channels, hidden_channels, (3, 5), padding=(1, 2), stride=1, bias=True)

    def forward(self, x_in):
        x = F.relu(self.conv_1(x_in))
        x = F.relu(self.conv_2(x) + x_in)
        return x

class Debug(torch.nn.Module):
    def __init__(self):
        super(Debug, self).__init__()
        self.g, _, self.f, self.o = get_network_params()
        self.conv_h = 64
        self.num_blocks = 5

        blocks = [ResBlock1(self.conv_h) for _ in range(self.num_blocks)]
        self.res_stack = nn.Sequential(*blocks)

        self.conv_in = nn.Conv2d(self.g + self.f, self.conv_h, (3, 5), padding=(1, 2), stride=1, bias=True)
        self.avg_pool = nn.AvgPool2d((11, 21), stride=1, padding=0)
        self.max_pool = nn.MaxPool2d((11, 21), stride=1, padding=0)
        self.conv_1d = nn.Conv2d(2 * self.conv_h, self.o, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def print_stats(self, t, name):
        print(f"{name}", t.size())
        print(f"{name} max:", torch.max(t))
        print(f"{name} min:", torch.min(t))
        print(f"{name} mean:", torch.mean(t))

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

class Conv2dWS(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dWS, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class ResBlock2(torch.nn.Module):
    def __init__(self, hidden_channels, num_groups):
        super(ResBlock2, self).__init__()
        self.conv_1 = Conv2dWS(hidden_channels, hidden_channels, (3, 5), padding=(1, 2), stride=1, bias=False)
        self.conv_2 = Conv2dWS(hidden_channels, hidden_channels, (3, 5), padding=(1, 2), stride=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups, hidden_channels)
        self.gn2 = nn.GroupNorm(num_groups, hidden_channels)

    def forward(self, x_in):
        x = F.relu(self.gn1(self.conv_1(x_in)))
        x = F.relu(self.gn2(self.conv_2(x)) + x_in)
        return x

class Debug2(torch.nn.Module):
    def __init__(self):
        super(Debug2, self).__init__()
        self.g, _, self.f, self.o = get_network_params()
        self.hidden_channels = 64
        self.num_blocks = 5
        self.num_groups = 16

        self.conv_in = nn.Sequential(
            Conv2dWS(self.g + self.f, self.hidden_channels, (3, 5), padding=(1, 2), stride=1, bias=False),
            nn.GroupNorm(self.num_groups, self.hidden_channels)
        )

        blocks = [ResBlock2(self.hidden_channels, self.num_groups) for _ in range(self.num_blocks)]
        self.res_stack = nn.Sequential(*blocks)

        self.avg_pool = nn.AvgPool2d((11, 21), stride=1, padding=0)
        self.max_pool = nn.MaxPool2d((11, 21), stride=1, padding=0)
        self.conv_1d = nn.Conv2d(2 * self.hidden_channels, self.o, 1, bias=True)
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

class ResBlock3(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(ResBlock3, self).__init__()
        self.conv_1 = nn.Conv2d(hidden_channels, hidden_channels, (3, 5), padding=(1, 2), stride=1, bias=False)
        self.conv_2 = nn.Conv2d(hidden_channels, hidden_channels, (3, 5), padding=(1, 2), stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.bn2 = nn.BatchNorm2d(hidden_channels)

    def forward(self, x_in):
        x = F.relu(self.bn1(self.conv_1(x_in)))
        x = F.relu(self.bn2(self.conv_2(x)) + x_in)
        return x

class Debug3(torch.nn.Module):
    def __init__(self):
        super(Debug3, self).__init__()
        self.g, _, self.f, self.o = get_network_params()
        self.hidden_channels = 64
        self.num_blocks = 5

        self.convbn_in = nn.Sequential(
            nn.Conv2d(self.g + self.f, self.hidden_channels, (3, 5), padding=(1, 2), stride=1, bias=False),
            nn.BatchNorm2d(self.hidden_channels)
        )

        blocks = [ResBlock3(self.hidden_channels) for _ in range(self.num_blocks)]
        self.res_stack = nn.Sequential(*blocks)

        self.avg_pool = nn.AvgPool2d((11, 21), stride=1, padding=0)
        self.max_pool = nn.MaxPool2d((11, 21), stride=1, padding=0)
        self.conv_1d = nn.Conv2d(2 * self.hidden_channels, self.o, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def combine_inputs(self, x_grid, x_vector):
        b = x_grid.size()[0]
        x_vector = x_vector.view(b, self.f, 1, 1).repeat(1, 1, 11, 21)
        return torch.cat((x_grid, x_vector), dim=1)

    def forward(self, x_grid, x_vector):
        x_in = self.combine_inputs(x_grid, x_vector)
        x = F.relu(self.convbn_in(x_in))
        x = self.res_stack(x)

        x_avg = self.avg_pool(x)
        x_max = self.max_pool(x)
        x = torch.cat((x_avg, x_max), dim=1)

        x = self.conv_1d(x).squeeze()
        x = self.sigmoid(x)
        return x

class Smaller(torch.nn.Module):
    def __init__(self):
        super(Smaller, self).__init__()
        self.conv_1 = nn.Conv2d(37, 32, (3, 5))
        self.conv_2 = nn.Conv2d(32, 36, (3, 5))
        self.conv_3 = nn.Conv2d(36, 40, (3, 5))
        self.conv_4 = nn.Conv2d(40, 44, (3, 5))
        self.conv_5 = nn.Conv2d(44, 48, (3, 5))
        self.conv_6 = nn.Conv2d(48, 1,  (1, 1))
        self.sigmoid = nn.Sigmoid()

    def combine_inputs(self, x_grid, x_vector):
        x_vector = x_vector.view(x_grid.size()[0], 29, 1, 1).repeat(1, 1, 11, 21)
        return torch.cat((x_grid, x_vector), dim=1)

    def forward(self, x_grid, x_vector):
        x = self.combine_inputs(x_grid, x_vector)
        x = F.relu(self.conv_1(x))
        x = F.relu(self.conv_2(x))
        x = F.relu(self.conv_3(x))
        x = F.relu(self.conv_4(x))
        x = F.relu(self.conv_5(x))
        x = self.sigmoid(self.conv_6(x).squeeze())
        return x
