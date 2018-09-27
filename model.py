import torch
import torch.nn as nn

# Number of bottlenecks
num_bn = 3
# The depth is half of the actual values in the paper because bottleneck blocks
# are used which contain two convlutional layers
depth = 16
multi_block_depth = depth // 2
growth_rate = 24


class BottleneckBlock(nn.Module):
    def __init__(self, input_size, growth_rate):
        super(BottleneckBlock, self).__init__()
        inter_size = num_bn * growth_rate
        self.norm1 = nn.BatchNorm2d(input_size)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(input_size, inter_size,
                               kernel_size=1, stride=1, bias=False)
        self.norm2 = nn.BatchNorm2d(inter_size)
        self.conv2 = nn.Conv2d(inter_size, growth_rate,
                               kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(self.relu(self.norm1(x)))
        out = self.conv2(self.relu(self.norm2(out)))
        return torch.cat([x, out], 1)


class TransitionBlock(nn.Module):
    def __init__(self, input_size, output_size):
        super(TransitionBlock, self).__init__()
        self.norm = nn.BatchNorm2d(input_size)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(input_size, output_size,
                              kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.conv(self.relu(self.norm(x)))
        return self.pool(out)


class DenseBlock(nn.Module):
    def __init__(self, input_size, growth_rate, depth):
        super(DenseBlock, self).__init__()
        layers = [BottleneckBlock(input_size + i * growth_rate, growth_rate)
                  for i in range(depth)]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(self, num_in_features=48):
        super(Encoder, self).__init__()
        self.conv0 = nn.Conv2d(
            3, num_in_features, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm0 = nn.BatchNorm2d(num_in_features)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        num_features = num_in_features
        self.block1 = DenseBlock(
            num_features, growth_rate=growth_rate, depth=depth)
        num_features = num_features + depth * growth_rate
        self.trans1 = TransitionBlock(num_features, num_features // 2)
        num_features = num_features // 2
        self.block2 = DenseBlock(
            num_features, growth_rate=growth_rate, depth=depth)

        num_features = num_features + depth * growth_rate
        self.trans2_norm = nn.BatchNorm2d(num_features)
        self.trans2_relu = nn.ReLU(inplace=True)
        self.trans2_conv = nn.Conv2d(num_features, num_features // 2,
                                     kernel_size=1, stride=1, bias=False)
        self.trans2_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.multi_block = DenseBlock(
            num_features, growth_rate=growth_rate, depth=multi_block_depth)
        num_features = num_features // 2
        self.block3 = DenseBlock(
            num_features, growth_rate=growth_rate, depth=depth)
        num_features = num_features + depth * growth_rate // 2

    def forward(self, x):
        out = self.relu(self.norm0(self.conv0(x)))
        out = self.max_pool(out)
        out = self.block1(out)
        out = self.trans1(out)
        out = self.block2(out)
        out_before_trans2 = self.trans2_relu(self.trans2_norm(out))
        out_A = self.trans2_conv(out_before_trans2)
        out_A = self.trans2_pool(out_A)
        out_A = self.block3(out_A)
        out_B = self.multi_block(out_before_trans2)

        return out_A, out_B
