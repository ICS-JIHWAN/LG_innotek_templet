import math
import torch.nn as nn


class convBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)


class residual_block(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            convBlock(in_channel, out_channel, kernel_size=3, stride=stride, padding=1),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel)
        )

        self.skip_connection = nn.Sequential()
        self.relu = nn.ReLU()

        if stride != 1 or in_channel != out_channel:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self, x):
        x = self.residual_function(x) + self.skip_connection(x)
        x = self.relu(x)
        return x


class resnet(nn.Module):
    def __init__(self, num_class=8):
        super().__init__()

        block = residual_block
        num_blocks = [2, 2, 2, 2]

        self.in_channels = 64
        self.conv1 = convBlock(3, self.in_channels, kernel_size=3, stride=1, padding=0)

        self.conv2_x = self._make_layer(block, 64, num_blocks[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_blocks[1], 1)
        self.conv4_x = self._make_layer(block, 256, num_blocks[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_blocks[3], 2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(512, num_class)
        )

        self.sigmoid = nn.Sigmoid()

    def _make_layer(self, block, out_channel, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)

        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channel, stride))
            self.in_channels = out_channel

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2_x(out)
        out = self.conv3_x(out)
        out = self.conv4_x(out)
        out = self.conv5_x(out)
        out = self.avg_pool(out)
        out = self.flatten(out)
        out = self.fc(out)

        return out

    def predict(self, output):
        sig_out = self.sigmoid(output)
        sig_out[sig_out > 0.5] = 1
        sig_out[sig_out <= 0.5] = 0
        return sig_out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
