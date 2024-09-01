import math
import torch.nn as nn


class lenet(nn.Module):
    def __init__(self, num_class=8):
        super().__init__()
        self.lenet = nn.Sequential(
            #
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2),
            nn.Tanh(),
            #
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2),
            nn.Tanh(),
            #
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1, padding=0),
            nn.Tanh(),
            #
            nn.Flatten(),
            nn.Linear(in_features=4320, out_features=2160),
            nn.Tanh(),
            nn.Linear(in_features=2160, out_features=num_class)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.lenet(x)

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
