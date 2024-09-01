import math
import torch.nn as nn


class alexnet(nn.Module):
    def __init__(self, num_class=8):
        super().__init__()
        self.alexnet = nn.Sequential(

            nn.Conv2d(3, 96, (5, 5), stride=2),
            nn.ReLU(),
            nn.LocalResponseNorm(96),

            nn.Conv2d(96, 256, (3, 3), padding=1),
            nn.ReLU(),
            nn.LocalResponseNorm(256),
            nn.MaxPool2d((3, 3), stride=2),

            nn.Conv2d(256, 384, (3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d((3, 3), stride=1),

            nn.Conv2d(384, 384, (3, 3), padding=1),
            nn.ReLU(),

            nn.Conv2d(384, 384, (3, 3), padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d(3),
            nn.Flatten(),

            nn.Dropout(0.5),

            nn.Linear(384 * 3 * 3, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_class),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        output = self.alexnet(x)
        return output

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
