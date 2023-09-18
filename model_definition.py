import torch.nn as nn


class EEGNet(nn.Module):
    def __init__(self, num_classes, num_channels):
        super(EEGNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, (1, 64), padding=(0, 32))
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.depthwise = nn.Conv2d(16, 16, (1, 1), groups=16)
        self.batchnorm2 = nn.BatchNorm2d(16)
        self.activation = nn.ELU()
        self.avg_pool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(0.25)

        self.separable = nn.Conv2d(16, 16, (1, 16))
        self.batchnorm3 = nn.BatchNorm2d(16)
        self.avg_pool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(0.25)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(1536, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.depthwise(x)
        x = self.batchnorm2(x)
        x = self.activation(x)
        x = self.avg_pool1(x)
        x = self.dropout1(x)

        x = self.separable(x)
        x = self.batchnorm3(x)
        x = self.activation(x)
        x = self.avg_pool2(x)
        x = self.dropout2(x)

        x = self.flatten(x)
        x = self.fc(x)
        x = self.softmax(x)

        return x
