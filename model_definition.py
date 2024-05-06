import torch.nn as nn
import torch.nn.functional as F
import torch


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class EEGNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(EEGNet, self).__init__()
        self.temporal_conv = nn.Conv2d(
            1,
            hidden_size,
            kernel_size=(8, input_size // 8),
            padding=(4, input_size // 16),
        )
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.depthwise_conv = nn.Conv1d(
            hidden_size, hidden_size * 2, kernel_size=8, groups=hidden_size
        )
        self.batch_norm2 = nn.BatchNorm1d(hidden_size * 2)
        self.activation = nn.ELU()
        self.avg_pool = nn.AvgPool1d(kernel_size=4)
        self.dropout = nn.Dropout(0.5)
        self.separable_conv = nn.Conv1d(
            hidden_size * 2, hidden_size * 2, kernel_size=16
        )
        self.batch_norm3 = nn.BatchNorm1d(hidden_size * 2)
        self.avg_pool2 = nn.AvgPool1d(kernel_size=8)
        self.fc = nn.Linear(
            hidden_size * 40, num_classes
        )  # This size depends on the output size after convolutions.

    def forward(self, x):
        x = self.temporal_conv(x)
        x = self.batch_norm1(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm2(x)
        x = self.activation(x)
        x = self.avg_pool(x)
        x = self.dropout(x)
        x = self.separable_conv(x)
        x = self.batch_norm3(x)
        x = self.activation(x)
        x = self.avg_pool2(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class FCNet(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(FCNet, self).__init__()

        self.fc1 = nn.Linear(8, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(hidden_size, hidden_size * 2)
        self.bn2 = nn.BatchNorm1d(hidden_size * 2)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.fc3(out)
        return out


class CustomNet(nn.Module):
    def __init__(self, num_classes):
        super(CustomNet, self).__init__()

        # Drop out
        self.dropout = nn.Dropout(0.1)
        # First Path
        self.fc1 = nn.Linear(8, 1024)
        self.fc2 = nn.Linear(1024, 512)

        # Second Path from y
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 32)

        # Third Path from y
        self.fc7 = nn.Linear(512, 128)
        self.fc8 = nn.Linear(128, 64)
        self.fc9 = nn.Linear(64, 32)

        # Final Layer
        self.fc10 = nn.Linear(64, num_classes)

    def forward(self, x):

        x = x.view(x.size(0), -1)
        x1 = F.relu(self.fc1(x))
        x1 = self.dropout(x1)  # Dropout after first layer
        y = F.relu(self.fc2(x1))

        # Second Path
        z1 = F.relu(self.fc3(y))
        z1 = self.dropout(z1)  # Dropout
        z1 = F.relu(self.fc4(z1))
        z1 = self.dropout(z1)  # Dropout
        z1 = F.relu(self.fc5(z1))
        z1 = self.dropout(z1)  # Dropout
        z1 = F.relu(self.fc6(z1))

        # Third Path
        z2 = F.relu(self.fc7(y))
        z2 = self.dropout(z2)  # Dropout
        z2 = F.relu(self.fc8(z2))
        z2 = self.dropout(z2)  # Dropout
        z2 = F.relu(self.fc9(z2))

        # Combining the two paths using Concatenation
        combined = torch.cat((z1, z2), dim=1)

        out = self.fc10(combined)
        return out


class CNN1DNet(nn.Module):
    def __init__(self, num_classes):
        super(CNN1DNet, self).__init__()
        # Assuming input shape [batch, 8, 1] for 8 channels of EMG data
        self.conv1 = nn.Conv1d(8, 16, kernel_size=1, stride=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(
            kernel_size=1
        )  # Might adjust depending on further model design considerations
        self.conv2 = nn.Conv1d(16, 32, kernel_size=1, stride=1)
        self.fc1 = nn.Linear(32, num_classes)

    def forward(self, x):

        x = x.view(x.size(1), -1)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.relu(self.conv2(x))
        # Flatten the output from the convolutional layers to feed into the fully connected layer
        x = torch.flatten(x, 1)
        x = x.view(x.size(1), -1)
        x = self.fc1(x)
        return x
