import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import QuantileTransformer
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from model_definition import CustomNet

from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter

from datetime import datetime
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

## 파일 읽고 데이터 라벨 쌍으로 전처리
current_dir = os.path.dirname(os.path.abspath(__file__))

train_csv_path = os.path.join(current_dir, "data", "train.csv")
val_csv_path = os.path.join(current_dir, "data", "val.csv")

train_df = pd.read_csv(train_csv_path)
val_df = pd.read_csv(val_csv_path)

# class가 0,7 인 행을 제거
train_dataframe = train_df[(train_df["class"] != 0) & (train_df["class"] != 7)].copy()
val_dataframe = val_df[(val_df["class"] != 0) & (val_df["class"] != 7)].copy()

# 데이터 셔플링
train_dataframe = train_dataframe.sample(frac=1).reset_index(drop=True)
val_dataframe = val_dataframe.sample(frac=1).reset_index(drop=True)

# 라벨을 1씩 감소(CEE 사용경우 라벨 0부터 시작해야함)
train_dataframe["class"] = train_dataframe["class"] - 1
val_dataframe["class"] = val_dataframe["class"] - 1

# Train set에서 클래스 분포 확인
train_class_distribution = train_dataframe["class"].value_counts().sort_index()
print("Train Class Distribution:")
print(train_class_distribution)

# Validation set에서 클래스 분포 확인
val_class_distribution = val_dataframe["class"].value_counts().sort_index()
print("\nValidation Class Distribution:")
print(val_class_distribution)

train_data = (train_dataframe.drop(columns=["label", "class", "time"])).values
train_label = (train_dataframe["class"]).values

val_data = (val_dataframe.drop(columns=["label", "class", "time"])).values
val_label = (val_dataframe["class"]).values

print(f"type: {type(train_data)}, shape: {train_data.shape}")
print(f"type: {type(train_label)}, shape: {train_label.shape}")
print(f"type: {type(val_data)}, shape: {val_data.shape}")
print(f"type: {type(val_label)}, shape: {val_label.shape}")

# 정규화
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)

train_data -= mean
train_data /= std

val_data -= mean
val_data /= std

# 정규화된 데이터셋 plot
plt.figure(figsize=(16, 6))

# Plot the distribution of features for training data
plt.subplot(1, 2, 1)
sns.boxplot(data=train_data)
plt.title("Distribution of Normalized Training Features")

# Plot the distribution of features for validation data
plt.subplot(1, 2, 2)
sns.boxplot(data=val_data)
plt.title("Distribution of Normalized Validation Features")

plt.tight_layout()
plt.show()

# Hyperparameters
input_size = train_data.shape[1]
num_classes = len(set(train_label))
hidden_size = 32
batch_size = 512
learning_rate = 0.001
num_epochs = 50

# Convert numpy arrays to PyTorch tensors
train_data_tensor = torch.FloatTensor(train_data)
train_label_tensor = torch.LongTensor(train_label)
val_data_tensor = torch.FloatTensor(val_data)
val_label_tensor = torch.LongTensor(val_label)

# Create DataLoader for training and validation datasets
train_dataset = TensorDataset(train_data_tensor, train_label_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
for data, labels in train_loader:
    print(f"Data shape: {data.shape}, Labels shape: {labels.shape}")
    break
val_dataset = TensorDataset(val_data_tensor, val_label_tensor)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
for data, labels in val_loader:
    print(f"Data shape: {data.shape}, Labels shape: {labels.shape}")
    break
# device 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TensorBoard Summary Writer
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
experiment_name = f"Experiment_{current_time}"
writer = SummaryWriter(log_dir=f"./runs/{experiment_name}")

# Initialize neural network, loss and optimizer, scheduler
model = CustomNet(num_classes).to(device)
# model = CNN1DNet(hidden_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, "min", patience=3, factor=0.5
)

# Model Summarys
print("Model Summary for CustomNet:")
summary(model, input_size=(8, 1), batch_size=batch_size)
