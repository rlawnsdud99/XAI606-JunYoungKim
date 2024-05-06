import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import QuantileTransformer
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from model_definition import CustomNet, CNN1DNet

from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter

from datetime import datetime
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import numpy as np

## 파일 읽고 데이터 라벨 쌍으로 전처리
current_dir = os.path.dirname(os.path.abspath(__file__))

train_csv_path = os.path.join(current_dir, "data", "train.csv")
val_csv_path = os.path.join(current_dir, "data", "val.csv")

train_df = pd.read_csv(train_csv_path)
val_df = pd.read_csv(val_csv_path)

# class가 0 인 행을 제거
train_dataframe = train_df[train_df["class"] != 0].copy()
val_dataframe = val_df[val_df["class"] != 0].copy()

# 데이터 셔플링
train_dataframe = train_dataframe.sample(frac=1).reset_index(drop=True)
val_dataframe = val_dataframe.sample(frac=1).reset_index(drop=True)

# class를 0부터 시작하도록 변경
train_dataframe["class"] -= 1
val_dataframe["class"] -= 1

# train_dataframe = train_df.copy()
# val_dataframe = val_df.copy()

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

mean = train_data.mean(axis=0)
std = train_data.std(axis=0)

train_data -= mean
train_data /= std

val_data -= mean
val_data /= std

# Hyperparameters
input_size = train_data.shape[1]
num_classes = len(set(train_label))
hidden_size = 32
batch_size = 2048
learning_rate = 0.001
num_epochs = 50

# 추가: Early stopping 관련 변수 설정
best_val_loss = float("inf")  # 초기 최고 validation loss 설정
patience = 50  # 몇 epoch 동안 개선이 없을지
counter = 0  # 개선이 없는 epoch 카운트

# Convert numpy arrays to PyTorch tensors
train_data_tensor = torch.FloatTensor(train_data)
train_label_tensor = torch.LongTensor(train_label)
val_data_tensor = torch.FloatTensor(val_data)
val_label_tensor = torch.LongTensor(val_label)

# Create DataLoader for training and validation datasets
train_dataset = TensorDataset(train_data_tensor, train_label_tensor)

# # Calculate class weights
# class_counts = np.bincount(train_label)
# total_samples = len(train_label)
# weights = 1.0 / class_counts
# adjusted_weights = np.power(
#     weights, 1
# )  # Increase the power to enhance minority class sampling

# sample_weights = adjusted_weights[train_label]
# # Create a WeightedRandomSampler
# sampler = WeightedRandomSampler(
#     weights=sample_weights, num_samples=len(sample_weights), replacement=True
# )

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
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)


# Model Summarys
print("Model Summary:")
summary(model, input_size=(8, 1), batch_size=batch_size)


# Function to log class distribution in a batch
def log_class_distribution(labels, batch_idx):
    class_distribution = np.bincount(labels.numpy(), minlength=len(class_counts))
    print(f"Batch {batch_idx} class distribution: {class_distribution}")


# ... (이전 코드는 생략)
for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    num_batches = 0
    correct_train = 0  # Training Accuracy Calculation
    total_train = 0  # Training Accuracy Calculation
    train_preds = []
    train_true = []
    for i, (data, labels) in enumerate(train_loader):
        # log_class_distribution(labels, i)
        data = data.to(device)
        labels = labels.to(device)
        outputs = model(data)

        loss = criterion(outputs, labels)
        # Training Accuracy Calculation
        _, predicted_train = torch.max(outputs.data, 1)
        train_preds.extend(predicted_train.cpu().numpy())
        train_true.extend(labels.cpu().numpy())
        total_train += labels.size(0)
        correct_train += (predicted_train == labels).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        num_batches += 1

    # scheduler.step()
    avg_train_loss = total_train_loss / num_batches
    train_accuracy = 100 * correct_train / total_train  # Training Accuracy Calculation

    # Log to TensorBoard
    writer.add_scalar("Training Loss", avg_train_loss, epoch)
    writer.add_scalar(
        "Training Accuracy", train_accuracy, epoch
    )  # Training Accuracy Logging

    print(
        f"Epoch [{epoch+1}/{num_epochs}], Average Training Loss: {avg_train_loss:.4f}, Training Accuracy: {train_accuracy}%"
    )
    # Compute F1 Score for training
    train_f1_scores = f1_score(train_true, train_preds, average=None)
    train_f1_macro = np.mean(train_f1_scores)
    print(f"Train F1 Scores by Class: {train_f1_scores}")
    print(f"Train Macro F1 Score: {train_f1_macro:.3f}")

    # Validation after each epoch
    model.eval()
    val_preds = []
    val_true = []
    with torch.no_grad():
        correct = 0
        total = 0
        val_loss = 0
        for data, labels in val_loader:
            data = data.to(device)
            labels = labels.to(device)  # Move data to CUDA

            outputs = model(data)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_preds.extend(predicted.cpu().numpy())
            val_true.extend(labels.cpu().numpy())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader)
        accuracy = 100 * correct / total

        # Log to TensorBoard
        writer.add_scalar("Validation Loss", val_loss, epoch)
        writer.add_scalar("Validation Accuracy", accuracy, epoch)

        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {accuracy}%")
        torch.save(model.state_dict(), "trained_model.pth")

        # Compute F1 Score for validation
        val_f1_scores = f1_score(val_true, val_preds, average=None)
        val_f1_macro = np.mean(val_f1_scores)
        print(f"Validation F1 Scores by Class: {val_f1_scores}")
        print(f"Validation Macro F1 Score: {val_f1_macro:.3f}")

        # Early stopping 체크
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0  # 개선이 있으면 counter 초기화
            torch.save(model.state_dict(), "best_trained_model.pth")  # 최고 모델 저장
        else:
            counter += 1  # 개선이 없으면 counter 증가
            print(f"EarlyStopping counter: {counter} out of {patience}")

            if counter >= patience:
                print("EarlyStopping")
                model.load_state_dict(
                    torch.load("best_trained_model.pth")
                )  # 최고 모델 불러오기
                break  # 학습 종료

    scheduler.step()
    # Log current learning rate
    current_lr = optimizer.param_groups[0]["lr"]
    writer.add_scalar("Learning Rate", current_lr, epoch)

writer.close()
