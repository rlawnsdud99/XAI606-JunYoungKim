import mne
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model_definition import EEGNet
import torch.nn as nn

raw = mne.io.read_raw_eeglab(f"subject_1.set", preload=True)

raw.filter(l_freq=1, h_freq=45)

ica = mne.preprocessing.ICA(n_components=10, random_state=97, max_iter=800)
ica.fit(raw)
raw.load_data()
ica.apply(raw)

raw.filter(l_freq=1, h_freq=8)

# Annotation에서 이벤트 생성
events, event_id_from_anno = mne.events_from_annotations(raw)

# 원하는 이벤트 ID만 선택
selected_event_ids = {
    key: value
    for key, value in event_id_from_anno.items()
    if key
    in [
        "S  1",
        "S  2",
        "S  3",
        "S  4",
        "S  5",
        "S  6",
        "S  7",
        "S  8",
        "S  9",
        "S 10",
        "S 11",
        "S 12",
    ]
}

# Epoch 생성
epoch_tmin, epoch_tmax = -0.2, 0.5  # 시작과 끝을 -1s ~ 1s로 지정
baseline = (None, 0)  # baseline correction을 위한 시간 범위
epochs = mne.Epochs(
    raw,
    events,
    selected_event_ids,
    tmin=epoch_tmin,
    tmax=epoch_tmax,
    picks="eeg",
    baseline=baseline,
    detrend=1,
    preload=True,
)

# 선택적으로 Epoch 데이터를 별도의 변수에 저장
data = epochs.get_data()


labels = epochs.events[:, -1]

# Normalization
data -= np.mean(data, axis=0)
data /= np.std(data, axis=0)


# 셔플하기
data, labels = shuffle(data, labels, random_state=42)

# 데이터 분할
train_size = 0.7
val_size = 0.15
test_size = 0.15

x_temp, x_test, y_temp, y_test = train_test_split(
    data, labels, test_size=test_size, random_state=42
)
x_train, x_val, y_train, y_val = train_test_split(
    x_temp, y_temp, test_size=val_size / (train_size + val_size), random_state=42
)

# 레이블을 0부터 시작하게 조정
y_train -= 1
y_val -= 1
y_test -= 1

# 각 세트의 크기 출력
print(f"Training set size: {x_train.shape[0]}")
print(f"Validation set size: {x_val.shape[0]}")
print(f"Test set size: {x_test.shape[0]}")

# 레이블 분포 확인
print("\nLabel distribution in training set:")
print(np.bincount(y_train))
print("\nLabel distribution in validation set:")
print(np.bincount(y_val))
print("\nLabel distribution in test set:")
print(np.bincount(y_test))

# 데이터 차원 확인
print("\nData dimensions:")
print(f"Training data shape: {x_train.shape}")
print(f"Validation data shape: {x_val.shape}")
print(f"Test data shape: {x_test.shape}")

# 레이블 차원 확인
print("\nLabel dimensions:")
print(f"Training label shape: {y_train.shape}")
print(f"Validation label shape: {y_val.shape}")
print(f"Test label shape: {y_test.shape}")

print(y_train)

# 데이터를 PyTorch Tensor로 변환
x_train_tensor = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
x_val_tensor = torch.tensor(x_val, dtype=torch.float32).unsqueeze(1)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)
print(f"x_train_tensor: {x_train_tensor.shape}")
# DataLoader 설정
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 모델 생성 및 컴파일
num_channels = x_train.shape[1]  # 데이터에 따라 변경
model = EEGNet(num_classes=len(np.unique(y_train)), num_channels=num_channels)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# 학습
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    for i, (x_batch, y_batch) in enumerate(train_loader):  # 새로운 차원 추가
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
    # Validation
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            outputs = model(x_batch)
            val_loss += criterion(outputs, y_batch).item()
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(y_batch.view_as(pred)).sum().item()

    val_loss /= len(val_loader.dataset)
    val_accuracy = 100.0 * correct / len(val_loader.dataset)

    print(
        f"Epoch: {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%"
    )

# Test
model.eval()
test_loss = 0
correct = 0
with torch.no_grad():
    for x_batch, y_batch in test_loader:
        outputs = model(x_batch)
        test_loss += criterion(outputs, y_batch).item()
        pred = outputs.argmax(dim=1, keepdim=True)
        correct += pred.eq(y_batch.view_as(pred)).sum().item()

test_loss /= len(test_loader.dataset)
test_accuracy = 100.0 * correct / len(test_loader.dataset)

print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
