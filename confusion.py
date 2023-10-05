from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
from model_definition import CustomNet  # 모델 정의가 이 파일에 있다고 가정
import os
from joblib import load
from sklearn.metrics import accuracy_score

# 파일 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
train_csv_path = os.path.join(current_dir, "data", "train.csv")
test_csv_path = os.path.join(current_dir, "data", "test_labeled.csv")
train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)

# 특정 클래스 제거 및 라벨 수정
train_df = train_df[(train_df["class"] != 0) & (train_df["class"] != 7)].copy()
test_df = test_df[(test_df["class"] != 0) & (test_df["class"] != 7)].copy()
train_df["class"] -= 1
test_df["class"] -= 1
# 데이터와 라벨 분리
train_data = train_df.drop(columns=["label", "class", "time"]).values
train_label = train_df["class"].values
test_data = test_df.drop(columns=["label", "class", "time"]).values
test_label = test_df["class"].values

# 정규화
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data = (train_data - mean) / std
test_data = (test_data - mean) / std  # 테스트 데이터도 학습 데이터의 평균과 표준편차로 정규화

# DataLoader 설정
batch_size = 512
train_tensor = TensorDataset(
    torch.FloatTensor(train_data), torch.LongTensor(train_label)
)
train_loader = DataLoader(train_tensor, batch_size=batch_size, shuffle=True)
test_tensor = TensorDataset(torch.FloatTensor(test_data), torch.LongTensor(test_label))
test_loader = DataLoader(test_tensor, batch_size=batch_size, shuffle=False)

# 모델 불러오기
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomNet(len(set(train_label))).to(device)
model.load_state_dict(torch.load("best_trained_model.pth"))
model.eval()

# CustomNet 예측
predictions = []
with torch.no_grad():
    for data, _ in test_loader:
        data = data.to(device)
        outputs = model(data)
        _, predicted = torch.max(outputs, 1)
        predictions.extend(predicted.cpu().numpy())
cm = confusion_matrix(test_label, predictions)

# KNN 모델로 예측
knn_model = load("knn_model.joblib")
knn_predictions = knn_model.predict(test_data)
cm_knn = confusion_matrix(test_label, knn_predictions)

# Neural Network 모델의 정확도
nn_accuracy = accuracy_score(test_label, predictions)

# KNN 모델의 정확도
knn_accuracy = accuracy_score(test_label, knn_predictions)

# Subplot 설정
fig, axes = plt.subplots(1, 2, figsize=(20, 7))

# 첫 번째 subplot: Neural Network Confusion Matrix
sns.heatmap(cm, annot=True, fmt="d", ax=axes[0])
axes[0].set_title(f"Custom Neural Network (Accuracy: {nn_accuracy:.2f})")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("True")

# 두 번째 subplot: KNN Confusion Matrix
sns.heatmap(cm_knn, annot=True, fmt="d", ax=axes[1])
axes[1].set_title(f"KNN (Accuracy: {knn_accuracy:.2f})")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("True")

plt.tight_layout()
plt.show()
