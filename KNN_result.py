import os
import pandas as pd

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import numpy as np

from joblib import dump, load

## 파일 읽고 데이터 라벨 쌍으로 전처리
current_dir = os.path.dirname(os.path.abspath(__file__))

train_csv_path = os.path.join(current_dir, "data", "train.csv")
val_csv_path = os.path.join(current_dir, "data", "val.csv")

train_df = pd.read_csv(train_csv_path)
val_df = pd.read_csv(val_csv_path)

# class가 0,7 인 행을 제거
train_dataframe = train_df[train_df["class"] != 0].copy()
val_dataframe = val_df[val_df["class"] != 0].copy()

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

mean = train_data.mean(axis=0)
std = train_data.std(axis=0)

train_data -= mean
train_data /= std

val_data -= mean
val_data /= std

# k-NN 모델 생성
k = 5  # k 값 설정
knn_model = KNeighborsClassifier(n_neighbors=k)

# 학습 시작 로깅
print("k-NN training start")
knn_model.fit(train_data, train_label)
print("k-NN training end")
# 모델 저장
dump(knn_model, "knn_model.joblib")

print("k-NN prediction start")
val_predictions = knn_model.predict(val_data)
print("k-NN prediction end")

# 정확도 계산
accuracy = accuracy_score(val_label, val_predictions)
print(f"Validation accuracy: {accuracy}")

# k-NN 모델의 예측 확률 구하기
val_probabilities = knn_model.predict_proba(val_data)

# t-SNE 설정
tsne = TSNE(n_components=2, random_state=0, verbose=1)

# t-SNE 적용
tsne_results = tsne.fit_transform(val_probabilities)

# 산점도 그리기
plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    tsne_results[:, 0],
    tsne_results[:, 1],
    c=val_label,
    cmap="viridis",
    alpha=0.6,
)
plt.colorbar(scatter, ticks=np.arange(min(val_label), max(val_label) + 1))
plt.title("t-SNE of Model Output")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.show()
