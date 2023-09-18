import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from model_definition import EEGNet
from sklearn.decomposition import PCA


def apply_pca_then_tsne(
    x_data, y_labels, title="PCA + t-SNE plot", n_pca_components=50, perplexity=50
):
    # PCA 적용
    pca = PCA(n_components=n_pca_components)
    x_pca = pca.fit_transform(x_data.reshape(x_data.shape[0], -1))

    # t-SNE 적용
    tsne = TSNE(n_components=3, perplexity=perplexity, random_state=42)
    x_tsne = tsne.fit_transform(x_pca)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(
        x_tsne[:, 0], x_tsne[:, 1], x_tsne[:, 2], c=y_labels, cmap="viridis"
    )

    ax.set_title(title)
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.set_zlabel("t-SNE Dimension 3")

    legend_labels = [
        plt.Line2D(
            [0],
            [0],
            linestyle="none",
            marker="o",
            alpha=1,
            markersize=10,
            markerfacecolor=plt.cm.viridis(i / np.max(y_labels)),
        )
        for i in np.unique(y_labels)
    ]
    ax.legend(legend_labels, np.unique(y_labels).astype(str), title="Labels")

    plt.show()


# Load preprocessed data from .npy files
X_train = np.load("x_train.npy")
y_train = np.load("y_train.npy")
X_val = np.load("x_val.npy")
y_val = np.load("y_val.npy")
X_test = np.load("x_test.npy")
y_test = np.load("y_test.npy")

# 모델 로드
model = EEGNet(num_classes=12, num_channels=32)  # 클래스 수와 채널 수는 예시
model.load_state_dict(torch.load("trained_model.pth"))
model.eval()

# 데이터 로드 (x_train, y_train은 이미 로드되어 있다고 가정)
x_sample = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)

# 마지막 레이어 전까지의 출력 얻기
with torch.no_grad():
    layer_output = model(x_sample, return_last_layer=True).numpy()

# t-SNE 적용
apply_pca_then_tsne(layer_output, y_train, title="PCA + t-SNE on Last Layer Output")
