from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import numpy as np
import scipy.io as scio
import random

import torch
# 模型构建
import torch.nn as nn
# 模型训练
import torch.optim as optim
from torch.autograd import Variable
# 数据准备
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from dataloader import CharacterDataset
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib
import os

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # 如果使用的是 CUDA
    torch.backends.cudnn.benchmark = False

class CNN3D(nn.Module):
    def __init__(self, in_channels=30, num_classes=16):
        super(CNN3D, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(7, 3, 3), padding=(0, 1, 1)),  # (1, C, H, W) → (8, C-6, H, W)
            nn.ReLU(),
            # nn.BatchNorm3d(8),  ← 已去除
            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            # nn.BatchNorm3d(16),  ← 已去除
            nn.AdaptiveAvgPool3d((1, 1, 1))  # 输出 (16, 1, 1, 1)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),        # 展平成 (batch, 16)
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

def extract_3d_patches(img_cube, label_map, patch_size=7, spectral_channels=30, ignored_label=0):
    """
    提取 3D Patch 数据（空间大小 patch_size × patch_size，光谱通道数为 spectral_channels），
    用于 3D-CNN 训练。输出格式 (N, 1, C, H, W)。

    参数:
        img_cube: ndarray, (H, W, C) → 原始高光谱图像数据
        label_map: ndarray, (H, W) → 标签图（0 为背景/无效类）
        patch_size: int, 空间尺寸（建议奇数，默认 7）
        spectral_channels: int, 使用的光谱通道数（默认 30）
        ignored_label: int, 默认为 0，表示忽略背景像素

    返回:
        patches: ndarray, shape = (N, 1, spectral_channels, patch_size, patch_size)
        labels:  ndarray, shape = (N,)
    """
    assert patch_size % 2 == 1, "Patch size must be odd."
    H, W, C = img_cube.shape
    pad = patch_size // 2

    # 截取光谱前 spectral_channels 个波段
    img_cube = img_cube[:, :, :spectral_channels]

    # pad spatial 维度
    padded_img = np.pad(img_cube, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    padded_label = np.pad(label_map, ((pad, pad), (pad, pad)), mode='constant')

    patches = []
    labels = []

    for i in range(pad, H + pad):
        for j in range(pad, W + pad):
            label = padded_label[i, j]
            if label == ignored_label:
                continue
            # 提取 patch，shape: (patch_size, patch_size, C)
            spatial_patch = padded_img[i - pad:i + pad + 1, j - pad:j + pad + 1, :]
            # 转换为 (C, H, W)
            patch = np.transpose(spatial_patch, (2, 0, 1))
            # 增加 batch 维度 → (1, C, H, W)
            patch = patch[np.newaxis, ...]
            patches.append(patch)
            labels.append(label - 1)  # 从 0 开始编号

    return np.array(patches, dtype='float32'), np.array(labels, dtype='int64')

data_path = '../data/Indian_pines/Indian_pines_corrected.mat'
label_path = '../data/Indian_pines/Indian_pines_gt.mat'
data = scio.loadmat(data_path)['indian_pines_corrected']
h, w, bands = data.shape
label = scio.loadmat(label_path)['indian_pines_gt'].flatten()


def train_and_evaluate(run_seed=42):
    set_seed(run_seed)

    # -------- 数据准备 --------
    data_reshaped = data.reshape(h * w, bands)

    # --- PCA: 从200通道降到30通道 ---
    pca = PCA(n_components=30)
    data_pca = pca.fit_transform(data_reshaped)  # shape: (h*w, 30)
    data_cube = data_pca.reshape(h, w, 30)  # shape: (145, 145, 30)

    label_map = label.reshape(h, w)

    # --- 提取 3D patch 数据 ---
    patches, patch_labels = extract_3d_patches(
        data_cube, label_map,
        patch_size=7, spectral_channels=30  # 注意要和前面一致
    )

    # 划分训练、验证、测试集
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        patches, patch_labels, test_size=0.15, random_state=42, stratify=patch_labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.176, random_state=42, stratify=y_train_full
    )

    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)), batch_size=128,
                              shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)), batch_size=128,
                            shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)), batch_size=128,
                             shuffle=False)

    # -------- 模型定义 --------
    # device = torch.device("cuda" if torch.backends.mps.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = CNN3D(in_channels=30, num_classes=16).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # -------- 训练阶段 --------
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    train_losses, val_losses = [], []

    for epoch in range(1, 101):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                val_loss += criterion(model(X_batch), y_batch).item()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"⏹️ Early stopping at epoch {epoch}")
                break

    # -------- 测试阶段 --------
    model.load_state_dict(best_model_weights)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            pred = model(X_batch).argmax(dim=1)
            correct += (pred == y_batch).sum().item()
            total += y_batch.size(0)

    acc = correct / total * 100
    print(f"✅ Test Accuracy: {acc:.2f}%")

    # -------- 分类图像可视化 --------
    pred_map = np.zeros((h, w), dtype=int)
    mask = (label_map != 0)

    for i in range(h):
        for j in range(w):
            if mask[i, j]:
                patch = data_cube[i - 3:i + 4, j - 3:j + 4, :30]  # 7×7×30 patch
                if patch.shape != (7, 7, 30):
                    continue
                patch = np.transpose(patch, (2, 0, 1))[np.newaxis, np.newaxis, ...]
                patch_tensor = torch.tensor(patch, dtype=torch.float32).to(device)
                with torch.no_grad():
                    pred_label = model(patch_tensor).argmax(dim=1).item()
                pred_map[i, j] = pred_label + 1  # 从1开始

    # 颜色映射
    cmap = mcolors.ListedColormap(plt.colormaps['tab20'].colors)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].imshow(label_map, cmap=cmap, vmin=1, vmax=16)
    axs[0].set_title("Ground Truth")
    axs[0].axis('off')

    axs[1].imshow(pred_map, cmap=cmap, vmin=1, vmax=16)
    axs[1].set_title(f"3D-CNN Prediction (Acc: {acc:.2f}%)")
    axs[1].axis('off')

    os.makedirs("figures", exist_ok=True)
    plt.savefig(f"figures/3DCNN_run{run_seed}_without_batchnorm.pdf", bbox_inches='tight')
    plt.close()

    # -------- Loss 曲线 --------
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"3D-CNN Loss Curve (Seed={run_seed})")
    plt.legend()
    os.makedirs("figures", exist_ok=True)
    plt.savefig(f"figures/3DCNN_loss_seed{run_seed}_without_batchnorm.pdf", bbox_inches='tight')
    plt.close()

    return acc

# 多次重复训练
repeats = 10
accuracies = []

for i in range(repeats):
    print(f"\n🔁 Running trial {i+1} with seed {i*10+42}")
    acc = train_and_evaluate(run_seed=i*10+42)
    print(f"✅ Accuracy for run {i+1}: {acc:.2f}%")
    accuracies.append(acc)

# 计算统计指标
mean_acc = np.mean(accuracies)
std_acc = np.std(accuracies)

# 输出与保存
print("\n📊 Summary of Repeated Training:")
for i, acc in enumerate(accuracies):
    print(f"Run {i+1}: {acc:.2f}%")
print(f"\nAverage Accuracy: {mean_acc:.2f}%, Std Dev: {std_acc:.2f}%")

os.makedirs("results", exist_ok=True)
with open("results/3dcnn_repeat_results_without_batchnorm.txt", "w") as f:
    for i, acc in enumerate(accuracies):
        f.write(f"Run {i+1}: {acc:.2f}%\n")
    f.write(f"\nAverage Accuracy: {mean_acc:.2f}%\n")
    f.write(f"Standard Deviation: {std_acc:.2f}%\n")