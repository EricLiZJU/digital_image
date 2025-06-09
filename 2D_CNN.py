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

def extract_patches(img_cube, label_map, patch_size=7, ignored_label=0):
    """
    提取中心像素有标签的 Patch 和对应的标签
    参数：
        img_cube: numpy array, shape (H, W, C), PCA 后的高光谱图像
        label_map: numpy array, shape (H, W), 语义标签图（0 为背景或无标签）
        patch_size: int, Patch 尺寸，默认 7×7，必须为奇数
        ignored_label: int, 默认 0（跳过背景类）

    返回：
        patches: np.array, shape (N, C, patch_size, patch_size)
        labels:  np.array, shape (N,)
    """
    assert patch_size % 2 == 1, "Patch size must be odd."

    H, W, C = img_cube.shape
    pad = patch_size // 2
    padded_img = np.pad(img_cube, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    padded_label = np.pad(label_map, ((pad, pad), (pad, pad)), mode='constant')

    patches = []
    labels = []

    for i in range(pad, H + pad):
        for j in range(pad, W + pad):
            label = padded_label[i, j]
            if label == ignored_label:
                continue
            patch = padded_img[i - pad:i + pad + 1, j - pad:j + pad + 1, :]
            patch = np.transpose(patch, (2, 0, 1))  # (C, H, W)
            patches.append(patch)
            labels.append(label - 1)  # 标签从 0 开始编号

    return np.array(patches, dtype='float32'), np.array(labels, dtype='int64')

class PatchDataset(torch.utils.data.Dataset):
    def __init__(self, patches, labels, transform=None):
        self.patches = patches  # (N, C, H, W)
        self.labels = labels    # (N,)
        self.transform = transform

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        img = self.patches[idx]
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

class CNN2D(nn.Module):
    def __init__(self, in_channels=3, num_classes=16):
        super(CNN2D, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),  # (3, 7, 7) -> (32, 7, 7)
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),  # (32, 7, 7) -> (32, 3, 3)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # (64, 3, 3)
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 3 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

"""
data_path = 'data/Indian_pines/Indian_pines_corrected.mat'
label_path = 'data/Indian_pines/Indian_pines_gt.mat'
data = scio.loadmat(data_path)['indian_pines_corrected']
h, w, bands = data.shape
label = scio.loadmat(label_path)['indian_pines_gt'].flatten()
"""
data_path = 'data/Pavia_university/PaviaU.mat'
label_path = 'data/Pavia_university/PaviaU_gt.mat'
data = scio.loadmat(data_path)['paviaU']
h, w, bands = data.shape
label = scio.loadmat(label_path)['paviaU_gt'].flatten()

def train_and_evaluate(run_seed=42):

    set_seed(run_seed)
    # 数据准备
    data_reshaped = data.reshape(h * w, bands)
    pca = PCA(n_components=3)
    data_pca = pca.fit_transform(data_reshaped).reshape(h, w, 3)
    label_map = label.reshape(h, w)

    # 提取 Patch 数据
    patches, patch_labels = extract_patches(data_pca, label_map, patch_size=7)

    # 划分训练、验证和测试集（70/15/15）
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        patches, patch_labels, test_size=0.15, random_state=42, stratify=patch_labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.176, random_state=42, stratify=y_train_full
    )

    # 早停机制
    best_val_loss = float('inf')
    patience = 10  # 容忍的 epoch 数
    patience_counter = 0
    best_model_weights = None

    # 转为 TensorDataset
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=512, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
        batch_size=512, shuffle=False
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
        batch_size=512, shuffle=False
    )

    # 模型定义
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN2D(in_channels=3, num_classes=16).to(device)
    optimizer = optim.Adadelta(model.parameters())
    criterion = nn.CrossEntropyLoss()

    # 模型训练
    train_losses = []
    val_losses = []
    print("开始训练 CNN...")
    for epoch in range(1, 201):  # 最多训练 200 轮
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

        # ---- 计算验证集 Loss ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                output = model(X_batch)
                loss = criterion(output, y_batch)
                val_loss += loss.item()
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # ---- 早停判断 ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_weights = model.state_dict()  # 保存当前最佳模型
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"⏹️ Early stopping at epoch {epoch}. Best Val Loss: {best_val_loss:.4f}")
                break

    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)

    # 模型测试
    print("测试模型...")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            output = model(X_batch)
            pred = output.argmax(dim=1)
            correct += (pred == y_batch).sum().item()
            total += y_batch.size(0)

    acc = correct / total * 100
    print(f"✅ Test Accuracy: {acc:.2f}%")

    # === 分类图像可视化 ===
    matplotlib.rcParams['font.family'] = 'Times New Roman'
    pred_map = np.zeros((h, w), dtype=int)
    gt_map = label_map.copy()
    mask = (gt_map != 0)

    for i in range(h):
        for j in range(w):
            if mask[i, j]:
                patch = data_pca[i - 3:i + 4, j - 3:j + 4, :]
                if patch.shape != (7, 7, 3):
                    continue
                patch = np.transpose(patch, (2, 0, 1))
                patch_tensor = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    pred = model(patch_tensor).argmax(dim=1).item()
                pred_map[i, j] = pred + 1

    # 可视化 + 保存图像
    colors = plt.colormaps.get_cmap('tab20')
    cmap = mcolors.ListedColormap(colors.colors)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].imshow(gt_map, cmap=cmap, vmin=1, vmax=16)
    axs[0].set_title("Ground Truth")
    axs[0].axis('off')

    axs[1].imshow(pred_map, cmap=cmap, vmin=1, vmax=16)
    axs[1].set_title(f"Prediction (Acc: {acc:.2f}%)")
    axs[1].axis('off')

    plt.suptitle(f"PaviaU - 2D-CNN Result (Seed {run_seed})", fontsize=14)
    os.makedirs("figures", exist_ok=True)
    plt.savefig(f"figures/PaviaU_2DCNN_run{run_seed}.pdf", bbox_inches='tight')
    plt.close()

    # === Loss 曲线图绘制 ===
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"2D-CNN Training Loss (Seed={run_seed}, Acc={acc:.2f}%)")
    plt.legend()
    plt.grid(True)
    os.makedirs("figures", exist_ok=True)
    plt.savefig(f"figures/2DCNN_loss_seed{run_seed}_pavia.pdf", bbox_inches='tight')
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
with open("results/2dcnn_repeat_results_pavia.txt", "w") as f:
    for i, acc in enumerate(accuracies):
        f.write(f"Run {i+1}: {acc:.2f}%\n")
    f.write(f"\nAverage Accuracy: {mean_acc:.2f}%\n")
    f.write(f"Standard Deviation: {std_acc:.2f}%\n")
