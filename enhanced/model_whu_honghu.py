from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import numpy as np
import scipy.io as scio
import random
from scipy.io import loadmat
import h5py
import torch
# 模型构建
import torch.nn as nn
# 模型训练
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # 如果使用的是 CUDA
    torch.backends.cudnn.benchmark = False

class CNN3D(nn.Module):
    def __init__(self, in_channels=30, num_classes=16, use_attention=True):
        super(CNN3D, self).__init__()

        # 空谱分离模块
        self.spectral_conv = nn.Conv3d(1, 8, kernel_size=(7,1,1), padding=(3,0,0))  # 先提取光谱特征
        self.spatial_conv = nn.Conv3d(8, 8, kernel_size=(1,3,3), padding=(0,1,1))  # 再提取空间特征
        self.bn1 = nn.BatchNorm3d(8)
        self.relu = nn.ReLU()

        # 频谱冗余抑制模块（注意力机制）
        self.use_attention = use_attention
        if self.use_attention:
            att_in_channels = 8  # 注意这个值必须和 spatial_conv 输出通道一致
            self.attention = nn.Sequential(
                nn.AdaptiveAvgPool3d((1, 1, 1)),  # shape: (B, 8, 1, 1, 1)
                nn.Conv3d(att_in_channels, att_in_channels // 2, kernel_size=1),
                nn.ReLU(),
                nn.Conv3d(att_in_channels // 2, att_in_channels, kernel_size=1),
                nn.Sigmoid()
            )

        # 第二层卷积提取更高层次特征
        self.conv2 = nn.Conv3d(8, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(16)

        # 自适应池化 + 全连接分类器
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))  # 输出 (batch, 16, 1, 1, 1)
        self.fc = nn.Sequential(
            nn.Flatten(),                 # (batch, 16)
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):  # x: (batch, 1, C, H, W)
        x = self.spectral_conv(x)
        x = self.spatial_conv(x)
        x = self.bn1(x)
        x = self.relu(x)

        # 注意力加权（频谱冗余抑制）
        if self.use_attention:
            att_map = self.attention(x)  # shape: (batch, C, 1, 1, 1)
            x = x * att_map

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.pool(x)
        x = self.fc(x)
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

def select_fixed_samples_per_class(X, y, num_per_class, seed=42):
    """
    从每个类别中固定抽取 num_per_class 个样本，用于小样本实验
    """
    random.seed(seed)
    class_to_indices = defaultdict(list)

    for idx, label in enumerate(y):
        class_to_indices[label].append(idx)

    selected_indices = []
    for label, indices in class_to_indices.items():
        if len(indices) < num_per_class:
            raise ValueError(f"Class {label} has only {len(indices)} samples, less than {num_per_class}.")
        selected = random.sample(indices, num_per_class)
        selected_indices.extend(selected)

    X_selected = X[selected_indices]
    y_selected = y[selected_indices]
    return X_selected, y_selected

file_path = '../../root/data/HSI_dataset/Matlab_data_format/Matlab_data_format/WHU-Hi-HongHu/WHU_Hi_HongHu.mat'

# 加载高光谱图像
hyperspec = loadmat(file_path)
data = hyperspec['WHU_Hi_HongHu']  # shape: (1217, 303, 274)

# 转置为 (H, W, C)
data = np.transpose(data, (1, 2, 0))  # (303, 274, 1217)

# 加载标签
gt = loadmat('../../root/data/HSI_dataset/Matlab_data_format/Matlab_data_format/WHU-Hi-HongHu/WHU_Hi_HongHu_gt.mat')
label_map = gt['WHU_Hi_HongHu_gt']  # shape = (2517, 2335)

# 转置标签，使其与 data 对齐
label_map = label_map.T  # → (2335, 2517) → (H, W)
# Step 1: 获取有效标签（排除背景等）
ignored_label = 0  # 可改为255或-1，取决于你数据集中表示“无效”的标签值
unique_labels = np.unique(label_map)
valid_labels = unique_labels[unique_labels != ignored_label]

# Step 2: 构造映射表（原始类 → 新类索引）
label_mapping = {original: new for new, original in enumerate(valid_labels)}
num_classes = len(valid_labels)
print(f"Detected {num_classes} valid classes.")

# 应用映射，所有不在valid_labels中的都将被映射为 -1（ignored）
def map_label(l):
    return label_mapping.get(l, -1)

vectorized_map = np.vectorize(map_label)
label_map_mapped = vectorized_map(label_map)  # 新标签映射图

# 最终 shape 检查
print("data shape:", data.shape)         # should be (H, W, C)
print("label_map shape:", label_map.shape)  # should match (H, W)


h, w, bands = data.shape  # new height, width, channels

def train_and_evaluate(run_seed=42):
    set_seed(run_seed)

    # -------- 数据准备 --------
    data_reshaped = data.reshape(h * w, bands)

    # --- PCA: 从200通道降到30通道 ---
    pca = PCA(n_components=30)
    data_pca = pca.fit_transform(data_reshaped)  # shape: (h*w, 30)
    data_cube = data_pca.reshape(h, w, 30)  # shape: (145, 145, 30)

    # --- 提取 3D patch 数据 ---
    patches, patch_labels = extract_3d_patches(
        data_cube, label_map_mapped,
        patch_size=7, spectral_channels=30,
        ignored_label=-1
    )
    print(f"Patch labels: min={patch_labels.min()}, max={patch_labels.max()}")
    print(f"num_classes = {num_classes}")
    print(f"[Debug] patch_labels range: {patch_labels.min()} ~ {patch_labels.max()}")
    valid_idx = (patch_labels >= 0) & (patch_labels < num_classes)
    patches = patches[valid_idx]
    patch_labels = patch_labels[valid_idx]

    # 划分训练、验证、测试集
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        patches, patch_labels, test_size=0.15, random_state=42, stratify=patch_labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.176, random_state=42, stratify=y_train_full
    )

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=1024,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)), batch_size=128,
                            shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)), batch_size=128,
                             shuffle=False)

    # -------- 模型定义 --------
    # device = torch.device("cuda" if torch.backends.mps.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = CNN3D(in_channels=30, num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # -------- 训练阶段 --------
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    train_losses, val_losses = [], []

    for epoch in range(1, 201):
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
    plt.savefig(f"figures/Pavia_3DCNN_run{run_seed}.pdf", bbox_inches='tight')
    plt.close()

    # -------- Loss 曲线 --------
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"3D-CNN Loss Curve (Seed={run_seed})")
    plt.legend()
    os.makedirs("figures", exist_ok=True)
    plt.savefig(f"figures/3DCNN_loss_seed{run_seed}_pavia.pdf", bbox_inches='tight')
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
with open("results/3dcnn_repeat_results_pavia.txt", "w") as f:
    for i, acc in enumerate(accuracies):
        f.write(f"Run {i+1}: {acc:.2f}%\n")
    f.write(f"\nAverage Accuracy: {mean_acc:.2f}%\n")
    f.write(f"Standard Deviation: {std_acc:.2f}%\n")