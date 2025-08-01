import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import scipy.io as scio
import numpy as np
import random
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os
import matplotlib.colors as mcolors
import matplotlib


# 设置随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Transformer模型类
class TransformerModel(nn.Module):
    def __init__(self, input_dim, embedding_dim, num_heads, num_layers, num_classes, dropout=0.1):
        self.input_dim = input_dim
        super(TransformerModel, self).__init__()

        # 输入嵌入层
        self.embedding = nn.Linear(input_dim, embedding_dim)

        # Positional Encoding：为每个patch添加空间信息
        self.pos_encoding = nn.Parameter(torch.zeros(1, 49, embedding_dim))  # max_len = 49

        # Transformer编码器层
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=2048,
            dropout=dropout
        )

        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        # 分类头
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        # x: (batch, 1, C, H, W)
        x = x.squeeze(1)  # (batch, C, H, W)
        x = x.permute(0, 2, 3, 1)  # (batch, H, W, C)
        x = x.view(x.size(0), -1, self.input_dim)  # (batch, H*W, C)

        x = self.embedding(x)  # Linear projection: (batch, N, embed_dim)
        x = x + self.pos_encoding[:, :x.size(1), :]  # Add positional encoding if needed
        x = self.encoder(x)  # Transformer encoder
        x = x.mean(dim=1)  # Global average pooling
        x = self.fc(x)  # Final classification
        return x


# 3D Patch提取函数
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


# 训练和评估函数
def train_and_evaluate(run_seed=42):
    set_seed(run_seed)

    # -------- 数据准备 --------
    h, w = 145, 145
    data_path = 'data/Indian_pines/Indian_pines_corrected.mat'
    label_path = 'data/Indian_pines/Indian_pines_gt.mat'
    data = scio.loadmat(data_path)['indian_pines_corrected'].reshape(-1, 200)
    label = scio.loadmat(label_path)['indian_pines_gt'].flatten()

    data_reshaped = data.reshape(h * w, 200)
    data_cube = data.reshape(h, w, 200)
    label_map = label.reshape(h, w)

    patches, patch_labels = extract_3d_patches(data_cube, label_map, patch_size=7, spectral_channels=30)

    # 划分训练、验证、测试集
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
        batch_size=128, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
        batch_size=128, shuffle=False
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
        batch_size=128, shuffle=False
    )

    # 模型定义
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerModel(
        input_dim=30,  # 每个patch有30个光谱通道
        embedding_dim=128,
        num_heads=4,
        num_layers=2,
        num_classes=16,
        dropout=0.3
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    criterion = nn.CrossEntropyLoss()

    # 模型训练
    train_losses = []
    val_losses = []
    print("开始训练 Transformer...")
    for epoch in range(1, 201):  # 最多训练200轮
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

        # 计算验证集 Loss
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

        # 早停判断
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
    mask = (label_map != 0)

    for i in range(h):
        for j in range(w):
            if mask[i, j]:
                patch = data_cube[i - 3:i + 4, j - 3:j + 4, :30]
                if patch.shape != (7, 7, 30):
                    continue
                patch = np.transpose(patch, (2, 0, 1))  # (C, H, W)
                patch = patch[np.newaxis, np.newaxis, ...]  # (1, 1, C, H, W)
                patch_tensor = torch.tensor(patch, dtype=torch.float32).to(device)

                with torch.no_grad():
                    output = model(patch_tensor)
                    pred = output.argmax(dim=1).item()
                pred_map[i, j] = pred + 1  # 标签从1开始

    # 可视化 + 保存图像
    colors = plt.colormaps.get_cmap('tab20')
    cmap = mcolors.ListedColormap(colors.colors)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].imshow(label_map, cmap=cmap, vmin=1, vmax=16)
    axs[0].set_title("Ground Truth")
    axs[0].axis('off')

    axs[1].imshow(pred_map, cmap=cmap, vmin=1, vmax=16)
    axs[1].set_title(f"Transformer Prediction (Acc: {acc:.2f}%)")
    axs[1].axis('off')

    os.makedirs("figures", exist_ok=True)
    plt.savefig(f"figures/Transformer_run{run_seed}.pdf", bbox_inches='tight')
    plt.close()

    # === 损失曲线图保存 ===
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Transformer Training Loss (Seed={run_seed}, Acc={acc:.2f}%)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"figures/Transformer_loss_seed{run_seed}.pdf", bbox_inches='tight')
    plt.close()

    # 返回精度和训练过程中的损失
    return acc, train_losses, val_losses


# 多次重复训练
repeats = 10
accuracies = []
train_losses_all = []
val_losses_all = []

for i in range(repeats):
    print(f"\n🔁 Running trial {i+1} with seed {i*10+42}")
    acc, train_losses, val_losses = train_and_evaluate(run_seed=i*10+42)
    print(f"✅ Accuracy for run {i+1}: {acc:.2f}%")
    accuracies.append(acc)
    train_losses_all.append(train_losses)
    val_losses_all.append(val_losses)

# 计算统计指标
mean_acc = np.mean(accuracies)
std_acc = np.std(accuracies)

# 输出与保存
print("\n📊 Summary of Repeated Training:")
for i, acc in enumerate(accuracies):
    print(f"Run {i+1}: {acc:.2f}%")
print(f"\nAverage Accuracy: {mean_acc:.2f}%, Std Dev: {std_acc:.2f}%")

os.makedirs("results", exist_ok=True)
with open("results/transformer_repeat_results.txt", "w") as f:
    for i, acc in enumerate(accuracies):
        f.write(f"Run {i+1}: {acc:.2f}%\n")
    f.write(f"\nAverage Accuracy: {mean_acc:.2f}%\n")
    f.write(f"Standard Deviation: {std_acc:.2f}%\n")

