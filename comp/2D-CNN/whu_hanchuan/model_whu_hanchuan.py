# train_3dcnn_with_logging.py
import os
import time
import json
import random
import torch
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.io import loadmat

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from ptflops import get_model_complexity_info
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CNN2D(nn.Module):
    def __init__(self, in_channels=3, num_classes=16):
        super(CNN2D, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),  # (C,7,7)->(32,7,7)
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2),  # (32,7,7)->(32,3,3)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 3 * 3, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.7),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def extract_2d_patches(img_cube, label_map, patch_size=7, ignored_label=0):
    """
    2D Patch 提取：输入 (H, W, C) -> 输出 (N, C, H, W)
    仅对中心像素有标签的 patch 进行采样；标签从0开始（剔除ignored_label=0）
    """
    assert patch_size % 2 == 1, "patch_size must be odd."
    H, W, C = img_cube.shape
    pad = patch_size // 2

    # 基本健壮性检查
    if label_map.shape != (H, W):
        raise ValueError(f"label_map shape {label_map.shape} != image spatial shape {(H, W)}")
    if C < 1:
        raise ValueError("img_cube must have at least 1 channel")
    if np.all(label_map == ignored_label):
        raise ValueError("All labels equal to ignored_label; no samples to extract.")

    # 边界填充
    padded_img = np.pad(img_cube, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    padded_label = np.pad(label_map, ((pad, pad), (pad, pad)), mode='constant', constant_values=ignored_label)

    # ✅ 正确的初始化
    patches = []
    labels = []

    for i in range(pad, H + pad):
        for j in range(pad, W + pad):
            lab = padded_label[i, j]
            if lab == ignored_label:
                continue
            patch = padded_img[i - pad:i + pad + 1, j - pad:j + pad + 1, :]
            # (H, W, C) -> (C, H, W)
            patch = np.transpose(patch, (2, 0, 1))
            patches.append(patch)
            labels.append(lab - 1)  # 标签从0开始

    patches = np.asarray(patches, dtype='float32')
    labels = np.asarray(labels, dtype='int64')

    if len(labels) == 0:
        raise ValueError("No labeled patches extracted. Check label_map and ignored_label setting.")

    return patches, labels


def evaluate_and_log_metrics_2d(y_true, y_pred, model, run_seed, dataset_name, acc,
                                in_channels, patch_size, log_root, train_time=None, val_time=None):
    log_dir = log_root
    os.makedirs(log_dir, exist_ok=True)

    conf_mat = confusion_matrix(y_true, y_pred)
    per_class_acc = conf_mat.diagonal() / conf_mat.sum(axis=1)
    aa = per_class_acc.mean()
    kappa = cohen_kappa_score(y_true, y_pred)

    metrics = {
        "seed": run_seed,
        "dataset": dataset_name,
        "overall_accuracy": round(acc, 4),
        "average_accuracy": round(aa, 4),
        "kappa": round(kappa, 4),
        "per_class_accuracy": [round(float(x), 4) for x in per_class_acc.tolist()]
    }
    with open(os.path.join(log_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # FLOPs/Params
    flops, params = get_model_complexity_info(
        model, (in_channels, patch_size, patch_size),
        as_strings=False, print_per_layer_stat=False
    )
    flops_info = {"FLOPs(M)": round(flops / 1e6, 2), "Params(K)": round(params / 1e3, 2)}
    with open(os.path.join(log_dir, "model_profile.json"), "w") as f:
        json.dump(flops_info, f, indent=2)

    time_info = {
        "train_time(s)": round(train_time, 2) if train_time else None,
        "val_time(s)": round(val_time, 2) if val_time else None
    }
    with open(os.path.join(log_dir, "time_log.json"), "w") as f:
        json.dump(time_info, f, indent=2)

    np.savetxt(os.path.join(log_dir, "confusion_matrix.csv"), conf_mat, fmt="%d", delimiter=",")



def train_and_evaluate_2d(run_seed=42,
                          dataset_name="Botswana",
                          data_path='../../root/data/HSI_dataset/Matlab_data_format/Matlab_data_format/WHU-Hi-HanChuan/WHU_Hi_HanChuan.mat',
                          label_path = '../../root/data/HSI_dataset/Matlab_data_format/Matlab_data_format/WHU-Hi-HanChuan/WHU_Hi_HanChuan_gt.mat',
                          pca_components=3,           # 为公平可改为30
                          patch_size=7,
                          batch_size=128,
                          max_epochs=200,
                          patience=10,
                          lr=1e-3):
    set_seed(run_seed)
    # === 读取并自动对齐 H/W/C 轴 ===
    hyperspec = loadmat(data_path)
    data = hyperspec['WHU_Hi_HanChuan']  # 原始是 3D，但轴顺序未知
    gt = loadmat(label_path)
    label_map = gt['WHU_Hi_HanChuan_gt']
    label_map = np.squeeze(label_map)  # 保证是 2D

    if data.ndim != 3:
        raise ValueError(f"HSI data must be 3D, got shape {data.shape}")

    # 尝试所有轴排列，找到与 label_map.shape 匹配的空间维（必要时转置 label_map）
    perms = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
    chosen = None
    label_transposed = False
    for p in perms:
        cand = np.transpose(data, p)  # (A,B,C) 含义待定
        H, W, C = cand.shape
        if (H, W) == label_map.shape:
            chosen = cand
            break
        if (W, H) == label_map.shape:
            chosen = cand
            label_map = label_map.T  # 让 label 的空间维与数据一致
            label_transposed = True
            break

    if chosen is None:
        raise ValueError(
            f"Cannot align data spatial dims with label_map.\n"
            f"data original shape={data.shape}, label_map shape={label_map.shape}.\n"
            f"Try checking the dataset keys or axis order."
        )

    data = chosen  # 确保 data 现在是 (H, W, C)
    H, W, bands = data.shape
    print(f"[Axis-Aligned] data shape -> (H,W,C)=({H},{W},{bands}), "
          f"label_map shape -> {label_map.shape}, "
          f"label_transposed={label_transposed}")

    # 背景/无效标签值（WHU 数据通常 0 为背景）
    ignored_label = 0

    # ---- PCA 降维到 C 通道（默认3；可设为30以公平对比）----
    data_reshaped = data.reshape(H * W, bands)
    pca = PCA(n_components=min(pca_components, bands))
    data_pca = pca.fit_transform(data_reshaped)
    data_cube = data_pca.reshape(H, W, min(pca_components, bands))

    # ---- 生成 2D Patch ----
    patches, patch_labels = extract_2d_patches(
        data_cube, label_map, patch_size=patch_size, ignored_label=0
    )
    num_classes = int(patch_labels.max()) + 1
    in_channels = pca_components

    # ---- 划分 70/15/15（与3D脚本保持一致随机种子与比例）----
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        patches, patch_labels, test_size=0.15, stratify=patch_labels, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.176, stratify=y_train_full, random_state=42
    )

    # ---- Dataloader ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
                            batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
                             batch_size=batch_size, shuffle=False)

    # ---- 模型&优化器 ----
    model = CNN2D(in_channels=in_channels, num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_weights = None
    train_losses, val_losses = [], []

    # ---- 训练 ----
    train_start = time.time()
    for epoch in range(1, max_epochs + 1):
        # Train
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

        # Val
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                val_loss += criterion(model(X_batch), y_batch).item()

        print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"⏹️ Early stopping at epoch {epoch}")
                break

    train_end = time.time()
    train_time = train_end - train_start

    # ---- 测试 ----
    val_start = time.time()
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
    model.eval()
    all_true_labels, all_pred_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = model(X_batch).argmax(dim=1)
            all_true_labels.extend(y_batch.cpu().numpy())
            all_pred_labels.extend(preds.cpu().numpy())

    acc = np.mean(np.array(all_true_labels) == np.array(all_pred_labels)) * 100.0
    print(f"✅ Test Accuracy: {acc:.2f}%")
    val_end = time.time()
    val_time = val_end - val_start

    # ---- 日志 & 曲线 ----
    log_root = f"comp_logs/2D-CNN/{dataset_name}/seed_{run_seed}"
    os.makedirs(log_root, exist_ok=True)

    # 保存loss曲线
    loss_log = {
        "train_loss": [round(float(l), 4) for l in train_losses],
        "val_loss": [round(float(l), 4) for l in val_losses]
    }
    with open(os.path.join(log_root, "loss_curve.json"), "w") as f:
        json.dump(loss_log, f, indent=2)

    # 评估指标/FLOPs/时间/混淆矩阵
    evaluate_and_log_metrics_2d(
        y_true=np.array(all_true_labels),
        y_pred=np.array(all_pred_labels),
        model=model,
        run_seed=run_seed,
        dataset_name=str(dataset_name),
        acc=acc,
        in_channels=in_channels,
        patch_size=patch_size,
        train_time=train_time,
        val_time=val_time,
        log_root=log_root,
    )

    # ---- 分类图像可视化 ----
    print("🖼️ Generating classification maps...")
    pred_map = np.zeros((H, W), dtype=int)
    gt_map = label_map.copy()
    mask = (gt_map != 0)

    for i in range(H):
        for j in range(W):
            if mask[i, j]:
                patch = data_cube[i - (patch_size // 2): i + (patch_size // 2) + 1,
                        j - (patch_size // 2): j + (patch_size // 2) + 1, :]
                if patch.shape != (patch_size, patch_size, in_channels):
                    continue
                patch = np.transpose(patch, (2, 0, 1))  # (C,H,W)
                patch_tensor = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    pred_label = model(patch_tensor).argmax(dim=1).item()
                pred_map[i, j] = pred_label + 1  # 可视化从1开始

    # 颜色映射
    num_classes = int(patch_labels.max()) + 1
    cmap = mcolors.ListedColormap(plt.colormaps['tab20'].colors[:num_classes])

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    axs[0].imshow(gt_map, cmap=cmap, vmin=1, vmax=num_classes)
    axs[0].set_title("Ground Truth")
    axs[0].axis('off')

    axs[1].imshow(pred_map, cmap=cmap, vmin=1, vmax=num_classes)
    axs[1].set_title(f"Prediction (Acc: {acc:.2f}%)")
    axs[1].axis('off')

    fig_path = os.path.join(log_root, f"{dataset_name}_2DCNN_run{run_seed}_vis.png")
    fig_path_pdf = os.path.join(log_root, f"{dataset_name}_2DCNN_run{run_seed}_vis.pdf")
    plt.savefig(fig_path, bbox_inches='tight', dpi=300)
    plt.savefig(fig_path_pdf, bbox_inches='tight')
    plt.close()
    print(f"✅ Classification map saved to:\n  {fig_path}\n  {fig_path_pdf}")

    return acc


if __name__ == "__main__":
    # ====== 根据你的数据集修改 ======
    dataset_name = "whu_hanchuan"

    # 公平对比建议：
    #   1) 先用 pca_components=3 跑常规2D；再用 pca_components=30 与3D保持一致的通道设定做第二组实验
    pca_components = 3       # 可改为 30
    patch_size = 7
    repeats = 10

    accs = []
    for i in range(repeats):
        seed = i * 10 + 42
        print(f"\n🔁 Running trial {i+1} with seed {seed}")
        acc = train_and_evaluate_2d(
            run_seed=seed,
            dataset_name=dataset_name,
            pca_components=pca_components,
            patch_size=patch_size,
            batch_size=128,
            max_epochs=200,
            patience=10,
            lr=1e-3
        )
        accs.append(acc)

    print("\n📊 Summary of Repeated Training:")
    for i, acc in enumerate(accs):
        print(f"Run {i+1}: {acc:.2f}%")
    print(f"\nAverage Accuracy: {np.mean(accs):.2f}%, Std Dev: {np.std(accs):.2f}%")