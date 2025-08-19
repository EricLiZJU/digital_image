# comp/HSIMAE/Botswana/model_Botswana.py
import os
import sys
import time
import json
import random
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from ptflops import get_model_complexity_info

# ==== 确保能导入附件里的 HSIMAE 实现 ====
# 如果 Models.py 不在同级目录，请按需修改下行路径（示例：工程根目录）
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
try:
    from Models import HSIViT   # 来自你提供的 HSIMAE 源码（附件 Models.py）
except Exception as e:
    raise ImportError(
        f"无法导入 Models.HSIViT：{e}\n"
        f"请确认 Models.py 在 PYTHONPATH 或与本脚本同级目录。"
    )


# ===================== 通用工具 / 与2D脚本保持一致 =====================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def extract_2d_patches(img_cube, label_map, patch_size=7, ignored_label=0):
    """
    输入 (H, W, C)  → 输出 (N, C, patch, patch)，仅保留中心像素有标签的 patch，标签从0开始
    """
    assert patch_size % 2 == 1, "patch_size must be odd."
    H, W, C = img_cube.shape
    pad = patch_size // 2
    if label_map.shape != (H, W):
        raise ValueError(f"label_map shape {label_map.shape} != image spatial shape {(H, W)}")
    if C < 1:
        raise ValueError("img_cube must have at least 1 channel")
    if np.all(label_map == ignored_label):
        raise ValueError("All labels equal to ignored_label; no samples to extract.")

    padded_img = np.pad(img_cube, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    padded_label = np.pad(label_map, ((pad, pad), (pad, pad)), mode='constant', constant_values=ignored_label)

    patches, labels = [], []
    for i in range(pad, H + pad):
        for j in range(pad, W + pad):
            lab = padded_label[i, j]
            if lab == ignored_label:
                continue
            patch = padded_img[i - pad:i + pad + 1, j - pad:j + pad + 1, :]
            patch = np.transpose(patch, (2, 0, 1))  # (C,H,W)
            patches.append(patch)
            labels.append(int(lab - 1))
    return np.asarray(patches, dtype='float32'), np.asarray(labels, dtype='int64')


def evaluate_and_log_metrics(y_true, y_pred, model, run_seed, dataset_name, acc,
                             in_channels, patch_size, log_root, train_time=None, val_time=None):
    os.makedirs(log_root, exist_ok=True)

    conf_mat = confusion_matrix(y_true, y_pred)
    with np.errstate(divide='ignore', invalid='ignore'):
        per_class_acc = conf_mat.diagonal() / conf_mat.sum(axis=1)
        per_class_acc = np.nan_to_num(per_class_acc, nan=0.0)
    aa = float(per_class_acc.mean())
    kappa = float(cohen_kappa_score(y_true, y_pred))

    with open(os.path.join(log_root, "metrics.json"), "w") as f:
        json.dump({
            "seed": run_seed,
            "dataset": dataset_name,
            "overall_accuracy": round(float(acc), 4),
            "average_accuracy": round(aa, 4),
            "kappa": round(kappa, 4),
            "per_class_accuracy": [round(float(x), 4) for x in per_class_acc.tolist()]
        }, f, indent=2)

    # FLOPs/Params —— 以 (in_channels, P, P) 作为 dummy 输入
    try:
        flops, params = get_model_complexity_info(
            model, (in_channels, patch_size, patch_size),
            as_strings=False, print_per_layer_stat=False
        )
        flops_info = {"FLOPs(M)": round(flops / 1e6, 2), "Params(K)": round(params / 1e3, 2)}
    except Exception:
        total_params = sum(p.numel() for p in model.parameters())
        flops_info = {"FLOPs(M)": None, "Params(K)": round(total_params / 1e3, 2)}

    with open(os.path.join(log_root, "model_profile.json"), "w") as f:
        json.dump(flops_info, f, indent=2)

    with open(os.path.join(log_root, "time_log.json"), "w") as f:
        json.dump({
            "train_time(s)": round(train_time, 2) if train_time else None,
            "val_time(s)": round(val_time, 2) if val_time else None
        }, f, indent=2)

    np.savetxt(os.path.join(log_root, "confusion_matrix.csv"), conf_mat, fmt="%d", delimiter=",")


# ===================== HSIMAE 分类包装器 =====================
class HSIMAEClassifier(nn.Module):
    """
    将 HSIMAE 的 HSIViT 分类分支封装为与你当前训练脚本兼容的模型：
    前向输入: (B, C, P, P) —— 其中 C 是 PCA 后的通道数；P 是 patch_size
    内部把 C 当作 HSIMAE 的 bands（光谱长度），并 reshape 成 (B, 1, T=C, P, P)
    """
    def __init__(
        self,
        in_channels: int,      # = PCA后的通道 (bands)
        num_classes: int,
        patch_size: int = 7,
        embed_dim: int = 128,  # 轻量配置，可按需调大
        depth: int = 6,
        s_depth: int = 0,      # 0 表示不做 dual-branch 的小块循环，简单一些
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        # 关键设定：
        #   img_size = patch_size，patch_size 同为 patch_size —— 这样空间上只有 1×1 个 token
        #   bands = in_channels，b_patch_size = in_channels —— 光谱方向只切 1 个 token
        # 这使得 HSIViT 刚好对一个 patch 做“通道注意力 + 轻量Transformer”分类，且与数据管线完美兼容。
        self.core = HSIViT(
            img_size=patch_size,
            patch_size=patch_size,
            in_chans=1,               # 我们把 (C,P,P) 作为 (T=bands,H,W)，因此输入通道是 1
            embed_dim=embed_dim,
            depth=depth,
            s_depth=s_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=nn.LayerNorm,
            bands=in_channels,        # 光谱长度 = PCA通道数
            b_patch_size=in_channels, # 一次性吃掉全部 bands（确保 T % u == 0）
            num_class=num_classes,
            no_qkv_bias=False,
            trunc_init=False,
            drop_path=drop_path,
        )

    def forward(self, x):
        # x: (B, C, P, P) -> (B, 1, T=C, P, P)
        x = x.unsqueeze(1)
        # HSIViT.forward 返回 (B, num_classes)
        return self.core(x)


# ===================== 训练与评估（与2D脚本一致） =====================
def train_and_evaluate_hsimae(run_seed=42,
                              dataset_name="Botswana",
                              data_path='../../root/data/HSI_dataset/Pavia_university/PaviaU.mat',
                              label_path='/root/data/HSI_dataset/Pavia_university/PaviaU_gt.mat',
                              pca_components=30,
                              patch_size=7,
                              batch_size=128,
                              max_epochs=200,
                              patience=10,
                              lr=1e-3,
                              embed_dim=128,
                              depth=6,
                              s_depth=0,
                              num_heads=8,
                              mlp_ratio=4.0,
                              drop_path=0.0):
    set_seed(run_seed)

    # ---- 数据加载 ----
    img = scio.loadmat(data_path)['paviaU']
    label = scio.loadmat(label_path)['paviaU_gt'].flatten()
    h, w, bands = img.shape
    data_reshaped = img.reshape(h * w, bands)
    label_map = label.reshape(h, w)

    # ---- PCA 降维到 in_channels=C ----
    C = min(pca_components, bands)
    img_2d = img.reshape(-1, bands)
    pca = PCA(n_components=C)
    img_pca = pca.fit_transform(img_2d).reshape(h, w, C)

    # ---- 生成 2D Patch ----
    patches, patch_labels = extract_2d_patches(img_pca, label_map, patch_size=patch_size, ignored_label=0)
    num_classes = int(patch_labels.max()) + 1
    in_channels = C

    # ---- 划分 70/15/15 ----
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        patches, patch_labels, test_size=0.15, stratify=patch_labels, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.176, stratify=y_train_full, random_state=42
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
                            batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
                             batch_size=batch_size, shuffle=False)

    # ---- 模型/优化器/损失 ----
    model = HSIMAEClassifier(
        in_channels=in_channels,
        num_classes=num_classes,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        s_depth=s_depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        drop_path=drop_path,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf'); patience_counter = 0
    best_model_weights = None
    train_losses, val_losses = [], []
    train_start = time.time()

    # ---- 训练 ----
    for epoch in range(1, max_epochs + 1):
        model.train(); train_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(Xb)                 # (B, num_classes)
            loss = criterion(out, yb)
            loss.backward(); optimizer.step()
            train_loss += loss.item()

        model.eval(); val_loss = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                val_loss += criterion(model(Xb), yb).item()

        print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        train_losses.append(train_loss); val_losses.append(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss; best_model_weights = model.state_dict(); patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"⏹️ Early stopping at epoch {epoch}")
                break

    train_time = time.time() - train_start

    # ---- 测试 ----
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
    model.eval(); all_true, all_pred = [], []
    val_start = time.time()
    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb = Xb.to(device)
            preds = model(Xb).argmax(dim=1)
            all_true.extend(yb.numpy()); all_pred.extend(preds.cpu().numpy())
    acc = np.mean(np.array(all_true) == np.array(all_pred)) * 100.0
    val_time = time.time() - val_start
    print(f"✅ Test Accuracy: {acc:.2f}%")

    # ---- 日志 ----
    log_root = f"comp_logs/HSIMAE/{dataset_name}/seed_{run_seed}"
    os.makedirs(log_root, exist_ok=True)
    with open(os.path.join(log_root, "loss_curve.json"), "w") as f:
        json.dump({
            "train_loss": [round(float(l), 4) for l in train_losses],
            "val_loss": [round(float(l), 4) for l in val_losses]
        }, f, indent=2)

    evaluate_and_log_metrics(
        y_true=np.array(all_true),
        y_pred=np.array(all_pred),
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

    return acc


if __name__ == "__main__":
    # ====== 数据集配置（按需修改） ======
    dataset_name = "PaviaU"

    pca_components = 30
    patch_size = 7
    repeats = 1

    accs = []
    for i in range(repeats):
        seed = i * 10 + 42
        print(f"\n🔁 Running trial {i+1} with seed {seed}")
        acc = train_and_evaluate_hsimae(
            run_seed=seed,
            dataset_name=dataset_name,
            pca_components=pca_components,
            patch_size=patch_size,
            batch_size=128,
            max_epochs=200,
            patience=10,
            lr=1e-3,
            embed_dim=128,    # 可调（变大更强、也更慢）
            depth=6,
            s_depth=0,        # 如想启用 dual-branch 的小块堆叠，可设 >0
            num_heads=8,
            mlp_ratio=4.0,
            drop_path=0.0,
        )
        accs.append(acc)

    print("\n📊 Summary of Repeated Training:")
    for i, acc in enumerate(accs):
        print(f"Run {i+1}: {acc:.2f}%")
    print(f"\nAverage Accuracy: {np.mean(accs):.2f}%, Std Dev: {np.std(accs):.2f}%")