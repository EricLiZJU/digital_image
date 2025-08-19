# comp/MambaHSI/Botswana/model_Botswana.py
import os
import time
import json

import h5py
import math
import random
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import loadmat
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from ptflops import get_model_complexity_info


# ===================== 通用工具 / 与2D脚本保持一致 =====================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def extract_2d_patches(img_cube, label_map, patch_size=7, ignored_label=0):
    assert patch_size % 2 == 1, "patch_size must be odd."
    H, W, C = img_cube.shape
    pad = patch_size // 2
    if label_map.shape != (H, W):
        raise ValueError(f"label_map shape {label_map.shape} != {(H, W)}")
    if C < 1:
        raise ValueError("img_cube must have at least 1 channel")
    if np.all(label_map == ignored_label):
        raise ValueError("All labels are ignored; no samples.")

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
            labels.append(lab - 1)
    return np.asarray(patches, dtype='float32'), np.asarray(labels, dtype='int64')


def evaluate_and_log_metrics_2d(y_true, y_pred, model, run_seed, dataset_name, acc,
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

    # FLOPs/Params
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


# ===================== 伪 Mamba（无需安装 mamba-ssm） =====================
# 若以后成功安装 mamba-ssm，可删掉这个类并 `from mamba_ssm import Mamba`
class Mamba(nn.Module):
    """
    轻量“伪 Mamba”：保持输入输出长度一致的线性 + 深度可分卷积近似。
    仅为跑通/对比使用，非官方实现。
    输入: (B, L, C)  -> 输出: (B, L, C)
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        hidden = d_model * expand
        self.proj_in  = nn.Linear(d_model, hidden)
        self.dwconv   = nn.Conv1d(hidden, hidden, kernel_size=d_conv,
                                  padding=d_conv // 2, groups=hidden)
        self.act      = nn.GELU()
        self.proj_out = nn.Linear(hidden, d_model)

    def forward(self, x):
        # x: (B, L, C)
        y = self.proj_in(x)          # (B, L, hidden)
        y = y.transpose(1, 2)        # (B, hidden, L)
        y = self.dwconv(y)           # (B, hidden, L)
        y = y.transpose(1, 2)        # (B, L, hidden)
        y = self.act(y)
        y = self.proj_out(y)         # (B, L, C)
        return y


# ===================== MambaHSI 模型（带形状健壮性处理） =====================
class SpeMamba(nn.Module):
    """
    频域 Mamba（安全版）：
    - 不把通道永久补到更大，只在内部 reshape 时临时补齐到 token_len * group_ch，然后最后裁回 C。
    - 残差和归一化都严格用原始 C 通道，避免 x + y 维度不一致。
    """
    def __init__(self, channels, token_num=8, use_residual=True, group_num=4):
        super().__init__()
        self.use_residual = use_residual
        self.C = channels

        # 让单个 token 的通道数尽量接近 C / token_num，但保证 >=1
        self.group_ch = max(1, channels // max(1, token_num))
        # 计算需要多少个 token 才能覆盖全部通道
        self.token_len = math.ceil(channels / self.group_ch)
        # 内部临时对齐到 token_len * group_ch
        self.C_int = self.token_len * self.group_ch

        self.mamba = Mamba(
            d_model=self.group_ch,
            d_state=16,
            d_conv=4,
            expand=2,
        )
        # 归一化严格用真实通道数 C
        self.proj = nn.Sequential(
            nn.GroupNorm(group_num, self.C),
            nn.SiLU()
        )

    def forward(self, x):
        """
        x: (B, C, H, W)  ->  (B, C, H, W)
        """
        B, C, H, W = x.shape
        assert C == self.C, f"SpeMamba got C={C}, expected {self.C}"

        # 临时补齐到整数 token_len * group_ch
        if C < self.C_int:
            pad_c = self.C_int - C
            pad = torch.zeros((B, pad_c, H, W), device=x.device, dtype=x.dtype)
            x_pad = torch.cat([x, pad], dim=1)  # (B, C_int, H, W)
        else:
            x_pad = x

        # (B, C_int, H, W) -> (B*H*W, token_len, group_ch)
        x_re = x_pad.permute(0, 2, 3, 1).contiguous()
        BHw = B * H * W
        x_seq = x_re.view(BHw, self.token_len, self.group_ch)

        # Mamba
        y_seq = self.mamba(x_seq)  # (BHW, token_len, group_ch)

        # 动态获取输出 token 数和 group_ch
        t_len, g_ch = y_seq.shape[1], y_seq.shape[2]
        c_out = t_len * g_ch

        # 还原空间结构
        y = y_seq.view(B, H, W, c_out).permute(0, 3, 1, 2).contiguous()  # (B, c_out, H, W)

        # 裁回真实通道数
        if c_out >= self.C:
            y = y[:, :self.C, :, :]
        else:
            # 如果输出通道比 C 少，就补零
            pad_c = self.C - c_out
            pad = torch.zeros((B, pad_c, H, W), device=x.device, dtype=x.dtype)
            y = torch.cat([y, pad], dim=1)

        # 投影 & 残差
        y = self.proj(y)
        return x + y if self.use_residual else y


class SpaMamba(nn.Module):
    def __init__(self, channels, use_residual=True, group_num=4, use_proj=True):
        super().__init__()
        self.use_residual = use_residual
        self.use_proj = use_proj
        self.mamba = Mamba(d_model=channels, d_state=16, d_conv=4, expand=2)
        if use_proj:
            self.proj = nn.Sequential(nn.GroupNorm(group_num, channels), nn.SiLU())

    def forward(self, x):
        # (B,C,H,W) -> (1, L=B*H*W, C) -> Mamba -> reshape back
        B, C, H, W = x.shape
        x_re = x.permute(0, 2, 3, 1).contiguous().view(1, B * H * W, C)  # (1, L, C)
        y = self.mamba(x_re)                                             # (1, L, C)
        # —— 健壮性：若某些实现导致 L 变化，强制对齐回 L_in —— #
        L_in, L_out = B * H * W, y.size(1)
        if L_out != L_in:
            if L_out > L_in:
                y = y[:, :L_in, :]
            else:
                pad = torch.zeros((1, L_in - L_out, C), device=y.device, dtype=y.dtype)
                y = torch.cat([y, pad], dim=1)
        y = y.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()          # (B,C,H,W)
        if self.use_proj:
            y = self.proj(y)
        return y + x if self.use_residual else y


class BothMamba(nn.Module):
    def __init__(self, channels, token_num, use_residual, group_num=4, use_att=True):
        super().__init__()
        self.use_att = use_att
        self.use_residual = use_residual
        if use_att:
            self.weights = nn.Parameter(torch.ones(2) / 2)
            self.softmax = nn.Softmax(dim=0)
        self.spa_m = SpaMamba(channels, use_residual=use_residual, group_num=group_num)
        self.spe_m = SpeMamba(channels, token_num=token_num, use_residual=use_residual, group_num=group_num)

    def forward(self, x):
        spa_x = self.spa_m(x)
        spe_x = self.spe_m(x)
        if self.use_att:
            w = self.softmax(self.weights)
            y = spa_x * w[0] + spe_x * w[1]
        else:
            y = spa_x + spe_x
        return y + x if self.use_residual else y


class MambaHSI(nn.Module):
    def __init__(self, in_channels=30, hidden_dim=64, num_classes=16,
                 use_residual=True, mamba_type='both', token_num=4, group_num=4, use_att=True):
        super().__init__()
        self.mamba_type = mamba_type
        self.embed = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1, 1, 0),
            nn.GroupNorm(group_num, hidden_dim),
            nn.SiLU()
        )
        if mamba_type == 'spa':
            self.mamba = nn.Sequential(
                SpaMamba(hidden_dim, use_residual, group_num),
                nn.AvgPool2d(2),
                SpaMamba(hidden_dim, use_residual, group_num),
                nn.AvgPool2d(2),
                SpaMamba(hidden_dim, use_residual, group_num)
            )
        elif mamba_type == 'spe':
            self.mamba = nn.Sequential(
                SpeMamba(hidden_dim, token_num, use_residual, group_num),
                nn.AvgPool2d(2),
                SpeMamba(hidden_dim, token_num, use_residual, group_num),
                nn.AvgPool2d(2),
                SpeMamba(hidden_dim, token_num, use_residual, group_num)
            )
        else:
            self.mamba = nn.Sequential(
                BothMamba(hidden_dim, token_num, use_residual, group_num, use_att),
                nn.AvgPool2d(2),
                BothMamba(hidden_dim, token_num, use_residual, group_num, use_att),
                nn.AvgPool2d(2),
                BothMamba(hidden_dim, token_num, use_residual, group_num, use_att)
            )
        self.cls_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 128, 1), nn.GroupNorm(group_num, 128), nn.SiLU(),
            nn.Conv2d(128, num_classes, 1)
        )

    def forward(self, x):
        # x : (B, C_in, P, P)
        x = self.embed(x)
        x = self.mamba(x)          # 尺寸：P=7 -> 7->3->1（两次池化），输出 (B, hidden, 1, 1)
        logits_map = self.cls_head(x)  # (B, num_classes, H', W')
        # 取中心像素 logits，输出 (B, num_classes) 以适配交叉熵
        B, C, H, W = logits_map.shape
        c_h, c_w = H // 2, W // 2
        logits = logits_map[:, :, c_h, c_w]
        return logits


# ===================== 训练与评估（与2D脚本一致） =====================
def train_and_evaluate_mambahsi(run_seed=42,
                                dataset_name="Botswana",
                                data_path='../../root/data/HSI_dataset/Chikusei_MATLAB/HyperspecVNIR_Chikusei_20140729.mat',
                                label_path='../../root/data/HSI_dataset/Chikusei_MATLAB/HyperspecVNIR_Chikusei_20140729_Ground_Truth.mat',
                                pca_components=30,
                                patch_size=7,
                                batch_size=64,
                                max_epochs=200,
                                patience=10,
                                lr=1e-3,
                                mamba_type='both'):
    set_seed(run_seed)

    # 数据加载
    with h5py.File(data_path, 'r') as f:
        # 通常 v7.3 的数据是以 dataset 名义存储，检查键名（比如 'chikusei'）
        print("Keys:", list(f.keys()))
        dset = f['chikusei']
        data = np.array(dset).astype(np.float32)  # shape: (128, 2335, 2517)
    data = np.moveaxis(data, 0, -1)  # → (2335, 2517, 128)
    gt = loadmat(label_path)
    label_map = gt['GT']['gt'][0, 0]
    label_map = label_map.T
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
    print("data shape:", data.shape)  # should be (H, W, C)
    print("label_map shape:", label_map.shape)  # should match (H, W)
    h, w, bands = data.shape
    data_reshaped = data.reshape(h * w, bands)
    data_pca = PCA(n_components=min(pca_components, bands)).fit_transform(
        data.reshape(-1, bands)).reshape(h, w, min(pca_components, bands))

    patches, patch_labels = extract_2d_patches(data_pca, label_map, patch_size=patch_size)
    num_classes = int(patch_labels.max()) + 1
    in_channels = data_pca.shape[2]

    # 划分 70/15/15
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        patches, patch_labels, test_size=0.15, stratify=patch_labels, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.176, stratify=y_train_full, random_state=42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
                            batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
                             batch_size=batch_size, shuffle=False)

    # 模型/优化器/损失
    model = MambaHSI(in_channels=in_channels, num_classes=num_classes, mamba_type=mamba_type).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf'); patience_counter = 0
    best_model_weights = None
    train_losses, val_losses = [], []
    train_start = time.time()

    # 训练
    for epoch in range(1, max_epochs + 1):
        model.train(); train_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(Xb)               # (B, num_classes) —— 已适配
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

    # 测试
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

    # 日志
    log_root = f"comp_logs/MambaHSI/{dataset_name}/seed_{run_seed}"
    os.makedirs(log_root, exist_ok=True)
    with open(os.path.join(log_root, "loss_curve.json"), "w") as f:
        json.dump({
            "train_loss": [round(float(l), 4) for l in train_losses],
            "val_loss": [round(float(l), 4) for l in val_losses]
        }, f, indent=2)

    evaluate_and_log_metrics_2d(
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
    # ====== 数据集配置 ======
    dataset_name = "chikusei"

    pca_components = 30
    patch_size = 7
    repeats = 2

    accs = []
    for i in range(repeats):
        seed = i * 10 + 42
        print(f"\n🔁 Running trial {i+1} with seed {seed}")
        acc = train_and_evaluate_mambahsi(
            run_seed=seed,
            dataset_name=dataset_name,
            pca_components=pca_components,
            patch_size=patch_size,
            batch_size=64,            # 可调
            max_epochs=200,
            patience=10,
            lr=1e-3,
            mamba_type='both'         # 'spa' / 'spe' / 'both'
        )
        accs.append(acc)

    print("\n📊 Summary of Repeated Training:")
    for i, acc in enumerate(accs):
        print(f"Run {i+1}: {acc:.2f}%")
    print(f"\nAverage Accuracy: {np.mean(accs):.2f}%, Std Dev: {np.std(accs):.2f}%")
