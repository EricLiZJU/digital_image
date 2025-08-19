# comp/MSA-GCN/Botswana/model_Botswana.py
import os
import time
import json
import math
import random
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import torch
import torch.nn as nn
import torch.nn.functional as F
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
        raise ValueError(f"label_map shape {label_map.shape} != image spatial shape {(H, W)}")
    if C < 1:
        raise ValueError("img_cube must have at least 1 channel")
    if np.all(label_map == ignored_label):
        raise ValueError("All labels equal to ignored_label; no samples to extract.")

    padded_img = np.pad(img_cube, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    padded_label = np.pad(label_map, ((pad, pad)), mode='constant', constant_values=ignored_label)

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

    # FLOPs/Params —— 用 (C, P, P) 做 dummy 输入
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


# ===================== MSA-GCN（单模态HSI版，兼容你的训练脚手架） =====================
def gaussian_graph(x: torch.Tensor, tau: float = 1.0, add_self: bool = True):
    """
    x: (B, N, D) 节点特征；N=P*P
    返回 A: (B, N, N) 的归一化相似度图（按节点维 softmax），可近似你给的 dist_mask。
    """
    # 计算 pairwise 欧氏距离的负值（越近越大）
    # 使用 (x_i - x_j)^2 = x2 + x2^T - 2 x x^T
    x2 = (x ** 2).sum(dim=-1, keepdim=True)            # (B,N,1)
    # (x_i - x_j)^2 = x2_i + x2_j - 2<x_i,x_j>
    dist2 = x2 + x2.transpose(1, 2) - 2.0 * x @ x.transpose(1, 2)  # (B,N,N)
    sim = torch.exp(-dist2 / (tau * x.size(-1)))        # 高斯核
    if add_self:
        sim = sim + torch.eye(sim.size(1), device=x.device).unsqueeze(0)
    # 行归一化
    sim = sim / (sim.sum(dim=-1, keepdim=True) + 1e-6)
    return sim


class Attention(nn.Module):
    """
    Multi-head Self-Attention（和你贴的实现一致思路）
    输入 (B, N, D) -> 输出 (B, N, D)
    """
    def __init__(self, dim, num_heads=4, head_dim=None, p_drop=0.1):
        super().__init__()
        if head_dim is None:
            assert dim % num_heads == 0, "dim must be divisible by num_heads"
            head_dim = dim // num_heads
        inner = head_dim * num_heads
        self.num_heads = num_heads
        self.scale = head_dim ** -0.5
        self.to_qkv = nn.Linear(dim, inner * 3, bias=True)
        self.proj = nn.Sequential(nn.Linear(inner, dim), nn.Dropout(p_drop))

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # 3*(B,N,inner)
        def reshape(t):
            return t.view(B, N, self.num_heads, -1).permute(0, 2, 1, 3)  # (B,h,N,dh)
        q, k, v = map(reshape, qkv)
        attn = (q @ k.transpose(-2, -1)) * self.scale          # (B,h,N,N)
        attn = attn.softmax(dim=-1)
        out = attn @ v                                         # (B,h,N,dh)
        out = out.permute(0, 2, 1, 3).reshape(B, N, -1)        # (B,N,h*dh)
        out = self.proj(out)                                   # (B,N,D)
        return out


class GraphConvolution(nn.Module):
    """
    带 attention 的 GCN 层：Y = softmax(A) X W，经 attention 加权再残差
    input:  X (B,N,Fin)
            A (B,N,N)
    output: (B,N,Fout)
    """
    def __init__(self, in_features, out_features, bias=True, att_heads=3, att_head_dim=None, p_drop=0.1):
        super().__init__()
        self.lin = nn.Linear(in_features, out_features, bias=bias)
        self.att = Attention(out_features, num_heads=att_heads, head_dim=att_head_dim, p_drop=p_drop)
        self.act = nn.ReLU(inplace=True)
        self.bn  = nn.BatchNorm1d(out_features)

    def forward(self, x, adj):
        # x: (B,N,Fin) -> (B,N,Fout)
        support = self.lin(x)                           # XW
        out = adj @ support                             # A XW
        # attention reweight
        attw = self.att(out)                            # (B,N,Fout)
        out = out * torch.sigmoid(attw)                 # gate
        # BN按(N作为“序列”，把(N*B, F)送入1dBN)
        B, N, F = out.shape
        out = self.bn(out.view(B * N, F)).view(B, N, F)
        out = self.act(out)
        return out


class MSAGCNClassifier(nn.Module):
    """
    MSA-GCN（单模态 HSI 版）：
    - 先用 1x1 Conv 把 (C,P,P) 投影到 d_model，作为每个像素节点的特征
    - 构图 A (高斯核/softmax 归一化)
    - 堆叠 2~3 层 带注意力的 GCN
    - 全局池化 (mean over N)
    - 线性分类为 num_classes
    """
    def __init__(self, in_channels, num_classes, patch_size=7,
                 d_model=64, gcn_layers=2, tau=1.0,
                 att_heads=4, p_drop=0.1):
        super().__init__()
        self.patch = patch_size
        self.d_model = d_model
        self.tau = tau

        # 光谱到节点特征：1×1 卷积 + 深度可分卷积（增强局部）
        self.feat = nn.Sequential(
            nn.Conv2d(in_channels, d_model, kernel_size=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, groups=d_model, bias=False),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True),
        )

        # 多层 GCN
        layers = []
        for i in range(gcn_layers):
            fin = d_model
            fout = d_model
            layers.append(GraphConvolution(fin, fout, att_heads=att_heads, p_drop=p_drop))
        self.gcn = nn.ModuleList(layers)

        self.dropout = nn.Dropout(p_drop)
        self.cls = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (B,C,P,P)
        B, C, P, P2 = x.shape
        assert P == self.patch and P == P2, f"expect square patch {self.patch}, got {x.shape}"

        f = self.feat(x)                      # (B,d,P,P)
        f = f.permute(0, 2, 3, 1).contiguous().view(B, P * P, self.d_model)  # (B,N,D), N=P*P

        # 构图（每个样本一张图）
        with torch.no_grad():
            A = gaussian_graph(f.detach(), tau=self.tau, add_self=True)      # (B,N,N)

        # 堆叠 GCN
        h = f
        for layer in self.gcn:
            h = layer(h, A)

        # 池化 & 分类
        h = h.mean(dim=1)               # (B,D)
        h = self.dropout(h)
        logits = self.cls(h)            # (B,num_classes)
        return logits


# ===================== 训练与评估（与2D脚本一致） =====================
def train_and_evaluate_msagcn(run_seed=42,
                              dataset_name="Botswana",
                              data_path='../../root/data/HSI_dataset/Salinas/Salinas_corrected.mat',
                              label_path='../../root/data/HSI_dataset/Salinas/Salinas_gt.mat',
                              pca_components=30,
                              patch_size=7,
                              batch_size=128,
                              max_epochs=200,
                              patience=10,
                              lr=1e-3,
                              d_model=64,
                              gcn_layers=2,
                              tau=1.0,
                              att_heads=4,
                              p_drop=0.1):
    set_seed(run_seed)

    # ---- 数据加载 ----
    data = scio.loadmat(data_path)['salinas_corrected']
    label = scio.loadmat(label_path)['salinas_gt'].flatten()
    h, w, bands = data.shape
    label_map = label.reshape(h, w)

    # ---- PCA ----
    C = min(pca_components, bands)
    data_pca = PCA(n_components=C).fit_transform(data.reshape(-1, bands)).reshape(h, w, C)

    # ---- 生成 2D Patch ----
    patches, patch_labels = extract_2d_patches(data_pca, label_map, patch_size=patch_size, ignored_label=0)
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
    model = MSAGCNClassifier(
        in_channels=in_channels,
        num_classes=num_classes,
        patch_size=patch_size,
        d_model=d_model,
        gcn_layers=gcn_layers,
        tau=tau,
        att_heads=att_heads,
        p_drop=p_drop
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
    log_root = f"comp_logs/MSA-GCN/{dataset_name}/seed_{run_seed}"
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
    # ====== 数据集配置 ======
    dataset_name = "Salina"

    pca_components = 30
    patch_size = 7
    repeats = 3

    # 模型超参（可按需微调）
    d_model = 64
    gcn_layers = 2
    tau = 1.0
    att_heads = 4
    p_drop = 0.1

    accs = []
    for i in range(repeats):
        seed = i * 10 + 42
        print(f"\n🔁 Running trial {i+1} with seed {seed}")
        acc = train_and_evaluate_msagcn(
            run_seed=seed,
            dataset_name=dataset_name,
            pca_components=pca_components,
            patch_size=patch_size,
            batch_size=128,
            max_epochs=200,
            patience=10,
            lr=1e-3,
            d_model=d_model,
            gcn_layers=gcn_layers,
            tau=tau,
            att_heads=att_heads,
            p_drop=p_drop
        )
        accs.append(acc)

    print("\n📊 Summary of Repeated Training:")
    for i, acc in enumerate(accs):
        print(f"Run {i+1}: {acc:.2f}%")
    print(f"\nAverage Accuracy: {np.mean(accs):.2f}%, Std Dev: {np.std(accs):.2f}%")