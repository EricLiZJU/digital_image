# comp/SGMAE/Botswana/model_Botswana.py
import os
import time
import json
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

# ===================== 取自你贴的 SGMAE 关键部件：Stem / DL =====================
from einops import rearrange

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1,
                 groups=1, bias=True, dropout=0, norm=nn.BatchNorm2d, act_func=nn.ReLU):
        super().__init__()
        self.dropout = nn.Dropout2d(dropout, inplace=False) if dropout > 0 else None
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.norm = norm(num_features=out_channels) if norm else None
        self.act = act_func() if act_func else None

    def forward(self, x):
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        if self.norm: x = self.norm(x)
        if self.act:  x = self.act(x)
        return x

class Stem(nn.Module):
    def __init__(self, in_chans=3, embed_dim=96):
        super().__init__()
        self.conv1 = ConvLayer(in_chans, embed_dim // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Sequential(
            ConvLayer(embed_dim // 2, embed_dim // 2, kernel_size=3, stride=1, padding=1, bias=False),
            ConvLayer(embed_dim // 2, embed_dim // 2, kernel_size=3, stride=1, padding=1, bias=False, act_func=None)
        )
        self.conv3 = nn.Sequential(
            ConvLayer(embed_dim // 2, embed_dim * 4, kernel_size=3, stride=1, padding=1, bias=False),
            ConvLayer(embed_dim * 4, embed_dim, kernel_size=1, bias=False, act_func=None)
        )

    def forward(self, x):              # x: (B, C, P, P)
        x = self.conv1(x)              # (B, D/2, P, P)
        x = self.conv2(x) + x          # 残差
        x = self.conv3(x)              # (B, D, P, P)
        x = rearrange(x, 'b d h w -> b h w d')  # (B, P, P, D)
        return x

class DL(nn.Module):
    def __init__(self, in_chans=30, num_classes=23, depths=[1], dims=64,
                 drop_rate=0.0, norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.embed_dim = dims[0]
        self.num_features = dims[-1]

        self.patch_embed = Stem(in_chans=in_chans, embed_dim=self.embed_dim)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):     # x: (B, C, P, P)
        x = self.patch_embed(x)        # (B, P, P, D)
        x = self.pos_drop(x)
        x = torch.flatten(x, 1, 2)     # (B, P*P, D)
        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2))  # (B, D, 1)
        x = torch.flatten(x, 1)        # (B, D)
        return x

    def forward(self, x):              # 这里不做 squeeze，直接支持 (B,C,P,P)
        x = self.forward_features(x)
        x = self.head(x)
        return x


# ===================== 通用工具（与你现有 2D 管线一致） =====================
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
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
    if np.all(label_map == ignored_label):
        raise ValueError("All labels are ignored; no samples.")

    padded_img = np.pad(img_cube, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    padded_label = np.pad(label_map, ((pad, pad), (pad, pad)), mode='constant', constant_values=ignored_label)

    patches, labels = [], []
    for i in range(pad, H + pad):
        for j in range(pad, W + pad):
            lab = padded_label[i, j]
            if lab == ignored_label: continue
            patch = padded_img[i-pad:i+pad+1, j-pad:j+pad+1, :]
            patch = np.transpose(patch, (2, 0, 1))  # (C,H,W)
            patches.append(patch); labels.append(int(lab - 1))
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
            "seed": run_seed, "dataset": dataset_name,
            "overall_accuracy": round(float(acc), 4),
            "average_accuracy": round(aa, 4),
            "kappa": round(kappa, 4),
            "per_class_accuracy": [round(float(x), 4) for x in per_class_acc.tolist()]
        }, f, indent=2)

    try:
        flops, params = get_model_complexity_info(
            model, (in_channels, patch_size, patch_size),
            as_strings=False, print_per_layer_stat=False
        )
        flops_info = {"FLOPs(M)": round(flops/1e6, 2), "Params(K)": round(params/1e3, 2)}
    except Exception:
        total_params = sum(p.numel() for p in model.parameters())
        flops_info = {"FLOPs(M)": None, "Params(K)": round(total_params/1e3, 2)}

    with open(os.path.join(log_root, "model_profile.json"), "w") as f:
        json.dump(flops_info, f, indent=2)

    with open(os.path.join(log_root, "time_log.json"), "w") as f:
        json.dump({
            "train_time(s)": round(train_time, 2) if train_time else None,
            "val_time(s)": round(val_time, 2) if val_time else None
        }, f, indent=2)

    np.savetxt(os.path.join(log_root, "confusion_matrix.csv"), conf_mat, fmt="%d", delimiter=",")


# ===================== SGMAE 分类包装器 =====================
class SGMAEClassifier(nn.Module):
    """
    用 SGMAE 的 DL(Stem) 做 patch encoder；默认仅 CNN 端。
    可选：在全部样本特征上加一个图头（kNN 图 + 小型 GCN），见 train_and_evaluate_sgmae(use_graph=True)。
    """
    def __init__(self, in_channels, num_classes, patch_size=7,
                 dl_dim=64, drop_rate=0.0):
        super().__init__()
        # 直接用 DL，但不使用其 head；我们自定义线性分类头
        self.backbone = DL(in_chans=in_channels, num_classes=0, dims=dl_dim, drop_rate=drop_rate)
        # DL 的输出维度 = 最后一层 dims（即 dl_dim * 2**(num_layers-1)，这里 num_layers=1 → dl_dim）
        self.feat_dim = int(dl_dim)  # 与 DL.forward_features 输出一致
        self.head = nn.Linear(self.feat_dim, num_classes)

    def forward(self, x):
        feat = self.backbone.forward_features(x)  # (B, D)
        logits = self.head(feat)                  # (B, num_classes)
        return logits

    @torch.no_grad()
    def extract_features(self, x):
        return self.backbone.forward_features(x)  # (B, D)


# -----------------（可选）图头：kNN 图 + 小型 GCN（transductive）-----------------
class SmallGCN(nn.Module):
    """非常小的 2~3 层 GCN：输入=DL 特征；输出=类别 logits。需要 torch_geometric."""
    def __init__(self, in_dim, num_classes, hidden=128, layers=2, dropout=0.2):
        super().__init__()
        try:
            from torch_geometric.nn import GCNConv
        except Exception as e:
            raise ImportError("use_graph=True 需要安装 torch_geometric。") from e

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden, cached=True, add_self_loops=False))
        for _ in range(layers-2):
            self.convs.append(GCNConv(hidden, hidden, cached=True, add_self_loops=False))
        self.convs.append(GCNConv(hidden, num_classes, cached=True, add_self_loops=False))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


# ===================== 训练与评估：默认仅 CNN（与 HSIMAE 管线完全一致） =====================
def train_and_evaluate_sgmae(run_seed=42,
                             dataset_name="Botswana",
                             data_path='../../root/data/HSI_dataset/Pavia_university/PaviaU.mat',
                             label_path='/root/data/HSI_dataset/Pavia_university/PaviaU_gt.mat',
                             pca_components=30,
                             patch_size=7,
                             batch_size=128,
                             max_epochs=200,
                             patience=10,
                             lr=1e-3,
                             dl_dim=64,
                             drop_rate=0.0,
                             use_graph=False,         # 打开则会在 DL 特征之上训练一个小型全图 GCN
                             gcn_hidden=128,
                             gcn_layers=2,
                             gcn_knn=10,
                             gcn_steps=200,
                             gcn_lr=3e-3):
    set_seed(run_seed)

    # ---- 数据加载 ----
    data = scio.loadmat(data_path)['paviaU']
    label = scio.loadmat(label_path)['paviaU_gt'].flatten()
    h, w, bands = data.shape
    label_map = label.reshape(h, w)

    # ---- PCA ----
    C = min(pca_components, bands)
    img_2d = data.reshape(-1, bands)
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

    # ---- 模型/优化器/损失（CNN）----
    model = SGMAEClassifier(
        in_channels=in_channels,
        num_classes=num_classes,
        patch_size=patch_size,
        dl_dim=dl_dim,
        drop_rate=drop_rate
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf'); patience_counter = 0
    best_model_weights = None
    train_losses, val_losses = [], []
    train_start = time.time()

    # ---- 训练（CNN）----
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

    # ---- （可选）图头：基于全部样本的 DL 特征 + 小型 GCN ----
    gcn_logits_all = None
    if use_graph:
        try:
            from torch_geometric.utils import dense_to_sparse
        except Exception as e:
            print("⚠️ use_graph=True 但未安装 torch_geometric；将只使用 CNN 结果。")
            use_graph = False

    if use_graph:
        print("🧩 构建全图 GCN 头（基于 DL 特征）...")
        model.eval()
        # 取全部样本（train/val/test）DL 特征
        with torch.no_grad():
            def get_feats(X):
                feats = []
                for i in range(0, len(X), batch_size):
                    Xb = torch.from_numpy(X[i:i+batch_size]).to(device)
                    feats.append(model.extract_features(Xb).cpu())
                return torch.cat(feats, dim=0)
            feats_train = get_feats(X_train_full)
            feats_val   = get_feats(X_val)
            feats_test  = get_feats(X_test)

        # 拼接为全图节点特征
        X_all = torch.cat([feats_train, feats_val, feats_test], dim=0)    # (N,D)
        y_all = torch.from_numpy(np.concatenate([y_train_full, y_val, y_test], axis=0))  # (N,)
        n_train, n_val, n_test = len(y_train_full), len(y_val), len(y_test)
        idx_train = torch.arange(0, n_train)
        idx_val   = torch.arange(n_train, n_train+n_val)
        idx_test  = torch.arange(n_train+n_val, n_train+n_val+n_test)

        # kNN 相似度图（cosine），对称化 + 去自环
        Xn = F.normalize(X_all, dim=1)
        sim = Xn @ Xn.t()                                 # (N,N)
        k = min(gcn_knn, sim.size(0)-1)
        topk_vals, topk_idx = torch.topk(sim, k=k+1, dim=1)   # 包含自己
        N = sim.size(0)
        adj = torch.zeros_like(sim)
        rows = torch.arange(N).unsqueeze(1).repeat(1, k+1)
        adj[rows, topk_idx] = 1.0
        adj = torch.maximum(adj, adj.t())
        adj.fill_diagonal_(0.0)
        edge_index, _ = dense_to_sparse(adj)
        # 小型 GCN
        gcn = SmallGCN(in_dim=X_all.size(1), num_classes=num_classes,
                       hidden=gcn_hidden, layers=gcn_layers, dropout=0.2).to(device)
        gcn_opt = torch.optim.Adam(gcn.parameters(), lr=gcn_lr, weight_decay=1e-4)
        ce = nn.CrossEntropyLoss()

        X_all = X_all.to(device)
        y_all = y_all.to(device)
        edge_index = edge_index.to(device)

        best_val, best_state = 1e9, None
        for step in range(1, gcn_steps+1):
            gcn.train()
            gcn_opt.zero_grad()
            out_all = gcn(X_all, edge_index)      # (N,C)
            loss = ce(out_all[idx_train], y_all[idx_train])
            loss.backward(); gcn_opt.step()

            if step % 10 == 0 or step == gcn_steps:
                gcn.eval()
                with torch.no_grad():
                    logits = gcn(X_all, edge_index)
                    val_loss = ce(logits[idx_val], y_all[idx_val]).item()
                if val_loss < best_val:
                    best_val, best_state = val_loss, gcn.state_dict()

        if best_state is not None:
            gcn.load_state_dict(best_state)
        gcn.eval()
        with torch.no_grad():
            gcn_logits_all = gcn(X_all, edge_index).cpu()

    # ---- 测试（CNN 或 融合）----
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
    model.eval(); all_true, all_pred = [], []
    val_start = time.time()
    with torch.no_grad():
        if use_graph and gcn_logits_all is not None:
            # 直接用图头在 test 索引上的预测（也可以与 CNN 平均融合，按需改）
            test_logits = gcn_logits_all[-len(y_test):]
            preds = test_logits.argmax(dim=1).numpy()
            all_pred.extend(preds.tolist())
            all_true.extend(y_test.tolist())
        else:
            for Xb, yb in test_loader:
                Xb = Xb.to(device)
                preds = model(Xb).argmax(dim=1)
                all_true.extend(yb.numpy()); all_pred.extend(preds.cpu().numpy())

    acc = np.mean(np.array(all_true) == np.array(all_pred)) * 100.0
    val_time = time.time() - val_start
    print(f"✅ Test Accuracy: {acc:.2f}%")

    # ---- 日志 ----
    log_root = f"comp_logs/SGMAE/{dataset_name}/seed_{run_seed}"
    os.makedirs(log_root, exist_ok=True)
    with open(os.path.join(log_root, "loss_curve.json"), "w") as f:
        json.dump({
            "train_loss": [round(float(l), 4) for l in train_losses],
            "val_loss":   [round(float(l), 4) for l in val_losses]
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
    dataset_name = "PaviaU"

    pca_components = 30
    patch_size = 7
    repeats = 1

    # （可选）图头开关与参数
    use_graph = False
    gcn_hidden = 128
    gcn_layers = 2
    gcn_knn = 10
    gcn_steps = 200
    gcn_lr = 3e-3

    accs = []
    for i in range(repeats):
        seed = i * 10 + 42
        print(f"\n🔁 Running trial {i+1} with seed {seed}")
        acc = train_and_evaluate_sgmae(
            run_seed=seed,
            dataset_name=dataset_name,
            pca_components=pca_components,
            patch_size=patch_size,
            batch_size=128,
            max_epochs=200,
            patience=10,
            lr=1e-3,
            dl_dim=64,
            drop_rate=0.0,
            use_graph=use_graph,
            gcn_hidden=gcn_hidden,
            gcn_layers=gcn_layers,
            gcn_knn=gcn_knn,
            gcn_steps=gcn_steps,
            gcn_lr=gcn_lr
        )
        accs.append(acc)

    print("\n📊 Summary of Repeated Training:")
    for i, acc in enumerate(accs):
        print(f"Run {i+1}: {acc:.2f}%")
    print(f"\nAverage Accuracy: {np.mean(accs):.2f}%, Std Dev: {np.std(accs):.2f}%")