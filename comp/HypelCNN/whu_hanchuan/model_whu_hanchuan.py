# train_hypelcnn_with_logging.py
import os, time, json, math, random
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import h5py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from scipy.io import loadmat

# ptflops 计算复杂度（与你现有脚本一致）
from ptflops import get_model_complexity_info


# ===================== 通用工具 =====================
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def extract_2d_patches(img_cube, label_map, patch_size=7, ignored_label=0):
    assert patch_size % 2 == 1, "patch_size must be odd."
    H, W, C = img_cube.shape
    pad = patch_size // 2
    if label_map.shape != (H, W):
        raise ValueError(f"label_map shape {label_map.shape} != {(H, W)}")
    if np.all(label_map == ignored_label):
        raise ValueError("All labels equal to ignored_label.")

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
    patches = np.asarray(patches, dtype='float32')
    labels = np.asarray(labels, dtype='int64')
    if len(labels) == 0:
        raise ValueError("No labeled patches extracted.")
    return patches, labels


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

    # FLOPs / Params （与2D脚本一致用 ptflops）
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


# ===================== PyTorch 版 HypelCNN =====================
class ScaleToOut(nn.Module):
    """通道对齐的 1x1 卷积（仅当 in/out 通道不一致时启用）。空间尺寸不变时用于残差对齐。"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.need = in_ch != out_ch
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False) if self.need else nn.Identity()
    def forward(self, x):
        return self.proj(x) if self.need else x


def conv_bn_act(in_ch, out_ch, k=1, stride=1, padding=None, bn_momentum=0.01, act=None, bias=False):
    if padding is None: padding = k // 2
    layers = [nn.Conv2d(in_ch, out_ch, k, stride=stride, padding=padding, bias=bias),
              nn.BatchNorm2d(out_ch, momentum=bn_momentum)]
    if act is not None: layers.append(act)
    return nn.Sequential(*layers)


class HypelCNN(nn.Module):
    """
    参考原 TF 版 HYPELCNNModel：
      - 谱向编码/解码：若干层 1x1 conv（带 BN + LeakyReLU），可残差
      - 空间层级块：对 k=1..patch_size 的奇数 k 做 k×k 卷积拼接，然后 1×1 融合，可残差
      - 全连接降维块：每次除以 degradation_coeff 直到接近类别数，层间 Dropout
    """
    def __init__(self,
                 in_channels=30,
                 num_classes=16,
                 filter_count=1200,
                 spectral_hierarchy_level=3,
                 spatial_hierarchy_level=3,
                 lrelu_alpha=0.18,
                 bn_decay=0.99,
                 dropout_ratio=0.4,
                 degradation_coeff=3,
                 use_residual=True,
                 patch_size=7):
        super().__init__()
        self.num_classes = num_classes
        self.use_residual = use_residual
        self.degradation_coeff = degradation_coeff
        bn_momentum = 1.0 - bn_decay
        act = nn.LeakyReLU(lrelu_alpha, inplace=True)
        self.act = act
        self.patch_size = patch_size

        # 输入通道先扩展到较高维度（对标 TF 的 level_filter_count）
        self.stem = conv_bn_act(in_channels, filter_count, k=1, bn_momentum=bn_momentum, act=act)

        # 谱向编码阶段（encoding）
        self.spec_enc = nn.ModuleList()
        in_ch = filter_count
        for i in range(spectral_hierarchy_level):
            out_ch = max(filter_count // (2 ** ((spectral_hierarchy_level - 1) - i)), 8)
            block = conv_bn_act(in_ch, out_ch, k=1, bn_momentum=bn_momentum, act=act)
            proj = ScaleToOut(in_ch, out_ch)
            self.spec_enc.append(nn.ModuleDict({"conv": block, "proj": proj}))
            in_ch = out_ch

        # 谱向解码阶段（decoding）
        self.spec_dec = nn.ModuleList()
        for i in range(spectral_hierarchy_level):
            out_ch = max(filter_count // (2 ** i), 8)
            block = conv_bn_act(in_ch, out_ch, k=1, bn_momentum=bn_momentum, act=act)
            proj = ScaleToOut(in_ch, out_ch)
            self.spec_dec.append(nn.ModuleDict({"conv": block, "proj": proj}))
            in_ch = out_ch

        # 空间层级块
        self.spatial_blocks = nn.ModuleList()
        cur_ch = in_ch
        for lvl in range(spatial_hierarchy_level):
            # 本层的每个分支通道数
            branch_ch = max(cur_ch // (2 ** lvl), 8)
            # odd k: 1,3,5,...,<=patch_size
            ks = list(range(1, patch_size + 1, 2))
            branches = nn.ModuleList([conv_bn_act(cur_ch, branch_ch, k=k, bn_momentum=bn_momentum, act=act)
                                      for k in ks])
            fuse = conv_bn_act(branch_ch * len(ks), cur_ch, k=1, bn_momentum=bn_momentum, act=None)
            proj = ScaleToOut(cur_ch, cur_ch)
            self.spatial_blocks.append(nn.ModuleDict({"branches": branches, "fuse": fuse, "proj": proj}))
            # 经过 1x1 融合后，通道仍维持 cur_ch

        # 分类头：Flatten -> FC 降维块（几何衰减）-> 最终分类
        # 这里全连接层的输入维度依赖于 spatial 输出的通道数与 patch_size
        self.dropout_ratio = dropout_ratio
        self.classifier = None  # 延迟初始化，根据前向时的展平维度构建
        self.final_fc = None

    # 替换原来的 _build_fc
    def _build_fc(self, flatten_dim):
        layers = []
        elem = flatten_dim
        if self.degradation_coeff < 2:
            self.degradation_coeff = 2
        fc_stage_count = max(1, int(math.floor(math.log(max(elem / self.num_classes, 1.0), self.degradation_coeff))))
        for _ in range(fc_stage_count - 1):
            elem = max(self.num_classes, elem // self.degradation_coeff)
            layers += [nn.Linear(flatten_dim, elem), nn.LeakyReLU(0.18, inplace=True), nn.Dropout(self.dropout_ratio)]
            flatten_dim = elem
        self.classifier = nn.Sequential(*layers) if layers else nn.Identity()
        self.final_fc = nn.Linear(flatten_dim, self.num_classes)

    # 替换原来的 forward（关键是把新建层迁移到 out.device）
    def forward(self, x):
        # x: (B, C, H, W)
        out = self.stem(x)

        for m in self.spec_enc:
            y = m["conv"](out)
            out = y + m["proj"](out) if self.use_residual else y
            out = F.leaky_relu(out, negative_slope=0.18, inplace=True)

        for m in self.spec_dec:
            y = m["conv"](out)
            out = y + m["proj"](out) if self.use_residual else y
            out = F.leaky_relu(out, negative_slope=0.18, inplace=True)

        for m in self.spatial_blocks:
            feats = [branch(out) for branch in m["branches"]]
            y = torch.cat(feats, dim=1)
            y = m["fuse"](y)
            out = y + m["proj"](out) if self.use_residual else y

        B = out.size(0)
        flat = out.view(B, -1)

        # 懒构建 + 立刻迁移到与输入同设备
        if self.classifier is None or self.final_fc is None:
            self._build_fc(flat.size(1))
            dev = out.device
            self.classifier = self.classifier.to(dev)
            self.final_fc = self.final_fc.to(dev)

        z = self.classifier(flat)
        logits = self.final_fc(z)
        return logits


# ===================== 训练与评测流程（与你2D脚本一致） =====================
def train_and_evaluate_hypelcnn(run_seed=42,
                                dataset_name="Botswana",
                                data_path='../../root/data/HSI_dataset/Matlab_data_format/Matlab_data_format/WHU-Hi-HanChuan/WHU_Hi_HanChuan.mat',
                                label_path='../../root/data/HSI_dataset/Matlab_data_format/Matlab_data_format/WHU-Hi-HanChuan/WHU_Hi_HanChuan_gt.mat',
                                pca_components=30,
                                patch_size=7,
                                batch_size=128,
                                max_epochs=200,
                                patience=10,
                                lr=1e-3,
                                filter_count=1200,
                                spectral_hierarchy_level=3,
                                spatial_hierarchy_level=3,
                                lrelu_alpha=0.18,
                                bn_decay=0.99,
                                dropout_ratio=0.4,
                                degradation_coeff=3,
                                use_residual=True):
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

    # ---- Patch 提取 ----
    patches, patch_labels = extract_2d_patches(
        data_cube, label_map, patch_size=patch_size, ignored_label=0
    )
    num_classes = int(patch_labels.max()) + 1
    in_channels = pca_components

    # ---- 划分 70/15/15 ----
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        patches, patch_labels, test_size=0.15, stratify=patch_labels, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.176, stratify=y_train_full, random_state=42
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
                              batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
                            batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
                             batch_size=batch_size, shuffle=False, pin_memory=True)

    # ---- 模型 & 优化器 ----
    model = HypelCNN(
        in_channels=in_channels,
        num_classes=num_classes,
        filter_count=filter_count,
        spectral_hierarchy_level=spectral_hierarchy_level,
        spatial_hierarchy_level=spatial_hierarchy_level,
        lrelu_alpha=lrelu_alpha,
        bn_decay=bn_decay,
        dropout_ratio=dropout_ratio,
        degradation_coeff=degradation_coeff,
        use_residual=use_residual,
        patch_size=patch_size
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf'); patience_counter = 0
    best_model_weights = None
    train_losses, val_losses = [], []

    # ---- 训练 ----
    train_start = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train(); train_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(Xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Val
        model.eval(); val_loss = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                val_loss += criterion(model(Xb), yb).item()

        print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        train_losses.append(train_loss); val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = model.state_dict()
            patience_counter = 0
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
            Xb, yb = Xb.to(device), yb.to(device)
            pred = model(Xb).argmax(dim=1)
            all_true.extend(yb.cpu().numpy()); all_pred.extend(pred.cpu().numpy())
    acc = np.mean(np.array(all_true) == np.array(all_pred)) * 100.0
    print(f"✅ Test Accuracy: {acc:.2f}%")
    val_time = time.time() - val_start

    # ---- 日志 ----
    log_root = f"comp_logs/HypelCNN/{dataset_name}/seed_{run_seed}"
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
        log_root=log_root,
        train_time=train_time,
        val_time=val_time
    )

    # ---- 分类图可视化 ----
    print("🖼️ Generating classification maps...")
    pred_map = np.zeros((H, W), dtype=int)
    gt_map = label_map.copy(); mask = (gt_map != 0)

    for i in range(H):
        for j in range(W):
            if mask[i, j]:
                patch = data_cube[i - (patch_size // 2): i + (patch_size // 2) + 1,
                                  j - (patch_size // 2): j + (patch_size // 2) + 1, :]
                if patch.shape != (patch_size, patch_size, in_channels):
                    continue
                patch = np.transpose(patch, (2, 0, 1))
                patch_tensor = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    pred_label = model(patch_tensor).argmax(dim=1).item()
                pred_map[i, j] = pred_label + 1

    num_classes_viz = int(patch_labels.max()) + 1
    cmap = mcolors.ListedColormap(plt.colormaps['tab20'].colors[:num_classes_viz])

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    axs[0].imshow(gt_map, cmap=cmap, vmin=1, vmax=num_classes_viz); axs[0].set_title("Ground Truth"); axs[0].axis('off')
    axs[1].imshow(pred_map, cmap=cmap, vmin=1, vmax=num_classes_viz); axs[1].set_title(f"Prediction (Acc: {acc:.2f}%)"); axs[1].axis('off')

    fig_path = os.path.join(log_root, f"{dataset_name}_HypelCNN_run{run_seed}_vis.png")
    fig_path_pdf = os.path.join(log_root, f"{dataset_name}_HypelCNN_run{run_seed}_vis.pdf")
    plt.savefig(fig_path, bbox_inches='tight', dpi=300)
    plt.savefig(fig_path_pdf, bbox_inches='tight')
    plt.close()
    print(f"✅ Classification map saved to:\n  {fig_path}\n  {fig_path_pdf}")

    return acc


if __name__ == "__main__":
    # ====== 根据你的数据集修改 ======
    dataset_name = "whu_hanchuan"

    # 公平对比：建议 pca_components=30 与 3D/DSFormer 对齐；patch_size=7 与其他一致
    pca_components = 30
    patch_size = 7
    repeats = 10

    accs = []
    for i in range(repeats):
        seed = i * 10 + 42
        print(f"\n🔁 Running trial {i+1} with seed {seed}")
        acc = train_and_evaluate_hypelcnn(
            run_seed=seed,
            dataset_name=dataset_name,
            pca_components=pca_components,
            patch_size=patch_size,
            batch_size=128,
            max_epochs=100,
            patience=10,
            lr=1e-3,
            # 下列是 HypelCNN 的可调超参（与原实现保持同名/同义）
            filter_count=1200,
            spectral_hierarchy_level=3,
            spatial_hierarchy_level=3,
            lrelu_alpha=0.18,
            bn_decay=0.99,
            dropout_ratio=0.4,
            degradation_coeff=3,
            use_residual=True
        )
        accs.append(acc)

    print("\n📊 Summary of Repeated Training:")
    for i, acc in enumerate(accs):
        print(f"Run {i+1}: {acc:.2f}%")
    print(f"\nAverage Accuracy: {np.mean(accs):.2f}%, Std Dev: {np.std(accs):.2f}%")
