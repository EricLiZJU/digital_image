# comp/HyLiOSR/Botswana/model_Botswana.py
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


# ===================== HyLiOSR (ResNet999) —— 适配为单模态 HSI 可训练分类器 =====================
class ResNet999(nn.Module):
    """
    根据你给的源码改成设备无关（不在定义时 .cuda()），并保持 forward 的返回结构。
    这里假定输入通道 = HSI_C + 2，其中最后 2 个通道是 LiDAR（可用 0 值占位）。
    """
    def __init__(self, data, band1, band2, dummynum, ncla1,
                 n_sub_prototypes=3, latent_dim=10,
                 temp_intra=1.0, temp_inter=0.1):
        super().__init__()
        self.data = data
        self.n_sub_prototypes = n_sub_prototypes
        self.latent_dim = latent_dim
        self.n_classes = ncla1
        self.temp_intra = temp_intra
        self.temp_inter = temp_inter

        # HSI 和 LiDAR 两个分支的第一层 3x3 valid 卷积
        if self.data == 3:
            self.conv0x = nn.Conv2d(band2 + 1, 32, kernel_size=3, padding=0)
            self.conv0  = nn.Conv2d(band1 - 1, 32, kernel_size=3, padding=0)
        else:
            self.conv0x = nn.Conv2d(band2, 32, kernel_size=3, padding=0)   # band2=2 (LiDAR)
            self.conv0  = nn.Conv2d(band1, 32, kernel_size=3, padding=0)   # band1=C (HSI)

        self.bn11 = nn.BatchNorm2d(64, eps=0.001, momentum=0.9)
        self.conv11 = nn.Conv2d(64, 64, kernel_size=3, padding='same')
        self.conv12 = nn.Conv2d(64, 64, kernel_size=3, padding='same')
        self.bn21 = nn.BatchNorm2d(64, eps=0.001, momentum=0.9)
        self.conv21 = nn.Conv2d(64, 64, kernel_size=3, padding='same')
        self.conv22 = nn.Conv2d(64, 64, kernel_size=3, padding='same')

        self.fc1 = nn.Linear(64, self.n_classes)  # 分类
        self.fc2 = nn.Linear(64, dummynum)        # dummy 分支

        self.fc_mu     = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

        # 原码里 prototypes 带 .cuda()，这里去掉以免设备问题
        self.prototypes = nn.Parameter(torch.randn(self.n_classes * self.n_sub_prototypes, self.latent_dim),
                                       requires_grad=True)

        # 反卷积重建支路
        self.dconv1 = nn.ConvTranspose2d(64, 64, kernel_size=1, padding=0)
        self.dconv2 = nn.ConvTranspose2d(64, 64, kernel_size=3, padding=0)
        self.dconv3 = nn.ConvTranspose2d(64, 64, kernel_size=3, padding=0)
        self.dconv4 = nn.ConvTranspose2d(64, 64, kernel_size=3, padding=0)

        if self.data == 3:
            self.dconvhsi   = nn.ConvTranspose2d(32, band1 - 1, kernel_size=3, padding=0)
            self.dconvlidar = nn.ConvTranspose2d(32, 2,         kernel_size=3, padding=0)
        else:
            self.dconvhsi   = nn.ConvTranspose2d(32, band1, kernel_size=3, padding=0)
            self.dconvlidar = nn.ConvTranspose2d(32, 1,     kernel_size=3, padding=0)

        self.bn1_de = nn.BatchNorm2d(64)
        self.bn2_de = nn.BatchNorm2d(64)

        nn.init.normal_(self.conv0x.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.conv0.weight,  mean=0.0, std=0.01)
        nn.init.normal_(self.conv11.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.conv12.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.fc1.weight,    mean=0.0, std=0.01)
        nn.init.normal_(self.fc_mu.weight,  mean=0.0, std=0.01)
        nn.init.normal_(self.fc_logvar.weight, mean=0.0, std=0.01)

    # 编码：前 3x3 valid 卷积（HSI 和 LiDAR）
    def encoder(self, x):
        x_hsi, x_lidar = torch.split(x, [x.shape[1]-2, 2], dim=1)  # 最后2通道视作 LiDAR
        h = self.conv0(x_hsi)        # -> (B,32,P-2,P-2)
        l = self.conv0x(x_lidar)     # -> (B,32,P-2,P-2)
        x = torch.cat([h, l], dim=1) # -> (B,64,P-2,P-2)
        x = F.relu(self.bn11(x))
        x = F.relu(self.conv11(x))
        x = self.conv12(x)
        x = torch.cat([h, l], dim=1) + x   # 残差（与源码对齐）
        return x

    def latter(self, x):
        x = F.adaptive_avg_pool2d(x, (1, 1))
        return torch.flatten(x, 1)  # (B,64)

    def distance(self, a, b):
        # pairwise Euclidean distance squared between rows of a and b
        a2 = (a**2).sum(dim=1, keepdim=True)           # (B,1)
        b2 = (b**2).sum(dim=1, keepdim=True).t()       # (1,M)
        prod = a @ b.t()                                # (B,M)
        return a2 + b2 - 2*prod                         # (B,M)

    def kl_div_to_prototypes(self, mean, logvar, prototypes):
        kl = self.distance(mean, prototypes)
        kl += torch.sum((logvar.exp() - logvar - 1), dim=1, keepdim=True)
        return 0.5 * kl

    def sampler(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps * std + mu
        return mu

    def fc_head(self, x_flat):
        pre  = self.fc1(x_flat)
        dummy= self.fc2(x_flat)
        mu   = self.fc_mu(x_flat)
        logv = self.fc_logvar(x_flat)
        return pre, dummy, mu, logv

    def restruct(self, x_flat):
        z = x_flat.reshape(-1, 64, 1, 1)
        z = F.relu(self.bn1_de(self.dconv1(z)))
        z = F.relu(self.dconv2(z))
        z = F.relu(self.bn2_de(self.dconv3(z)))
        z = F.relu(self.dconv4(z))
        HSI_x, LiDAR_x = torch.split(z, [32, 32], dim=1)
        HSI   = self.dconvhsi(HSI_x)
        LiDAR = self.dconvlidar(LiDAR_x)
        return HSI, LiDAR

    def forward(self, x):
        # 期望 x 的最后2通道为 LiDAR；若没有 LiDAR 调用方会补零
        x_enc = self.encoder(x)                     # (B,64,P-2,P-2)
        x_flat= self.latter(x_enc)                  # (B,64)
        pre, dpre, mu, logv = self.fc_head(x_flat)  # 分类 + VAE 参数
        z = self.sampler(mu, logv)
        dist = self.distance(z, self.prototypes)    # (B, n_classes * n_sub_prototypes)
        kld  = self.kl_div_to_prototypes(mu, logv, self.prototypes)  # (B, M)
        HSI_rec, LiDAR_rec = self.restruct(x_flat)
        return pre, HSI_rec, dpre, z, dist, kld, LiDAR_rec

    def mgpl_loss(self, outputs, x_in, y, n_classes, n_sub_prototypes=3,
                  temp_intra=1.0, temp_inter=0.1, w_rec=1.0, w_kld=0.1, w_ent=0.1, w_dis=0.1):
        """
        复现源码里的 MGPL 损失（做了轻微稳健化）。y 为类别索引 (B,)。
        x_in 拆成 HSI / LiDAR 的前后通道用于重建损失。
        """
        pre, x_hsi_rec, dummypre, latent_z, dist, kl_div, lidar_rec = outputs
        x_hsi, x_lidar = torch.split(x_in, [x_in.shape[1]-2, 2], dim=1)

        # 按类重组原型距离
        B = x_in.size(0)
        dist_reshape = dist.view(B, n_classes, n_sub_prototypes)          # (B,C,K)
        dist_class_min, _ = torch.min(dist_reshape, dim=2)                # (B,C)
        preds = torch.argmin(dist_class_min, dim=1)

        # 选真实类对应的 K 个原型
        y_onehot = F.one_hot(y, num_classes=n_classes).to(dist.dtype)     # (B,C)
        mask = y_onehot.repeat_interleave(n_sub_prototypes, dim=1).bool() # (B, C*K)
        dist_y = dist[mask].view(B, n_sub_prototypes)                     # (B,K)
        kld_y  = kl_div[mask].view(B, n_sub_prototypes)                   # (B,K)

        q_w_z_y = F.softmax(-dist_y / temp_intra, dim=1)                  # (B,K)

        # 重建损失（只对 HSI 支路，L1）
        rec_loss = F.l1_loss(x_hsi_rec, x_hsi)

        # KL + 熵
        kld_loss = torch.mean(torch.sum(q_w_z_y * kld_y, dim=1))
        ent_loss = torch.mean(torch.sum(q_w_z_y * torch.log(q_w_z_y * n_sub_prototypes + 1e-7), dim=1))

        # inter-class separation
        LSE_all    = torch.logsumexp(-dist / temp_inter, dim=1)           # (B,)
        LSE_target = torch.logsumexp(-dist_y / temp_inter, dim=1)         # (B,)
        dis_loss   = torch.mean(LSE_all - LSE_target)

        total = (w_rec * rec_loss) + (w_kld * kld_loss) + (w_ent * ent_loss) + (w_dis * dis_loss)
        return total, {'rec': rec_loss.item(), 'kld': kld_loss.item(),
                       'ent': ent_loss.item(), 'dis': dis_loss.item()}


class HyLiOSRClassifier(nn.Module):
    """
    包装为与你脚手架兼容的分类器：
    - 输入: (B,C,P,P) 仅 HSI；内部自动在通道后拼 2 个 0 通道作为 LiDAR 占位
    - 前向返回: logits (B,num_classes)
    - 训练时可选返回 MGPL 复合损失所需的中间量
    """
    def __init__(self, in_channels, num_classes, patch_size=7,
                 n_sub_prototypes=3, latent_dim=10,
                 temp_intra=1.0, temp_inter=0.1,
                 mgpl_on=False,  # 是否启用 MGPL 复合损失
                 mgpl_weights=(1.0, 0.1, 0.1, 0.1)  # (rec,kld,ent,dis) 权重
                 ):
        super().__init__()
        self.P = patch_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.mgpl_on = mgpl_on
        self.mgpl_w = mgpl_weights

        # band1 = HSI 通道数；band2 = LiDAR 通道数(固定 2)
        self.core = ResNet999(
            data=0, band1=in_channels, band2=2, dummynum=16, ncla1=num_classes,
            n_sub_prototypes=n_sub_prototypes, latent_dim=latent_dim,
            temp_intra=temp_intra, temp_inter=temp_inter
        )

    def forward(self, x, y=None):
        # x: (B,C,P,P) —— 若没有 LiDAR，这里自动拼 2 通道 0
        B, C, H, W = x.shape
        assert H == self.P and W == self.P, f"Patch size mismatch: expect {self.P}, got {(H,W)}"
        lidar_dummy = torch.zeros(B, 2, H, W, device=x.device, dtype=x.dtype)
        x_cat = torch.cat([x, lidar_dummy], dim=1)  # (B, C+2, P, P)

        outputs = self.core(x_cat)  # tuple
        logits = outputs[0]

        if self.training and self.mgpl_on and (y is not None):
            total_mgpl, parts = self.core.mgpl_loss(
                outputs, x_cat, y, n_classes=self.num_classes,
                n_sub_prototypes=self.core.n_sub_prototypes,
                temp_intra=self.core.temp_intra, temp_inter=self.core.temp_inter,
                w_rec=self.mgpl_w[0], w_kld=self.mgpl_w[1], w_ent=self.mgpl_w[2], w_dis=self.mgpl_w[3]
            )
            return logits, total_mgpl, parts
        return logits, None, None


# ===================== 训练与评估（与2D脚本一致） =====================
def train_and_evaluate_hyliosr(run_seed=42,
                               dataset_name="Botswana",
                               data_path='../../root/data/HSI_dataset/Pavia_university/PaviaU.mat',
                               label_path='/root/data/HSI_dataset/Pavia_university/PaviaU_gt.mat',
                               pca_components=30,
                               patch_size=7,
                               batch_size=128,
                               max_epochs=200,
                               patience=10,
                               lr=1e-3,
                               mgpl_on=False,             # 是否叠加 HyLiOSR 复合损失
                               mgpl_lambda=0.1,           # 复合损失整体权重（与CE相加）
                               mgpl_weights=(1.0,0.1,0.1,0.1),  # (rec,kld,ent,dis)
                               n_sub_prototypes=3,
                               latent_dim=10,
                               temp_intra=1.0,
                               temp_inter=0.1):
    set_seed(run_seed)

    # ---- 数据加载 ----
    data = scio.loadmat(data_path)['paviaU']
    label = scio.loadmat(label_path)['paviaU_gt'].flatten()
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
    model = HyLiOSRClassifier(
        in_channels=in_channels,
        num_classes=num_classes,
        patch_size=patch_size,
        n_sub_prototypes=n_sub_prototypes,
        latent_dim=latent_dim,
        temp_intra=temp_intra,
        temp_inter=temp_inter,
        mgpl_on=mgpl_on,
        mgpl_weights=mgpl_weights
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    ce_loss = nn.CrossEntropyLoss()

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
            logits, mgpl_loss_val, _ = model(Xb, yb)
            loss = ce_loss(logits, yb)
            if mgpl_on and (mgpl_loss_val is not None):
                loss = loss + mgpl_lambda * mgpl_loss_val
            loss.backward(); optimizer.step()
            train_loss += loss.item()

        model.eval(); val_loss = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                logits, _, _ = model(Xb)   # 验证不叠加 MGPL
                val_loss += ce_loss(logits, yb).item()

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
            logits, _, _ = model(Xb)
            preds = logits.argmax(dim=1)
            all_true.extend(yb.numpy()); all_pred.extend(preds.cpu().numpy())
    acc = np.mean(np.array(all_true) == np.array(all_pred)) * 100.0
    val_time = time.time() - val_start
    print(f"✅ Test Accuracy: {acc:.2f}%")

    # ---- 日志 ----
    log_root = f"comp_logs/HyLiOSR/{dataset_name}/seed_{run_seed}"
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
    repeats = 2

    # 是否叠加 HyLiOSR 复合损失（建议先关，只用 CE 跑通对比）
    mgpl_on = False
    mgpl_lambda = 0.1
    mgpl_weights = (1.0, 0.1, 0.1, 0.1)

    accs = []
    for i in range(repeats):
        seed = i * 10 + 42
        print(f"\n🔁 Running trial {i+1} with seed {seed}")
        acc = train_and_evaluate_hyliosr(
            run_seed=seed,
            dataset_name=dataset_name,
            pca_components=pca_components,
            patch_size=patch_size,
            batch_size=128,
            max_epochs=200,
            patience=10,
            lr=1e-3,
            mgpl_on=mgpl_on,
            mgpl_lambda=mgpl_lambda,
            mgpl_weights=mgpl_weights,
            n_sub_prototypes=3,
            latent_dim=10,
            temp_intra=1.0,
            temp_inter=0.1
        )
        accs.append(acc)

    print("\n📊 Summary of Repeated Training:")
    for i, acc in enumerate(accs):
        print(f"Run {i+1}: {acc:.2f}%")
    print(f"\nAverage Accuracy: {np.mean(accs):.2f}%, Std Dev: {np.std(accs):.2f}%")