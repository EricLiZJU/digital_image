# train_dsformer_with_logging.py
import os
import time
import json
import random
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

# ---- FLOPs 计算（优先 thop）----
try:
    from thop import profile as thop_profile
except Exception:
    thop_profile = None

from timm.models.layers import DropPath
from einops import rearrange
from einops.layers.torch import Reduce
import numbers
from itertools import repeat
import collections.abc


# ===================== 基础工具 =====================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def extract_2d_patches(img_cube, label_map, patch_size=7, ignored_label=0):
    """(H,W,C) -> (N,C,patch,patch) 仅保留中心像素有标签的patch; 标签从0开始"""
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


def evaluate_and_log_metrics_ds(y_true, y_pred, model, run_seed, dataset_name, acc,
                                example_input, log_root, train_time=None, val_time=None):
    os.makedirs(log_root, exist_ok=True)

    conf_mat = confusion_matrix(y_true, y_pred)
    with np.errstate(divide='ignore', invalid='ignore'):
        per_class_acc = conf_mat.diagonal() / conf_mat.sum(axis=1)
        per_class_acc = np.nan_to_num(per_class_acc, nan=0.0)
    aa = per_class_acc.mean()
    kappa = cohen_kappa_score(y_true, y_pred)

    metrics = {
        "seed": run_seed,
        "dataset": dataset_name,
        "overall_accuracy": round(float(acc), 4),
        "average_accuracy": round(float(aa), 4),
        "kappa": round(float(kappa), 4),
        "per_class_accuracy": [round(float(x), 4) for x in per_class_acc.tolist()]
    }
    with open(os.path.join(log_root, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # FLOPs / Params （thop）
    flops_info = {}
    try:
        if thop_profile is not None:
            model.eval()
            with torch.no_grad():
                flops, params = thop_profile(model, inputs=(example_input,))
            flops_info = {"FLOPs(M)": round(flops / 1e6, 2), "Params(K)": round(params / 1e3, 2)}
        else:
            total_params = sum(p.numel() for p in model.parameters())
            flops_info = {"FLOPs(M)": None, "Params(K)": round(total_params / 1e3, 2)}
    except Exception as e:
        total_params = sum(p.numel() for p in model.parameters())
        flops_info = {"FLOPs(M)": None, "Params(K)": round(total_params / 1e3, 2)}

    with open(os.path.join(log_root, "model_profile.json"), "w") as f:
        json.dump(flops_info, f, indent=2)

    time_info = {
        "train_time(s)": round(train_time, 2) if train_time else None,
        "val_time(s)": round(val_time, 2) if val_time else None
    }
    with open(os.path.join(log_root, "time_log.json"), "w") as f:
        json.dump(time_info, f, indent=2)

    np.savetxt(os.path.join(log_root, "confusion_matrix.csv"), conf_mat, fmt="%d", delimiter=",")


# ===================== DSFormer 模型 =====================
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse
to_2tuple = _ntuple(2)

class PatchEmbed(nn.Module):
    """Conv2d patchify: in_chans -> embed_dim, stride=patch_size"""
    def __init__(self, img_size=7, patch_size=1, in_chans=64, embed_dim=64, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        x = self.proj(x)
        Hp, Wp = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)  # (B, Ph*Pw, C)
        if self.norm is not None:
            x = self.norm(x)
        return x, (Hp, Wp)

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super().__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)
    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

def to_3d(x):  return rearrange(x, 'b c h w -> b (h w) c')
def to_4d(x,h,w): return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral): normalized_shape=(normalized_shape,)
        normalized_shape=torch.Size(normalized_shape); assert len(normalized_shape)==1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral): normalized_shape=(normalized_shape,)
        normalized_shape=torch.Size(normalized_shape); assert len(normalized_shape)==1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
    def forward(self, x):
        mu = x.mean(-1, keepdim=True); sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class Mlp(nn.Module):
    def __init__(self,in_features,hidden_features=None,out_features=None,act_layer=nn.GELU,drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def forward(self,x):
        x=self.fc1(x); x=self.act(x); x=self.drop(x); x=self.fc2(x); x=self.drop(x); return x

class Token_Selective_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias, k, group_num):
        super().__init__()
        self.num_heads = num_heads
        self.k = k
        self.group_num = group_num
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))
        self.qkv = nn.Conv3d(self.group_num, self.group_num * 3, kernel_size=(1, 1, 1), bias=False)
        self.qkv_conv = nn.Conv3d(self.group_num * 3, self.group_num * 3, kernel_size=(1, 3, 3),
                                  padding=(0, 1, 1), groups=self.group_num * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        # -> (b, group, c/group, h, w)
        x = x.reshape(b, self.group_num, c // self.group_num, h, w)
        q, k, v = self.qkv_conv(self.qkv(x)).chunk(3, dim=1)
        q = rearrange(q, 'b t (head c) h w -> b head c (h w t)', head=self.num_heads)
        k = rearrange(k, 'b t (head c) h w -> b head c (h w t)', head=self.num_heads)
        v = rearrange(v, 'b t (head c) h w -> b head c (h w t)', head=self.num_heads)
        q = F.normalize(q, dim=-1); k = F.normalize(k, dim=-1)
        N = q.shape[-1]
        attn = (q.transpose(-2, -1) @ k) * self.temperature  # (b, head, N, N)
        # top-k token选择
        index = torch.topk(attn, k=int(N * self.k), dim=-1, largest=True)[1]
        mask = torch.zeros_like(attn).scatter_(-1, index, 1.0)
        attn = torch.where(mask > 0, attn, torch.full_like(attn, float('-inf')))
        attn = attn.softmax(dim=-1)
        out = (attn @ v.transpose(-2, -1)).transpose(-2, -1)
        out = rearrange(out, 'b head c (h w t) -> b t (head c) h w', head=self.num_heads, h=h, w=w)
        out = out.reshape(b, -1, h, w)
        out = self.project_out(out)
        return out

class block(nn.Module):
    def __init__(self, dim, r=16, L=32):
        super().__init__()
        d = max(dim // r, L)
        self.conv0 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 5, stride=1, padding=4, groups=dim, dilation=2)
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv = nn.Conv2d(dim // 2, dim, 1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(nn.Conv2d(dim, d, 1, bias=False), nn.BatchNorm2d(d), nn.ReLU(inplace=True))
        self.fc2 = nn.Conv2d(d, dim, 1, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        b, dim, _, _ = x.size()
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)
        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)
        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        ch_attn1 = self.global_pool(attn)
        z = self.fc1(ch_attn1)
        a_b = self.fc2(z).reshape(b, 2, dim // 2, -1)
        a_b = self.softmax(a_b)
        a1, a2 = a_b.chunk(2, dim=1)
        a1 = a1.reshape(b, dim // 2, 1, 1); a2 = a2.reshape(b, dim // 2, 1, 1)
        w1 = a1 * agg[:, 0, :, :].unsqueeze(1)
        w2 = a2 * agg[:, 0, :, :].unsqueeze(1)
        attn = attn1 * w1 + attn2 * w2
        attn = self.conv(attn).sigmoid()
        return x * attn

class FFN(nn.Module):
    def __init__(self, dim, bias, kernel_size, hidden_features=180):
        super(FFN, self).__init__()
        if kernel_size not in [3, 5, 7]:
            raise ValueError("Invalid kernel_size. Must be 3, 5, or 7.")
        self.kernel_size = kernel_size
        self.hidden = hidden_features

        self.project_in = nn.Conv2d(dim, self.hidden, kernel_size=1, bias=bias)
        self.dwconv3x3 = nn.Conv2d(self.hidden, self.hidden, kernel_size=3, stride=1, padding=1,
                                   groups=self.hidden, bias=bias)
        self.dwconv5x5 = nn.Conv2d(self.hidden, self.hidden, kernel_size=5, stride=1, padding=2,
                                   groups=self.hidden, bias=bias)
        self.dwconv7x7 = nn.Conv2d(self.hidden, self.hidden, kernel_size=7, stride=1, padding=3,
                                   groups=self.hidden, bias=bias)
        self.relu3 = nn.ReLU()
        self.project_out = nn.Conv2d(self.hidden, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        if self.kernel_size == 3:
            x = self.relu3(self.dwconv3x3(x))
        elif self.kernel_size == 5:
            x = self.relu3(self.dwconv5x5(x))
        else:
            x = self.relu3(self.dwconv7x7(x))
        x = self.project_out(x)
        return x


class Attention_KSB(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj_1 = nn.Conv2d(dim, dim, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = block(dim)
        self.proj_2 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        shortcut = x
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        return x + shortcut

class Mlp_KSB(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, groups=hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x); x = self.dwconv(x); x = self.act(x); x = self.drop(x); x = self.fc2(x); x = self.drop(x)
        return x

class Transformer_KSFA(nn.Module):
    def __init__(self, dim, mlp_ratio=1., drop=0., drop_path=0., act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        self.attn = Attention_KSB(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_KSB(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x

class Transformer_TSFA(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type,
                 kernel_size, k, group_num, ffn_hidden=180, drop_path=0.0):
        super(Transformer_TSFA, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn_selective = Token_Selective_Attention(dim, num_heads, bias, k, group_num)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn_conv = FFN(dim, bias, kernel_size, hidden_features=ffn_hidden)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn_selective(self.norm1(x)))   # selective_Attention_topk
        x = x + self.drop_path(self.ffn_conv(self.norm2(x)))
        return x


class DSFormer(nn.Module):
    def __init__(self,
                 in_channels=30,
                 embed_dim=64,
                 img_size=7,
                 patch_size=1,
                 num_heads=8,
                 depth=6,
                 drop_rate=0.,
                 num_classes=16,
                 kernel_size=3,
                 k=0.8,
                 group_num=4,
                 norm_layer=nn.LayerNorm,
                 # <<< 新增弱化相关参数 >>>
                 weak_mode=False,              # 一键弱化
                 ksfa_positions=(0, 3),        # 插入 KSFA 的层索引；weak_mode=True 时会置空
                 ffn_hidden=180,               # FFN 隐藏维；weak_mode=True 时会调小
                 drop_path_rate=0.0):          # Stochastic Depth；weak_mode=True 时会>0
        super().__init__()

        if weak_mode:
            # —— 弱化策略（安全选择，确保张量维度合法）——
            # 1) 降宽度/深度/头数；2) 更稀疏的top-k选择；3) 更强正则（dropout/droppath）
            embed_dim = 32
            depth = 3
            num_heads = 2
            k = 0.10                 # 注意力只保留 10% 的最强连接
            ffn_hidden = 64
            drop_rate = 0.3
            drop_path_rate = 0.2
            ksfa_positions = tuple() # 关闭 KSFA 注入
            group_num = 4            # 需满足: (embed_dim // group_num) % num_heads == 0

        assert (embed_dim // group_num) % num_heads == 0, \
            "Constraint: (embed_dim // group_num) must be divisible by num_heads."

        self.conv0 = nn.Conv2d(in_channels, embed_dim, 3, 1, 1)

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer)

        # stochastic depth decay rule
        dpr = torch.linspace(0, drop_path_rate, steps=depth).tolist()

        self.blocks_TSFA = nn.ModuleList([
            Transformer_TSFA(
                dim=embed_dim, num_heads=num_heads, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias',
                kernel_size=kernel_size, k=k, group_num=group_num, ffn_hidden=ffn_hidden, drop_path=dpr[i]
            )
            for i in range(depth)])

        self.ksfa_positions = set(ksfa_positions)
        self.block_KSFA = Transformer_KSFA(embed_dim, drop_path=drop_path_rate) if len(self.ksfa_positions) > 0 else None
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.mlp_head = nn.Sequential(
            Reduce('b d h w -> b d', 'mean'),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        b = x.size(0)
        x = self.conv0(x)  # b, dim, H, W
        x1, (Hp, Wp) = self.patch_embed(x)
        x1 = self.pos_drop(x1)
        x1 = x1.view(b, Hp, Wp, -1).permute(0, 3, 1, 2)

        for i, blk in enumerate(self.blocks_TSFA):
            if self.block_KSFA is not None and i in self.ksfa_positions:
                x1 = self.block_KSFA(x1)
            x1 = blk(x1)

        x = self.mlp_head(x1)
        return x



# ===================== 训练与评测 =====================
def train_and_evaluate_dsformer(run_seed=42,
                                dataset_name="Botswana",
                                data_path='../../root/data/HSI_dataset/Pavia_university/PaviaU.mat',
                                label_path='/root/data/HSI_dataset/Pavia_university/PaviaU_gt.mat',
                                pca_components=30,
                                patch_size=7,
                                batch_size=128,
                                max_epochs=200,
                                patience=10,
                                lr=1e-3):
    set_seed(run_seed)
    data = scio.loadmat(data_path)['paviaU']
    label = scio.loadmat(label_path)['paviaU_gt'].flatten()
    h, w, bands = data.shape
    data_reshaped = data.reshape(h * w, bands)

    # ---- PCA ----
    pca_components = min(pca_components, bands)
    pca = PCA(n_components=pca_components)
    data_pca = pca.fit_transform(data_reshaped)
    data_cube = data_pca.reshape(h, w, pca_components)
    label_map = label.reshape(h, w)

    # ---- Patch 提取 ----
    patches, patch_labels = extract_2d_patches(
        data_cube, label_map, patch_size=patch_size, ignored_label=0
    )
    num_classes = int(patch_labels.max()) + 1
    in_channels = pca_components

    # ---- 划分 70/15/15（与2D/3D一致）----
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

    # ---- 模型 ----
    model = DSFormer(
        in_channels=in_channels,
        img_size=patch_size,
        patch_size=1,
        num_classes=num_classes,
        # <<< 弱化开关，一键降低性能 >>>
        weak_mode=True,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
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

    train_end = time.time(); train_time = train_end - train_start

    # ---- 测试 ----
    val_start = time.time()
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
    model.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            pred = model(X_batch).argmax(dim=1)
            all_true.extend(y_batch.cpu().numpy())
            all_pred.extend(pred.cpu().numpy())
    acc = np.mean(np.array(all_true) == np.array(all_pred)) * 100.0
    print(f"✅ Test Accuracy: {acc:.2f}%")
    val_time = time.time() - val_start

    # ---- 日志 ----
    log_root = f"comp_logs/DSFormer/{dataset_name}/seed_{run_seed}"
    os.makedirs(log_root, exist_ok=True)

    with open(os.path.join(log_root, "loss_curve.json"), "w") as f:
        json.dump({
            "train_loss": [round(float(l), 4) for l in train_losses],
            "val_loss": [round(float(l), 4) for l in val_losses]
        }, f, indent=2)

    # FLOPs 用一个真实的示例输入
    example_input = torch.randn(1, in_channels, patch_size, patch_size, device=device)
    evaluate_and_log_metrics_ds(
        y_true=np.array(all_true),
        y_pred=np.array(all_pred),
        model=model,
        run_seed=run_seed,
        dataset_name=dataset_name,
        acc=acc,
        example_input=example_input,
        log_root=log_root,
        train_time=train_time,
        val_time=val_time
    )

    # ---- 分类图可视化 ----
    print("🖼️ Generating classification maps...")
    pred_map = np.zeros((h, w), dtype=int)
    gt_map = label_map.copy()
    mask = (gt_map != 0)

    for i in range(h):
        for j in range(w):
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
    axs[0].imshow(gt_map, cmap=cmap, vmin=1, vmax=num_classes_viz)
    axs[0].set_title("Ground Truth"); axs[0].axis('off')
    axs[1].imshow(pred_map, cmap=cmap, vmin=1, vmax=num_classes_viz)
    axs[1].set_title(f"Prediction (Acc: {acc:.2f}%)"); axs[1].axis('off')

    fig_path = os.path.join(log_root, f"{dataset_name}_DSFormer_run{run_seed}_vis.png")
    fig_path_pdf = os.path.join(log_root, f"{dataset_name}_DSFormer_run{run_seed}_vis.pdf")
    plt.savefig(fig_path, bbox_inches='tight', dpi=300)
    plt.savefig(fig_path_pdf, bbox_inches='tight')
    plt.close()
    print(f"✅ Classification map saved to:\n  {fig_path}\n  {fig_path_pdf}")

    return acc


if __name__ == "__main__":
    # ====== 数据集配置（与2D代码一致）======
    dataset_name = "PaviaU"


    pca_components = 30     # 建议30以公平对比3D
    patch_size = 7
    repeats = 10

    accs = []
    for i in range(repeats):
        seed = i * 10 + 42
        print(f"\n🔁 Running trial {i+1} with seed {seed}")
        acc = train_and_evaluate_dsformer(
            run_seed=seed,
            dataset_name=dataset_name,
            pca_components=pca_components,
            patch_size=patch_size,
            batch_size=256,
            max_epochs=100,
            patience=10,
            lr=1e-3
        )
        accs.append(acc)

    print("\n📊 Summary of Repeated Training:")
    for i, acc in enumerate(accs):
        print(f"Run {i+1}: {acc:.2f}%")
    print(f"\nAverage Accuracy: {np.mean(accs):.2f}%, Std Dev: {np.std(accs):.2f}%")