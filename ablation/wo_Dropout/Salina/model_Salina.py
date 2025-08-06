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


class CNN3D(nn.Module):
    def __init__(self, in_channels=30, num_classes=16, use_attention=True,
                 use_bn=False, use_dropout=False, use_conv2=True, use_fc64=True):
        super(CNN3D, self).__init__()
        self.use_attention = use_attention
        self.use_bn = use_bn
        self.use_dropout = use_dropout
        self.use_conv2 = use_conv2
        self.use_fc64 = use_fc64

        self.spectral_conv = nn.Conv3d(1, 8, kernel_size=(7,1,1), padding=(3,0,0))
        self.spatial_conv = nn.Conv3d(8, 8, kernel_size=(1,3,3), padding=(0,1,1))
        if self.use_bn:
            self.bn1 = nn.BatchNorm3d(8)
        self.relu = nn.ReLU()

        if self.use_attention:
            self.attention = nn.Sequential(
                nn.AdaptiveAvgPool3d((1, 1, 1)),
                nn.Conv3d(8, 4, 1),
                nn.ReLU(),
                nn.Conv3d(4, 8, 1),
                nn.Sigmoid()
            )

        if self.use_conv2:
            self.conv2 = nn.Conv3d(8, 16, kernel_size=3, padding=1)
            if self.use_bn:
                self.bn2 = nn.BatchNorm3d(16)
            self.final_channels = 16
        else:
            self.final_channels = 8

        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        if self.use_fc64:
            fc_layers = [
                nn.Flatten(),
                nn.Linear(self.final_channels, 64),
                nn.ReLU()
            ]
            if self.use_dropout:
                fc_layers.append(nn.Dropout(0.5))
            fc_layers.append(nn.Linear(64, num_classes))
        else:
            fc_layers = [
                nn.Flatten(),
                nn.Linear(self.final_channels, num_classes)
            ]

        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x):
        x = self.spectral_conv(x)
        x = self.spatial_conv(x)
        if self.use_bn:
            x = self.bn1(x)
        x = self.relu(x)
        if self.use_attention:
            att_map = self.attention(x)
            x = x * att_map
        if self.use_conv2:
            x = self.conv2(x)
            if self.use_bn:
                x = self.bn2(x)
            x = self.relu(x)
        x = self.pool(x)
        x = self.fc(x)
        return x


def extract_3d_patches(img_cube, label_map, patch_size=7, spectral_channels=30, ignored_label=0):
    assert patch_size % 2 == 1
    H, W, C = img_cube.shape
    pad = patch_size // 2
    img_cube = img_cube[:, :, :spectral_channels]
    padded_img = np.pad(img_cube, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    padded_label = np.pad(label_map, ((pad, pad), (pad, pad)), mode='constant')
    patches, labels = [], []
    for i in range(pad, H + pad):
        for j in range(pad, W + pad):
            label = padded_label[i, j]
            if label == ignored_label:
                continue
            spatial_patch = padded_img[i - pad:i + pad + 1, j - pad:j + pad + 1, :]
            patch = np.transpose(spatial_patch, (2, 0, 1))[np.newaxis, ...]
            patches.append(patch)
            labels.append(label - 1)
    return np.array(patches, dtype='float32'), np.array(labels, dtype='int64')


def evaluate_and_log_metrics(y_true, y_pred, model, run_seed, dataset_name, acc, train_time=None, val_time=None):
    log_dir = f"ablation_logs/wo_Dropout/{dataset_name}/seed_{run_seed}"
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
        "per_class_accuracy": [round(x, 4) for x in per_class_acc.tolist()]
    }
    with open(os.path.join(log_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    flops, params = get_model_complexity_info(model, (1, 30, 7, 7), as_strings=False, print_per_layer_stat=False)
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


def train_and_evaluate(run_seed=42, dataset_name="Salina"):
    set_seed(run_seed)
    data_path = '../../root/data/HSI_dataset/Salinas/Salinas_corrected.mat'
    label_path = '../../root/data/HSI_dataset/Salinas/Salinas_gt.mat'
    data = scio.loadmat(data_path)['salinas_corrected']
    label = scio.loadmat(label_path)['salinas_gt'].flatten()
    h, w, bands = data.shape
    data_reshaped = data.reshape(h * w, bands)

    pca = PCA(n_components=30)
    data_pca = pca.fit_transform(data_reshaped)
    data_cube = data_pca.reshape(h, w, 30)
    label_map = label.reshape(h, w)

    patches, patch_labels = extract_3d_patches(data_cube, label_map, patch_size=7, spectral_channels=30)
    num_classes = int(patch_labels.max()) + 1

    X_train_full, X_test, y_train_full, y_test = train_test_split(patches, patch_labels, test_size=0.15, stratify=patch_labels, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.176, stratify=y_train_full, random_state=42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN3D(in_channels=30, num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)), batch_size=128, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)), batch_size=128)
    test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)), batch_size=128)

    best_val_loss = float('inf')
    patience, patience_counter = 10, 0
    best_model_weights = None
    train_start = time.time()
    train_losses = []
    val_losses = []

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

    model.load_state_dict(best_model_weights)
    model.eval()
    all_true_labels, all_pred_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = model(X_batch).argmax(dim=1)
            all_true_labels.extend(y_batch.cpu().numpy())
            all_pred_labels.extend(preds.cpu().numpy())

    acc = np.mean(np.array(all_true_labels) == np.array(all_pred_labels)) * 100
    print(f"✅ Test Accuracy: {acc:.2f}%")

    # 日志路径
    log_dir = f"ablation_logs/wo_Dropout/{dataset_name}/seed_{run_seed}"
    os.makedirs(log_dir, exist_ok=True)

    # 保存 loss 曲线
    loss_log = {
        "train_loss": [round(l, 4) for l in train_losses],
        "val_loss": [round(l, 4) for l in val_losses]
    }
    with open(os.path.join(log_dir, "loss_curve.json"), "w") as f:
        json.dump(loss_log, f, indent=2)

    evaluate_and_log_metrics(
        y_true=np.array(all_true_labels),
        y_pred=np.array(all_pred_labels),
        model=model,
        run_seed=run_seed,
        dataset_name=dataset_name,
        acc=acc,
        train_time=train_time
    )

    # -------- 分类图像可视化 --------
    print("🖼️ Generating classification maps...")
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
                pred_map[i, j] = pred_label + 1  # 为了可视化，从1开始

    # 颜色映射
    cmap = mcolors.ListedColormap(plt.colormaps['tab20'].colors[:num_classes])

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    axs[0].imshow(label_map, cmap=cmap, vmin=1, vmax=num_classes)
    axs[0].set_title("Ground Truth")
    axs[0].axis('off')

    axs[1].imshow(pred_map, cmap=cmap, vmin=1, vmax=num_classes)
    axs[1].set_title(f"Prediction (Acc: {acc:.2f}%)")
    axs[1].axis('off')

    # 保存图像
    fig_path_pdf = os.path.join(log_dir, f"{dataset_name}_run{run_seed}_vis.pdf")
    plt.savefig(fig_path_pdf, bbox_inches='tight')
    plt.close()
    print(f"✅ Classification map saved to: {fig_path_pdf}")

    return acc


if __name__ == "__main__":
    repeats = 10
    accs = []
    for i in range(repeats):
        print(f"\n🔁 Running trial {i+1} with seed {i*10+42}")
        acc = train_and_evaluate(run_seed=i*10+42)
        accs.append(acc)
    print("\n📊 Summary of Repeated Training:")
    for i, acc in enumerate(accs):
        print(f"Run {i+1}: {acc:.2f}%")
    print(f"\nAverage Accuracy: {np.mean(accs):.2f}%, Std Dev: {np.std(accs):.2f}%")