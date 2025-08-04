import h5py
from scipy.io import loadmat
import numpy as np

def inspect_hdf5_mat(data_path):
    with h5py.File(data_path, 'r') as f:
        print(f"\n📁 HDF5 MAT File: {data_path}")
        for key in f.keys():
            obj = f[key]
            if isinstance(obj, h5py.Dataset):
                print(f"✅ Dataset: {key}, shape: {obj.shape}, dtype: {obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"📂 Group: {key} → keys: {list(obj.keys())}")

# 示例路径
data_path = '../../root/data/HSI_dataset/Houston/Houston18.mat'
gt_path = '../../root/data/HSI_dataset/Houston/Houston18_7gt.mat'

inspect_hdf5_mat(data_path)
inspect_hdf5_mat(gt_path)

# 载入数据
with h5py.File(data_path, 'r') as f:
    data = np.array(f['ori_data']).astype(np.float32)  # shape: (C, H, W)
    data = np.transpose(data, (1, 2, 0))  # → shape: (H, W, C)

with h5py.File(gt_path, 'r') as f:
    label_map = np.array(f['map']).astype(np.int32)
    label_map = label_map.T  # 需要转置以对齐 data


# 归一化（按像素或整体，以下为整体归一化）
data = data / data.max()

# 检查形状
print("data shape:", data.shape)        # (1476, 256, 145)
print("label_map shape:", label_map.shape)  # (1476, 256)