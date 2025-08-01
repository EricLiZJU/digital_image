import h5py
from scipy.io import loadmat

label_path = '../../root/data/HSI_dataset/Chikusei_MATLAB/HyperspecVNIR_Chikusei_20140729_Ground_Truth.mat'
mat = loadmat(label_path)

# 提取结构体字段
gt_struct = mat['GT']  # shape = (1,1)

# 提取标签矩阵
label_map = gt_struct['gt'][0, 0]  # dtype=object, shape=(2335, 2517)

print("Label map shape:", label_map.shape)
print("Label map dtype:", label_map.dtype)