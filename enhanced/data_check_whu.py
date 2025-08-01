import scipy.io as scio
from scipy.io import loadmat
import numpy as np

def inspect_mat_and_gt(data_path, gt_path):
    # 载入高光谱图像
    data_mat = scio.loadmat(data_path)
    print(f"\n📁 Data file: {data_path}")
    for key in data_mat:
        if not key.startswith("__"):
            print(f"✅ Key: {key}, shape: {data_mat[key].shape}, dtype: {data_mat[key].dtype}")

    # 载入GT标签
    gt_mat = scio.loadmat(gt_path)
    print(f"\n📁 GT file: {gt_path}")
    for key in gt_mat:
        if not key.startswith("__"):
            print(f"✅ Key: {key}, shape: {gt_mat[key].shape}, dtype: {gt_mat[key].dtype}")

# 示例路径（替换为实际路径）
data_path = '../../root/data/HSI_dataset/Matlab_data_format/Matlab_data_format/WHU-Hi-HanChuan/WHU_Hi_HanChuan.mat'
gt_path = '../../root/data/HSI_dataset/Matlab_data_format/Matlab_data_format/WHU-Hi-HanChuan/WHU_Hi_HanChuan_gt.mat'

inspect_mat_and_gt(data_path, gt_path)
mat = loadmat(data_path)

print("Keys:", mat.keys())
for key in mat:
    if not key.startswith('__'):
        print(f"{key}: shape = {mat[key].shape}, dtype = {mat[key].dtype}")