import torch
import scipy.io as scio

data_path = 'data/Pavia_university/PaviaU.mat'
label_path = 'data/Pavia_university/PaviaU_gt.mat'

data = scio.loadmat(label_path)
print("包含的键:", data.keys())

