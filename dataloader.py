import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# 载入训练数据
class CharacterDataset(Dataset):
    def __init__(self, datas, labels, transform=None, target_transform=None):
        self.datas = datas
        self.labels = labels
        self.labels = torch.Tensor(labels).long()
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        data = self.datas[index].reshape(-1, 1, 1)  # 运行CNN时需要将reshape去掉
        label = self.labels[index]

        if self.transform is not None:
            data = self.transform(data)  # 转换为tensor格式
        return data, label

    def __len__(self):
        return self.datas.shape[0]

# # 训练集载入测试
# path = './dataset/train.txt'
# transform = transforms.Compose([transforms.ToTensor()])
# characters = CharacterDataset(path, transform=transform)
# image, label = characters.__getitem__(0)
# print(image.shape)
# image.show()
# print(characters.__len__())

# # 测试集载入测试
# path = './dataset/data_road/test.txt'
# transform = transforms.Compose([transforms.ToTensor()])
# road = RoadDataset_test(path, transform=transform)

# data_loader = DataLoader(road, batch_size=8, shuffle=True)
# for index, images in enumerate(data_loader):
#     print(images.size())
#     break

# print(road.__len__())
# image = road.__getitem__(0)
# print(image.size())
# image.show()
# # # label.show()
# print(label.shape)
# # print(type(image))
