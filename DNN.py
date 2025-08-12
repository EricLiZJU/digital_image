# 超光谱图像分割

import numpy as np
import scipy.io as scio
import random

import torch
# 模型构建
import torch.nn as nn
# 模型训练
import torch.optim as optim
from torch.autograd import Variable
# 数据准备
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataloader import CharacterDataset

# --- 基本参数设置 ---
number_classes = 16  # 分割类别（不含背景类）
features_num = 200  # 特征数
loops = 1

epochs = 200
batch_size = 1024
neurons = 256
dropout_c = 0.3
rate = 0.7  # 训练数据占比
epoch_best = 200
device = torch.device("mps" if torch.mps.is_available() else "cpu")
# ------------------------------------------------------------ 定义网络结构 ------------------------------------------------------------
print('构建模型……')
print(device)


class Classification(nn.Module):
    # 定义构造函数
    def __init__(self):
        super(Classification, self).__init__()
        # 定义网络结构
        self.output = nn.Sequential(
            nn.Flatten(),

            nn.Linear(features_num, int(neurons / 2)),
            nn.ReLU(True),
            # # 随机关闭神经元
            nn.Dropout(dropout_c),
            nn.BatchNorm1d(int(neurons / 2)),

            nn.Linear(int(neurons / 2), neurons),
            nn.ReLU(True),
            # # 随机关闭神经元
            nn.Dropout(dropout_c),
            nn.BatchNorm1d(neurons),

            nn.Linear(neurons, int(neurons / 2)),
            nn.ReLU(True),
            # # 随机关闭神经元
            nn.Dropout(dropout_c),
            nn.BatchNorm1d(int(neurons / 2)),

            # 输出检测结果
            nn.Linear(int(neurons / 2), number_classes),
        )

    # 定义前向传播函数
    def forward(self, x):
        output = self.output(x)  # 获得分类结果
        return output


# ------------------------------------------------------- 数据准备 ------------------------------------------------------
print('载入数据……')
# 读取训练数据

data_path = 'data/Indian_pines/Indian_pines_corrected.mat'
label_path = 'data/Indian_pines/Indian_pines_gt.mat'
data = scio.loadmat(data_path)['indian_pines_corrected'].reshape(-1, 200)
label = scio.loadmat(label_path)['indian_pines_gt'].flatten()

"""
data_path = 'data/Pavia_university/PaviaU.mat'
label_path = 'data/Pavia_university/PaviaU_gt.mat'
data = scio.loadmat(data_path)['paviaU'].reshape(-1, 200)
label = scio.loadmat(label_path)['paviaU_gt'].flatten()
"""
# 统计各类像素的数据
count = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: [], 11: [], 12: [], 13: [], 14: [],
         15: []}
for i in range(label.shape[0]):
    if label[i] != 0:
        count[label[i] - 1].append(data[i])

accuracy_all = []
for loop in range(loops):
    if loops != 1:
        print('loop:', loop)

    # 打乱数据
    for key in count:
        random.shuffle(count[key])

    # 构造训练集与测试集
    train_datas = []
    train_labels = []
    test_datas = []
    test_labels = []
    count_train = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: [], 11: [], 12: [], 13: [],
                   14: [], 15: []}
    number_max_train = 0  # 统计数据量最多的类别的样本数，作为其他类别数据扩充的标准
    # 获取基本的训练集和测试集
    for key in count:
        train_num = int(count[key].__len__() * rate)  # 获取当前类中用于训练的样本数
        if number_max_train < train_num:
            number_max_train = train_num
        for j in range(count[key].__len__()):  # 将当前样本拆分到训练集与测试集
            if j < train_num:
                count_train[key].append(count[key][j])
            else:
                test_datas.append(count[key][j])
                test_labels.append(key)
    # 训练集数据扩充（重复）
    for key in count_train:
        number_temp_train = count_train[key].__len__()
        for i in range(number_max_train - number_temp_train):
            count_train[key].append(count_train[key][i % number_temp_train])
    # 交替放置各类数据（shuffle）
    for i in range(number_max_train):
        for key in range(number_classes):
            train_datas.append(count_train[key][i])
            train_labels.append(key)

    train_datas = np.array(train_datas, dtype='float32')
    train_labels = np.array(train_labels)
    test_datas = np.array(test_datas, dtype='float32')
    test_labels = np.array(test_labels)

    # 数据特征内归一化（样本间同一位置的特征归一化）
    mean = train_datas.mean(axis=0)
    std = train_datas.std(axis=0)
    train_datas = (train_datas - mean) / std
    test_datas = (test_datas - mean) / std

    # ------------------------------------------------------------ 网络训练 ------------------------------------------------------------
    # 实例化模型
    CC = Classification().to(device)

    # 构造损失函数
    lossFun = nn.CrossEntropyLoss()  # 已经包含了softmax，不能再在网络的最后添加softmax，否则不收敛

    # 构造参数优化器
    optimizer = optim.Adadelta(CC.parameters())

    # # 载入并更新网络、优化器权重参数
    # weights_net = torch.load('./models/DNN_NOBG_final.pth')
    # CC.load_state_dict(weights_net)
    # weights_optimizer = torch.load('./models/DNN_NOBG_optimizer_final.pth')
    # optimizer.load_state_dict(weights_optimizer)

    # 训练数据准备
    transform = transforms.Compose([transforms.ToTensor(), ])
    characters_train = CharacterDataset(train_datas, train_labels, transform=transform)
    data_loader_train = DataLoader(characters_train, batch_size=batch_size, shuffle=False)

    # 测试数据准备
    transform = transforms.Compose([transforms.ToTensor(), ])
    characters_test = CharacterDataset(test_datas, test_labels, transform=transform)
    data_loader_test = DataLoader(characters_test, batch_size=batch_size, shuffle=False)

    # 训练模型
    loss_epochs_train = []  # 训练过程中每个epoch的平均损失保存在此列表中，用于显示
    loss_epochs_test = []
    epochs_x = []  # 显示用的横坐标
    print('开始训练……')
    for epoch in range(epochs):
        # 模型训练
        CC.train()  # 训练模式，dropout开启
        loss_train = []
        for index, (images, labels) in enumerate(data_loader_train):
            images_cuda = Variable(images).to(device)  # 将图像载入cuda
            CC_output = CC(images_cuda)  # 前向传播，输出尺寸为(batch size, number_classes)
            labels_cuda = Variable(labels).to(device)  # 将标签载入cuda
            loss = lossFun(CC_output, labels_cuda)  # 计算误差
            optimizer.zero_grad()  # 将梯度初始化为零
            loss.backward()  # 反向传播
            optimizer.step()  # 更新所有参数
            loss_train.append(loss.data.item())
            if loops == 1:
                print('epoch:{}, batch:{}, loss:{:.6f}'.format(epoch + 1, index + 1, loss.data.item()))
        if (epoch + 1) % 10 == 0:  # 每训练10个epoch保存一次模型
            torch.save(CC.state_dict(), './models/DNN_NOBG_' + str(epoch + 1) + '.pth')
            torch.save(optimizer.state_dict(), './models/DNN_NOBG_optimizer_' + str(epoch + 1) + '.pth')

        # 模型测试（每个epoch使用测试集进行一次验证）
        CC.eval()  # 评估模式，dropout关闭
        loss_test = []
        # print('开始验证……')
        for index, (images, labels) in enumerate(data_loader_test):
            images_cuda = Variable(images).to(device)  # 将图像载入cuda
            CC_output = CC(images_cuda)  # 前向传播，输出尺寸为(batch size, number_classes)
            labels_cuda = Variable(labels).to(device)  # 将标签载入cuda
            loss = lossFun(CC_output, labels_cuda)  # 计算误差
            loss_test.append(loss.data.item())
        # print('平均损失：', np.mean(loss_test))

        # 记录损失
        loss_epochs_train.append(np.mean(loss_train))
        loss_epochs_test.append(np.mean(loss_test))
        epochs_x.append(epoch + 1)

    # # 最终模型保存
    torch.save(CC.state_dict(), './models/DNN_NOBG_final.pth')
    torch.save(optimizer.state_dict(), './models/DNN_NOBG_optimizer_final.pth')

    # 损失可视化
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.rcParams['font.family'] = 'Times New Roman'

    plt.figure('loss')
    plt.plot(epochs_x, loss_epochs_train, label='Training loss')
    plt.plot(epochs_x, loss_epochs_test, label='Validation loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(frameon=False)
    plt.title(f'loss of DNN on Indian_pines dataset', fontsize=12, y=1.02)
    # 添加副标题，使用 Axes 坐标系 (0,0)-(1,1)，y=1.02 就在标题正下方
    plt.text(0.5, 0.96,
             f'(batch_size:{batch_size}, neurons:{neurons}, dropout_rate:{dropout_c})',
             fontsize=10, ha='center', transform=plt.gca().transAxes)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("figures/Indian_pines_DNN_loss_8.pdf", bbox_inches='tight')
    plt.show()


    epoch_best = int(loss_epochs_test.index(min(loss_epochs_test)) / 10 + 0.6) * 10
    if epoch_best < 10:
        epoch_best = 10
    print('epoch_best:', epoch_best)

    # ------------------------------------------------------------ 网络测试 ------------------------------------------------------------
    print('开始测试……')
    # 模型实例化
    CC_temp = Classification()
    CC_temp.eval()  # 评估模式，dropout关闭
    # 载入并更新网络权重参数
    weights_net = torch.load(
        './models/DNN_NOBG_' + str(
            epoch_best) + '.pth', map_location='cpu')
    CC_temp.load_state_dict(weights_net)

    # 添加一个softmax网络（原始网络缺一个softmax）
    softmax = nn.Sequential(nn.Softmax(dim=1))

    # 测试集分类测试
    error_detail = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    number_detail = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    number_all = 0
    error_all = 0

    test_datas = np.array(test_datas, dtype='float32')
    test_labels = np.array(test_labels)
    for i in range(test_datas.shape[0]):
        image = test_datas[i].reshape(-1, 1, 1)
        label = test_labels[i]
        transform = transforms.Compose([transforms.ToTensor(), ])
        image = transform(image).view(1, -1, 1, 1)
        output = softmax(CC_temp(image))
        index = output.topk(1)[1].numpy()[0][0]
        if label != index:
            error_all += 1
            error_detail[label] += 1
        number_all += 1
        number_detail[label] += 1
    print('测试图像数量: {}, 误分类数量: {}, 分类准确率: {:.2f}%'.format(number_all, error_all,
                                                                         (1 - error_all / number_all) * 100))
    if loops == 1:
        for i in range(number_detail.__len__()):
            print('class:{:2}, {:4} / {:4} = {:.2f}%'.format(i, number_detail[i] - error_detail[i], number_detail[i],
                                                             (1 - error_detail[i] / number_detail[i]) * 100))

    if loops != 1:
        accuracy_all.append((1 - error_all / number_all) * 100)
        print('当前平均精度: {:.2f}%'.format(np.mean(accuracy_all)))
if loops != 1:
    print('accuracy_all:\n', accuracy_all)

