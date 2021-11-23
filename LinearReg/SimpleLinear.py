import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l

# 创建一个简单的数据集
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)


def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)  # 数据打包
    # shuffle:是否随机打乱顺序
    return data.DataLoader(dataset, batch_size, shuffle=is_train)   # 数据封装


batch_size = 10
data_iter = load_array((features, labels), batch_size)


from torch import nn
net = nn.Sequential(nn.Linear(2, 1))    # Sequential：一个有序的容器，神经网络模块将按照在传入构造器的顺序依次被添加到计算图中执行
# nn.Linear(2, 1)：输入维度为2吗，输出维度为1

# 初始化模型参数
net[0].weight.data.normal_(0, 0.01)     # normal 使 data 值满足正态分布(第一个参数0表示均值，第二个参数0.01表示方差）
net[0].bias.data.fill_(0)

param = net.parameters()

# 定义 loss
loss = nn.MSELoss()
# 定义优化器
optimizer = torch.optim.SGD(net.parameters(), lr=0.03)
# 开始训练
num_epochs = 3
for epoch in range(num_epochs):
    for x, y in data_iter:
        l = loss(net(x), y)
        optimizer.zero_grad()   # 梯度清零
        l.backward()
        optimizer.step()        # 模型更新
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}: loss {l:f}')
