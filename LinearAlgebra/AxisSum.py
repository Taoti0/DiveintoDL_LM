import torch

"""
 轴求和就是去掉 axis 对应的那一个维度
 当 keepdims = True 时，保留维度，但相当于将该维度压缩为一列(行）
"""

# 创建一个张量（三维矩阵）
a = torch.ones(2, 5, 4)
print(a)
"""
a >>> tensor([[[1., 1., 1., 1.],
         [1., 1., 1., 1.],
         [1., 1., 1., 1.],
         [1., 1., 1., 1.],
         [1., 1., 1., 1.]],

        [[1., 1., 1., 1.],
         [1., 1., 1., 1.],
         [1., 1., 1., 1.],
         [1., 1., 1., 1.],
         [1., 1., 1., 1.]]])
"""

# 所有维度求和
print(a.sum())
"""
a.sum() >>> tensor(40.)
a.sum(),shape >>> torch.Size([])
"""

# 第一个维度求和
print(a.sum(axis=1))
"""
a.sum(axis=1) >>> tensor([[5., 5., 5., 5.],
        [5., 5., 5., 5.]])
a.sum(axis=1).shape >>> torch.Size([2, 4])
"""

# 第一个维度求和，且保留维度数
print(a.sum(axis=1, keepdims=True))
"""
a.sum(axis=1) >>> tensor([[[5., 5., 5., 5.]],

        [[5., 5., 5., 5.]]])
a.sum(axis=1).shape >>> torch.Size([2, 1, 4])
"""
