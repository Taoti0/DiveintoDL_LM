import torch

# 生成张量
x = torch.arange(12)    # >>> tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])

# 查看张量形状
x.shape     # >>> torch.Size([12])
x.numel()   # >>> 12

# 修改张量形状
new_x = x.reshape(3, 4)
""" 
new_x >>> tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])
"""

# 生成指定张量
a = torch.tensor([[2, 1, 3, 4], [1, 2, 3, 4]])
"""
a >>> tensor([[2, 1, 3, 4],
        [1, 2, 3, 4]])
"""

# 运算
torch.exp(x)    # 指数运算
torch.log(x)    # 对数运算
x.sum()         # 元素相加

# 张量连结
X = torch.arange(12, dtype=torch.float32).reshape(3, 4)
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 2, 2, 1]])
X_Y_0 = torch.cat((X, Y), dim=0)
"""
X_Y_0 >>> tensor([[ 0.,  1.,  2.,  3.],
        [ 4.,  5.,  6.,  7.],
        [ 8.,  9., 10., 11.],
        [ 2.,  1.,  4.,  3.],
        [ 1.,  2.,  3.,  4.],
        [ 4.,  2.,  2.,  1.]])
"""
X_Y_1 = torch.cat((X, Y), dim=1)
"""
X_Y_1 >>> tensor([[ 0.,  1.,  2.,  3.,  2.,  1.,  4.,  3.],
        [ 4.,  5.,  6.,  7.,  1.,  2.,  3.,  4.],
        [ 8.,  9., 10., 11.,  4.,  2.,  2.,  1.]])
"""

# 索引
c = X[0:2, :]   # 取第0行到第2行所有列

# 内存地址
address = id(X)   # >>> 140241210499456

# 数据类型转换
A = X.numpy()
B = torch.tensor(A)
typeA, typeB = type(A), type(B)
"""
typeA >>> <class 'numpy.ndarray'>
typeB >>> <class 'torch.Tensor'>
"""

a = torch.tensor([3.5])
print(a, a.item(), float(a), int(a))    # 张量转换为标量
"""
>>> tensor([3.5000])
    3.5
    3.5
    3
"""

