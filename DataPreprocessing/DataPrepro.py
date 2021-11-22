import os


# # 创建一个案例用的文件
# os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house.csv')
# with open(data_file, 'w') as f:
#     f.write('NumRooms, Alley, Price\n')     # 列名
#     f.write('NA, Pave, 127500\n')
#     f.write('2, NA, 106000\n')
#     f.write('4, NA, 178100\n')
#     f.write('NA, NA, 140000\n')


import pandas as pd


data = pd.read_csv(data_file)
# print(data)
"""
   NumRooms  Alley   Price
0       NaN   Pave  127500
1       2.0     NA  106000
2       4.0     NA  178100
3       NaN     NA  140000
"""

""" 处理缺失数据"""
numRooms_alley = data.iloc[:, 0:2]
numRooms_alley = numRooms_alley.fillna(numRooms_alley.mean())   # 按均值填充
# print(numRooms_alley)
"""
   NumRooms  Alley
0       3.0   Pave
1       2.0     NA
2       4.0     NA
3       3.0     NA
"""
numRooms_alley = pd.get_dummies(numRooms_alley, dummy_na=True)  # 非数值的列无法通过均值填充，此方法将这类列转换为数值(类似于 OneHot 编码)
# print(numRooms_alley)
"""
   NumRooms   Alley_ NA   Alley_ Pave   Alley_nan
0       3.0           0             1           0
1       2.0           1             0           0
2       4.0           1             0           0
3       3.0           1             0           0
"""

 
""" 转换为张量 """
import torch
price = data.iloc[:, 2]
price = torch.tensor(price.values)
# print(price)
""" 
tensor([127500, 106000, 178100, 140000]) 
"""
