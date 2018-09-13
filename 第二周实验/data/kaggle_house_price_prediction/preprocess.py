# -*- coding:utf-8 -*-
import pandas as pd

# 读取数据
data = pd.read_csv('kaggle_hourse_price_train.csv')

# 丢弃有缺失值的特征（列）
data.dropna(axis = 1, inplace = True)

# 只保留整数的特征
data = data[[col for col in data.dtypes.index if data.dtypes[col] == 'int64']]

# 保存处理后的结果到csv文件
data.to_csv("kaggle房价预测处理后数据.csv", index = False)