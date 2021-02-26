'''
Author: your name
Date: 2020-11-20 10:10:49
LastEditTime: 2020-11-20 10:35:18
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /PyCode/project_demo/Datafountain/Learn/集成学习/test2.py
'''
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('Telco-Customer-Churn.csv')
pd.set_option('display.max_columns', None)  # 显示所有列
data.head(10)

# 查看csv文件中重复的值
dupNum = data.shape[0] - data.drop_duplicates().shape[0]
print('数据集中有%s列重复的值' % dupNum)

# 缺失值处理
data.isnull().any()

# 查看一些缺失值
data[data['TotalCharges'] == ' ']
#  convert_numeric如果为True，则尝试强制转换为数字，不可转换的变为NaN
data['TotalCharges'] = data['TotalCharges'].apply(pd.to_numeric, errors='coerce') 
print("此时TotalCharges是否已经转换为浮点型：", data['TotalCharges'].dtype == 'float')
print("此时TotalCharges存在%s行缺失样本。" % data['TotalCharges'].isnull().sum())

# 固定值填充
fnDf = data['TotalCharges'].fillna(0).to_frame()
print("如果采用固定值填充方法还存在%s行缺失样本。" % fnDf['TotalCharges'].isnull().sum())