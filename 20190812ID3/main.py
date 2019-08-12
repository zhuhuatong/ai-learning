# -*- coding: utf-8 -*-
import create as ct
import draw as dr
import yanzheng as yz
from sklearn import datasets
import numpy as np
import lisanhua as lsh
import random

#加载iris数据集
iris = datasets.load_iris()
all_data = iris.data[:,:]
all_target = iris.target[:]
labels = iris.feature_names[:]

#常量定义
n = 150#数据集总数
m = int(n*2/3)#创建用的数据量
q = 4#数据维度
l = 7#离散化个数

#对数据离散化
a = []
all_data,a = lsh.lsh(all_data,l)

#将target和数据合并
all_data = all_data.tolist()
all_target = all_target.tolist()
for i in range(len(all_target)):
	all_data[i].append(all_target[i])

#将数据打乱
random.shuffle(all_data)

#创建决策树数据集
cj_data = all_data[:m]

#创建决策树
myTree=ct.createTree(cj_data,labels)

#创建验证数据集
all_data = np.array(all_data)#转化为numpy
yz_target = np.array(all_data[m:n,q:q+1])
yz_data = np.array(all_data[m:n,:q])
yz_labels = np.array(iris.feature_names[:])

#验证决策树正确率
yz_shu = yz.yanzheng(myTree,yz_data,yz_labels,yz_target)
yz_bfb = float(yz_shu)/(n-m)

#结果反馈
print(myTree)
print(yz_shu)
print(yz_bfb)
dr.createPlot(myTree)

