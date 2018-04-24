# !/usr/bin/python
# coding:utf-8
# Author :wuyy

from matplotlib import pyplot
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster   import KMeans
from scipy import sparse
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import pickle
from sklearn.externals import joblib


#加载数据
data = pd.read_table('jaingwei.txt',sep = ",")
data=data.T
x = data.ix[1:,0:5]
print(x)
card = data.ix[:,0]

x1 = np.array(x)
print("x1:",x1)
xx = preprocessing.scale(x1)

print("preprocessing.scale xx:",xx)
num_clusters = 3

clf = KMeans(n_clusters=num_clusters,  n_init=1, n_jobs = 1,verbose=1) #job=-1 并行化处理
clf.fit(xx)
print("label:",clf.labels_)
labels = clf.labels_
#score是轮廓系数
score = metrics.silhouette_score(xx, labels)
# clf.inertia_用来评估簇的个数是否合适，距离越小说明簇分的越好
print ("clf.inertia_",clf.inertia_)
print (score)
