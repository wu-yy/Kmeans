# !/usr/bin/python
#coding:utf-8
#author:wuyy
'''

AE模型（Auto-encoder）
主要是能够把数据缩放，如果你输入的维数比较大，譬如实际的特征是几千维的，全部拿到算法里跑，效果不见得好，
因为并不是所有特征都是有用的，用AE模型后，你可以压缩成m维（就是隐含层的节点数）

'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing


class AutoEncoder():
    """ Auto Encoder
    layer      1     2    ...    ...    L-1    L
      W        0     1    ...    ...    L-2
      B        0     1    ...    ...    L-2
      Z              0     1     ...    L-3    L-2
      A              0     1     ...    L-3    L-2
    """

    def __init__(self, X, Y, nNodes):
        # training samples
        self.X = X
        self.Y = Y
        # number of samples
        self.M = len(self.X)
        # layers of networks
        self.nLayers = len(nNodes)
        # nodes at layers
        self.nNodes = nNodes
        # parameters of networks
        self.W = list()
        self.B = list()
        self.dW = list()
        self.dB = list()
        self.A = list()
        self.Z = list()
        self.delta = list()
        for iLayer in range(self.nLayers - 1):
            self.W.append(
                np.random.rand(nNodes[iLayer] * nNodes[iLayer + 1]).reshape(nNodes[iLayer], nNodes[iLayer + 1]))
            self.B.append(np.random.rand(nNodes[iLayer + 1]))
            self.dW.append(np.zeros([nNodes[iLayer], nNodes[iLayer + 1]]))
            self.dB.append(np.zeros(nNodes[iLayer + 1]))
            self.A.append(np.zeros(nNodes[iLayer + 1]))
            self.Z.append(np.zeros(nNodes[iLayer + 1]))
            self.delta.append(np.zeros(nNodes[iLayer + 1]))

        # value of cost function
        self.Jw = 0.0
        # active function (logistic function)
        self.sigmod = lambda z: 1.0 / (1.0 + np.exp(-z))
        # learning rate 1.2
        self.alpha = 2.5
        # steps of iteration 30000
        self.steps = 10000

    def BackPropAlgorithm(self):
        # clear values
        self.Jw -= self.Jw
        for iLayer in range(self.nLayers - 1):
            self.dW[iLayer] -= self.dW[iLayer]
            self.dB[iLayer] -= self.dB[iLayer]
        # propagation (iteration over M samples)
        for i in range(self.M):
            # Forward propagation
            for iLayer in range(self.nLayers - 1):
                if iLayer == 0:  # first layer
                    self.Z[iLayer] = np.dot(self.X[i], self.W[iLayer])
                else:
                    self.Z[iLayer] = np.dot(self.A[iLayer - 1], self.W[iLayer])
                self.A[iLayer] = self.sigmod(self.Z[iLayer] + self.B[iLayer])
                # Back propagation
            for iLayer in range(self.nLayers - 1)[::-1]:  # reserve
                if iLayer == self.nLayers - 2:  # last layer
                    self.delta[iLayer] = -(self.X[i] - self.A[iLayer]) * (self.A[iLayer] * (1 - self.A[iLayer]))
                    self.Jw += np.dot(self.Y[i] - self.A[iLayer], self.Y[i] - self.A[iLayer]) / self.M
                else:
                    self.delta[iLayer] = np.dot(self.W[iLayer].T, self.delta[iLayer + 1]) * (
                    self.A[iLayer] * (1 - self.A[iLayer]))
                # calculate dW and dB
                if iLayer == 0:
                    self.dW[iLayer] += self.X[i][:, np.newaxis] * self.delta[iLayer][:, np.newaxis].T
                else:
                    self.dW[iLayer] += self.A[iLayer - 1][:, np.newaxis] * self.delta[iLayer][:, np.newaxis].T
                self.dB[iLayer] += self.delta[iLayer]
                # update
        for iLayer in range(self.nLayers - 1):
            self.W[iLayer] -= (self.alpha / self.M) * self.dW[iLayer]
            self.B[iLayer] -= (self.alpha / self.M) * self.dB[iLayer]

    def PlainAutoEncoder(self):
        for i in range(self.steps):
            self.BackPropAlgorithm()
            print("step:%d" % i, "Jw=%f" % self.Jw)

    def ValidateAutoEncoder(self):
        a = np.array([i for i in range(1, 6)])
        df = pd.DataFrame(a, columns=['weidu'])
        for i in range(self.M):
            print( self.X[i])

            for iLayer in range(self.nLayers - 1):
                if iLayer == 0:  # input layer
                    self.Z[iLayer] = np.dot(self.X[i], self.W[iLayer])
                else:
                    self.Z[iLayer] = np.dot(self.A[iLayer - 1], self.W[iLayer])
                self.A[iLayer] = self.sigmod(self.Z[iLayer] + self.B[iLayer])
                print("\t layer=%d" % iLayer, self.A[iLayer])
                if iLayer==0:
                    df[str(i+1)]=self.A[iLayer]

        df.to_csv("jaingwei.txt",index=False)




data = []
index = []
f = open('./data_matrix.txt', 'r')
for line in f.readlines():
    ss = line.replace('\n', '').split(',')
    index.append(ss[0])
    ss1 = ss[1:]
    tmp = []
    for i in range(len(ss1)):
        tmp.append(float(ss1[i]))
    data.append(tmp)
f.close()

x = np.array(data)
# 归一化处理
xx = preprocessing.scale(x)
nNodes = np.array([10, 5, 10])
ae3 = AutoEncoder(xx, xx, nNodes)
ae3.PlainAutoEncoder()
ae3.ValidateAutoEncoder()

print("ae结果：",ae3.A[0])

# # 这是个例子，输出的结果也是这个
# xx = np.array([[0,0,0,0,0,0,0,1], [0,0,0,0,0,0,1,0], [0,0,0,0,0,1,0,0], [0,0,0,0,1,0,0,0],[0,0,0,1,0,0,0,0], [0,0,1,0,0,0,0,0]])
# nNodes = np.array([ 8, 3, 8 ])
# ae2 = AutoEncoder(xx,xx,nNodes)
# ae2.PlainAutoEncoder()
# ae2.ValidateAutoEncoder()