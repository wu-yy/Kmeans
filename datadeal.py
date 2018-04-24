# !/usr/bin/python
#coding:utf-8
#author:wuyy
'''

数据预处理
'''
import pandas as pd
import  numpy as np
import time
import re

#加载文件
x=pd.read_table('info.txt',sep = ",")
x=x.dropna(axis=0)
a1=list(x.iloc[:,0])
a2=list(x.iloc[:,1])
a3=list(x.iloc[:,2])
print("数据表:",x)

#A是商品类别
dicta=dict(zip(a2,zip(a1,a3)))
print("dicta:",dicta)
A=list(dicta.keys())
#B是用户id
B=list(set(a1))

#创建商品类别字典
a = np.arange(len(A))
lista = list(a)
dict_class = dict(zip(A,lista))

#将商品分类写入
f=open('class.txt','w')
for k ,v in dict_class.items():
     f.write(str(k)+'\t'+str(v)+'\n')
f.close()

start=time.clock()
#创建大字典存储数据
dictall = {}
for i in range(len(a1)):
    if a1[i] in dictall.keys():
        value = dictall[a1[i]]
        j = dict_class[a2[i]]
        value[j] = a3[i]
        dictall[a1[i]]=value
    else:
        value = list(np.zeros(len(A)))
        j = dict_class[a2[i]]
        value[j] = a3[i]
        dictall[a1[i]]=value
print('dictall:',dictall)

#将字典转化为dataframe
dictall1 = pd.DataFrame(dictall)
dictall_matrix = dictall1.T
print("dictall_matrix:",dictall_matrix)
dictall_matrix.to_csv("data_matrix.txt",index=True,header=None)
# fw2=open("dictall_matrix.txt",'w')
# fw2.write(dictall_matrix)
# fw2.close()
dictall_matrix
end = time.clock()
print ("赋值过程运行时间是:%f s"%(end-start))

df=pd.DataFrame(columns=['id','id1'])
df[id]=1
print(df)