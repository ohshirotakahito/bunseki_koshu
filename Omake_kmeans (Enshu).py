# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 16:23:48 2021

@author: ohshi
"""
# -*- coding: utf-8 -*-
"""
このプラグラムはk-meansの挙動を理解するためのプログラムです。
実際に使う場合はこのプログラムを参考にするのではなく、
他のライブラリを使用することを強く推奨します。
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

file = os.path.join(os.getcwd(),'testdata','Cluster1.csv')
x = np.genfromtxt(file,delimiter=',')#csvファイルのデータ区切りを指定

n =0

plt.plot(x[:,0],x[:,1],'ko',alpha=0.15 )
plt.savefig('kmeans-'+str(n))
n +=1
plt.show()

classes = np.random.randint(3,size=len(x) ) 

k = 3 
dim = x.shape[1]

means = np.zeros([k, dim ] )
colors = ['r','b','g']

for i in range(k):
    means[i] = np.average(x[classes==i], axis=0)
    
    plt.plot(x[classes==i][:,0],x[classes==i][:,1], colors[i]+'o',alpha=0.15 ) 
    plt.plot(means[i][0],means[i][1], colors[i]+'x' ,markersize=10)
plt.savefig('kmeans-'+str(n))
n +=1

plt.show()

max_count = 1000

itr=0
while  itr < max_count:
    itr += 1
    distances = np.zeros([k, len(x) ] )
    for i in range(k):
        for j in range(dim):
            distances[i] += (x[:,j] - means[i][j])**2
            
    temp_classes = np.argmin(distances,axis=0)
    
    if np.all(temp_classes==classes):
        break
    else:
        classes = temp_classes
        for i in range(k):
            means[i] = np.average(x[classes==i], axis=0)    
        
    for i in range(k):
        means[i] = np.average(x[classes==i], axis=0)
        
        plt.plot(x[classes==i][:,0],x[classes==i][:,1], colors[i]+'o',alpha=0.15 ) 
        plt.plot(means[i][0],means[i][1], colors[i]+'x' ,markersize=10)

    plt.savefig('kmeans-'+str(n))
    n +=1
    plt.show()
    