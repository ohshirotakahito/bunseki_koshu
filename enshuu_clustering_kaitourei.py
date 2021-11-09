# -*- coding: utf-8 -*-

"""
演習課題
1.次のプログラムを実行してください。
2.i)KMeans法を用いてtestdata中にあるCluster2.csVを3クラスターに分けてください。
  ii)結果を評価してください
"""

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os

#演習課題1
file = os.path.join(os.getcwd(),'testdata','Cluster1.csv')
x = np.genfromtxt(file,delimiter=',')#csvファイルのデータ区切りを指定

#読み込んだデータの確認
plt.plot(x[:,0],x[:,1],'bo')#'bo'は'b'が青色,'o'が丸いマーカーの散布図
plt.show()
#2次元ヒストグラム
plt.hist2d(x[:,0],x[:,1],bins=[20,20])
plt.show()

cluster = KMeans(n_clusters=3 )#クラスター数は3と指定

#学習
cluster.fit(x)#データを学習
y = cluster.predict(x)#データがどのクラスターに属するか予測
#上2行はy=cluster.fit_predict(x)と書いてもよい

#結果の出力
print(y)
for i in np.unique(y):
    plt.plot(x[:,0][y==i] ,x[:,1][y==i] ,'o')
plt.show()


#演習課題2
#↓↓↓↓↓↓↓↓↓プログラムを記入
file = os.path.join(os.getcwd(),'testdata','Cluster2.csv')
x = np.genfromtxt(file,delimiter=',')#csvファイルのデータ区切りを指定

#読み込んだデータの確認
plt.plot(x[:,0],x[:,1],'bo')#'bo'は'b'が青色,'o'が丸いマーカーの散布図
plt.show()

x_scaled = (x - np.average(x,axis=0) ) / np.std(x,axis=0)

# =============================================================================
# 
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# x_scaled = scaler.fit_transform(x)
# print(x_scaled)
# 
# plt.plot(x_scaled[:,0],x_scaled[:,1],'ro')#'ro'は'r'が赤色
# plt.show()
# =============================================================================

cluster = KMeans(n_clusters=3 )#クラスター数は3と指定
cluster.fit(x_scaled)#データを学習
y = cluster.predict(x_scaled)#データがどのクラスターに属するか予測



#学習
# =============================================================================
# cluster.fit(x)#データを学習
# y = cluster.predict(x)#データがどのクラスターに属するか予測
# =============================================================================


#結果の出力
print(y)
for i in np.unique(y):
    plt.plot(x[:,0][y==i] ,x[:,1][y==i] ,'o')
plt.show()


