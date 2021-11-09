# -*- coding: utf-8 -*-
"""
講習5を参考にして以下の課題を行ってください
以下の課題のcsvファイルでは0-4行目が特徴量、5行目が正解のラベルが書いてあります。

演習課題2
RandomForestを用いてtestdata中にあるClf2_Train.csVを学習して
Clf2_Test.csVのデータを分類して、F値を評価してください。

演習課題3
k最近傍法を用いてtestdata中にあるClf3_Train.csVを学習して
Clf3_Test.csVのデータを分類して、F値を評価してください。


演習課題２と３の結果を比較・考察してください。

"""
print(__doc__)

import os 
import numpy as np
from ConfusionMatrix import plot_confusion_matrix 
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt

print("演習課題2")

file_train = os.path.join(os.getcwd(),'testdata','Clf2_Train.csv')
file_test = os.path.join(os.getcwd(),'testdata','Clf2_Test.csv')
#特徴量の読み込み
x_train = np.genfromtxt(file_train,usecols = range(5),#csvファイルの0-5行目の読み込み
                        delimiter=',')#csvファイルのデータ区切りを指定
#ラベルの読み込み
y_train = np.genfromtxt(file_train,usecols = 5,#5行目の読み込み
                        dtype = 'U', #文字として読み込む
                        delimiter=',')

#テスト用データ
#特徴量の読み込み
x_test = np.genfromtxt(file_test,usecols = range(5),#csvファイルの0-5行目の読み込み
                        delimiter=',')#csvファイルのデータ区切りを指定
#ラベルの読み込み
y_test = np.genfromtxt(file_test,usecols = 5,#5行目の読み込み
                        dtype = 'U', #文字列(ユニコード)として読み込む
                        delimiter=',')

plt.hist(x_train[y_train=='A'][:,1],color = 'r'
         ,range=(-2,8),bins=50 )
plt.show()
plt.hist(x_train[y_train=='B'][:,1],color = 'b'
         ,range=(-2,8),bins=50)
plt.show()

plt.plot(x_train[y_train=='A'][:,0] , x_train[y_train=='A'][:,1] ,
         'r.',label='Train_A')
plt.plot(x_train[y_train=='B'][:,0] , x_train[y_train=='B'][:,1] ,
         'b.',label='Train_B')

plt.legend()
plt.show()

print(np.sum(y_train=='A' ))
print(np.sum(y_train=='B' ))
print(np.sum(y_test=='A' ))
print(np.sum(y_test=='B' ))


# =============================================================================
# clf = RandomForestClassifier()
# clf.fit(x_train,y_train)
# y_pred=clf.predict(x_test)
# plot_confusion_matrix(y_test,y_pred)
# print('F_measure',metrics.f1_score(y_test, y_pred , average='weighted') )
# =============================================================================




print("=================================")
print("演習課題3")
file_train = os.path.join(os.getcwd(),'testdata','Clf3_Train.csv')
file_test = os.path.join(os.getcwd(),'testdata','Clf3_Test.csv')
#特徴量の読み込み
x_train = np.genfromtxt(file_train,usecols = range(5),#csvファイルの0-5行目の読み込み
                        delimiter=',')#csvファイルのデータ区切りを指定
#ラベルの読み込み
y_train = np.genfromtxt(file_train,usecols = 5,#5行目の読み込み
                        dtype = 'U', #文字として読み込む
                        delimiter=',')

#テスト用データ
#特徴量の読み込み
x_test = np.genfromtxt(file_test,usecols = range(5),#csvファイルの0-5行目の読み込み
                        delimiter=',')#csvファイルのデータ区切りを指定
#ラベルの読み込み
y_test = np.genfromtxt(file_test,usecols = 5,#5行目の読み込み
                        dtype = 'U', #文字列(ユニコード)として読み込む
                        delimiter=',')

plt.hist(x_train[y_train=='A'][:,0],color = 'r'
         ,range=(0,0.003),bins=50 )
plt.show()
plt.hist(x_train[y_train=='B'][:,0],color = 'b'
         ,range=(0,0.003),bins=50)
plt.show()

# =============================================================================
# x_train[:,0]=1000*x_train[:,0]
# x_test[:,0]=1000*x_test[:,0]
# =============================================================================

clf = RandomForestClassifier()
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
plot_confusion_matrix(y_test,y_pred)
print('F_measure',metrics.f1_score(y_test, y_pred , average='weighted') )

print(np.sum(y_train=='A' ))
print(np.sum(y_train=='B' ))
print(np.sum(y_test=='A' ))
print(np.sum(y_test=='B' ))
plt.plot(x_train[y_train=='A'][:,0] , x_train[y_train=='A'][:,1] ,
         'ro',label='Train_A')
plt.plot(x_train[y_train=='B'][:,0] , x_train[y_train=='B'][:,1] ,
         'bo',label='Train_B')








