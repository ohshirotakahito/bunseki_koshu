# -*- coding: utf-8 -*-
"""
講習5を参考にして以下の課題を行ってください
以下の課題のcsvファイルでは0-4行目が特徴量、5行目が正解のラベルが書いてあります。

演習課題2
RandomForestを用いてtestdata中にあるClf2_Train.csVを学習して
Clf2_Test.csVのデータを分類して、F値を評価してください。

"""
print(__doc__)

import os 
import numpy as np
from ConfusionMatrix import plot_confusion_matrix 
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

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



clf = RandomForestClassifier()
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
plot_confusion_matrix(y_test,y_pred)
print('F_measure',metrics.f1_score(y_test, y_pred , average='weighted') )









