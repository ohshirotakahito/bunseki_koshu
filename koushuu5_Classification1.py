# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 11:12:25 2021

@author: komoto
"""

import os 
import numpy as np
from ConfusionMatrix import plot_confusion_matrix #同階層にあるConfusionMatrix.pyからplot_confusion_matrixをload
from sklearn import metrics

file_train = os.path.join(os.getcwd(),'testdata','Clf1_Train.csv')
file_test = os.path.join(os.getcwd(),'testdata','Clf1_Test.csv')

#データの読み込み
#学習用データ
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


#読み込んだデータの確認
print(x_train)
print(y_train)
print(x_test)
print(y_test)


#機械学習ライブラリのインポート
#sklearnのensembleの中にあるRandomForestClassifierをインポート
from sklearn.ensemble import RandomForestClassifier 

#機械学習分類器の設定
clf = RandomForestClassifier()#何も指定がなければデフォルトのパラメータで設定される

#学習
clf.fit(x_train,y_train)#この後にデータを学習した分類器で分類ができるようになる。

#分類
y_pred=clf.predict(x_test)#学習した分類器でx_testを一つ一つ分類してy_predに入れる

#結果の確認
#混同行列の表示
plot_confusion_matrix(y_test,y_pred)

#f値の表示
print('F_measure',metrics.f1_score(y_test, y_pred , average='weighted') )

"""
演習課題
サポートベクターマシン用いて上のx_train,y_trainを学習して
x_testを分類して、評価してください。
サポートベクターマシンは以下のようにインポートできます。
from sklearn.svm import SVC

"""

#from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
clf = SVC()
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
plot_confusion_matrix(y_test,y_pred)
print('F_measure',metrics.f1_score(y_test, y_pred , average='weighted') )







