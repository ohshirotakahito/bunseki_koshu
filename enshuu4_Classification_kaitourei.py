# -*- coding: utf-8 -*-
"""
以下の課題のcsvファイルでは0-4行目が特徴量、5行目が正解のラベルが書いてあります。

演習課題1
1.以下のプログラムはSupport Vector Machineを用いて分類を行うプログラムです。
　　プログラムを実行してください。

2.RandomForestを用いてtestdata中にあるClf2_Train.csvを学習して
　　Clf2_Test.csvのデータを分類して、F値を評価してください。

"""
print(__doc__)

import os 
import numpy as np
from ConfusionMatrix import plot_confusion_matrix 
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier


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
# sklearnのsvmの中のSVCを読み込み
from sklearn.svm import SVC

#機械学習分類器の設定
clf = SVC()#何も指定がなければデフォルトのパラメータで設定される

#学習
clf.fit(x_train,y_train)#この後にデータを学習した分類器で分類ができるようになる。

#分類
y_pred=clf.predict(x_test)#学習した分類器でx_testを一つ一つ分類してy_predに入れる

#結果の確認
#混同行列の表示
plot_confusion_matrix(y_test,y_pred)

#f値の表示
print('F_measure',metrics.f1_score(y_test, y_pred , average='weighted') )


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


num_A = np.sum(y_train=='A' ) #　y_train中のAの個数
num_B = np.sum(y_train=='B' ) #　y_train中のBの個数

print('Aのデータ数',num_A)
print('Bのデータ数',num_B)
min_num = np.min([num_A,num_B]) # num_A,num_Bの最小値
## 学習データの数をそろえる
## 今回はシグナル数が少ないものにそろえる(undersample)
indice_A = np.arange(len(x_train) )[y_train=='A' ] # y_train中のAの要素を持つインデックス
indice_A_undersample = np.random.permutation(indice_A )[:min_num] #ランダムに並び替えて先頭500個（num_A,num_Bの最小値）を取り出す

indice_B = np.arange(len(x_train) )[y_train=='B' ] # y_train中のBの要素を持つインデックス

indice_undersample = np.append(indice_A_undersample,indice_B ) # undersampleした後のインデックス

x_train_undersample = x_train[indice_undersample ] # undersampleした後のx_train
y_train_undersample = y_train[indice_undersample ] # undersampleした後のy_train

# =============================================================================
# #実用的にはアンダーサンプルにはこちらを使う
# from imblearn.under_sampling import RandomUnderSampler 
# rus = RandomUnderSampler()
# x_train_undersample,y_train_undersample = rus.fit_resample(x_train,y_train)
# =============================================================================

clf = RandomForestClassifier()
#clf.fit(x_train,y_train)
clf.fit(x_train_undersample,y_train_undersample)
y_pred=clf.predict(x_test)
plot_confusion_matrix(y_test,y_pred)
print('F_measure',metrics.f1_score(y_test, y_pred , average='weighted') )









