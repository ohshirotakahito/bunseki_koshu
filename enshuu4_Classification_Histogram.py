# -*- coding: utf-8 -*-
"""
enshuu_Classificationで用いたデータを解析するためのプログラム

"""
import os
import numpy as np
import matplotlib.pyplot as plt

file_train = os.path.join(os.getcwd(),'testdata','Clf1_Train.csv')
file_test = os.path.join(os.getcwd(),'testdata','Clf1_Test.csv')
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

mols=['A','B']

for i in range(x_train.shape[1] ):
    xrange= [np.min(x_train[:,i] ), np.max(x_train[:,i] )]
    for mol in mols:
        x = x_train[:,i][y_train==mol] # A,Bのxを順にとりだし
        plt.hist(x, bins= 100, range=xrange,label=mol,alpha=0.5,
                 density=True) # density=Trueで規格化
    plt.legend()
    plt.xlabel('Feature '+ str(i))       
    plt.ylabel('Count') 
    plt.show()
    






