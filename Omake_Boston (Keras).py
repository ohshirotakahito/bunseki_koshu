# -*- coding: utf-8 -*-
# =============================================================================
# 1. Title: Boston Housing Data
# 
# 2. Sources:
#    (a) Origin:  This dataset was taken from the StatLib library which is
#                 maintained at Carnegie Mellon University.
#    (b) Creator:  Harrison, D. and Rubinfeld, D.L. 'Hedonic prices and the 
#                  demand for clean air', J. Environ. Economics & Management,
#                  vol.5, 81-102, 1978.
#    (c) Date: July 7, 1993
# 
# 3. Past Usage:
#    -   Used in Belsley, Kuh & Welsch, 'Regression diagnostics ...', Wiley, 
#        1980.   N.B. Various transformations are used in the table on
#        pages 244-261.
#     -  Quinlan,R. (1993). Combining Instance-Based and Model-Based Learning.
#        In Proceedings on the Tenth International Conference of Machine 
#        Learning, 236-243, University of Massachusetts, Amherst. Morgan
# 
# =============================================================================
#モジュール読み出し
#import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#モジュール読み出し
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

##データの読み出し
#テキストデータに変換
txt_data = open("testdata/housing.txt","r")
array=[]
df=[]

line=[]
for line in txt_data:
    line=line.strip()#改行を消去
    line=line.split()#タブで区切る
    line = [float(s) for s in line]#strをfloatに変換
    df.append(line)
txt_data.close()

array = np.array(df)#NumPy形式に変換

#print(array[:3],type(array)
#array = array.T#転置する

#データリスト
title=["0#CRIM",
       "1#ZN",
       "2#INDUS",
       "3#CHAS",
       "4#NOX",
       "5#RM",
       "6#AGE",
       "7#DIS",
       "8#RAD",
       "9#TAX",
       "10#PTRATIO",
       "11#B",
       "12#LSTAT",
       "13#MEDV"]

print('input_data_number:',len(array))

# =============================================================================
# 0#CRIM      per capita crime rate by town
# 1#ZN        proportion of residential land zoned for lots over 
#                  25,000 sq.ft.
# 2#INDUS     proportion of non-retail business acres per town
# 3#CHAS      Charles River dummy variable (= 1 if tract bounds 
#                  river; 0 otherwise)
# 4#NOX       nitric oxides concentration (parts per 10 million)
# 5#RM        average number of rooms per dwelling
# 6#AGE       proportion of owner-occupied units built prior to 1940
# 7#DIS       weighted distances to five Boston employment centres
# 8#RAD       index of accessibility to radial highways
# 9#TAX      full-value property-tax rate per $10,000
# 10#PTRATIO  pupil-teacher ratio by town
# 11#B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks 
#              by town
# 12#LSTAT    % lower status of the population
# 13#MEDV     Median value of owner-occupied homes in $1000's
# =============================================================================

#予想する目的変数の選択(ここを0～13までかえる)
T=2#INDUS
AA=title[T]


#Xは環境変数（0-T-1, (T+1)-21）, Y(目的変数)は5のインデックス
X = []
Y = []
a = []
    
for i in range(len(array)):
    a = array[i]
    a = a.tolist()#type'ndarryをlistにする'tolistを用いると既存のndarrayをリストへ変換
    b = a[0:T-1]
    b1 = a[T+1:21]
    b[len(b):len(b)] = b1
    c = a[T]
    X.append(b)
    Y.append(c)

Y = np.array(Y)#listをndarrayに変換

# データの標準化   
X = preprocessing.scale(X)
 
# ラベルをone-hot-encoding形式に変換
#Y = np_utils.to_categorical(Y)
#print(Y)

# 訓練データとテストデータに分割
 ##inputパラメータはtest_size
X_train,X_test,Y_train,Y_test =train_test_split(X,Y,test_size=0.3)
 
# 訓練データとテストデータのshapeをチェック
#print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)  
#(354, 12) (152, 12) (354,) (152,))出力層3，入力 input_shapeを12

#入力データのshapeをチェック
D1 = X_train.shape[1]
print('input_index:',D1)
 
# モニューラルネットワークモデルの構築
model = Sequential()
model.add(Dense(1000, activation = 'relu'))
model.add(Dense(800, activation = 'relu'))
model.add(Dense(100, activation =  'relu'))
model.add(Dense(1))

# モデルをコンパイル 
model.compile(Adam(lr=1e-3), loss="mean_squared_error")
#Kerasのオプティマイザの共通パラメータ
#SGD(確率的勾配降下法オプティマイザ),RMSprop(RMSPropオプティマイザ)
#Adagrad(Adagradオプティマイザ),Adadelta(Adadeltaオプティマイザ)
#Adam(Adamオプティマイザ)

#トレーニングデータで学習し，テストデータで評価（平均2乗誤差を用いる）
history = model.fit(X_train, Y_train, batch_size=128, epochs=100, verbose=1, 
          validation_data=(X_test, Y_test))

#print(model.evaluate(X_test, Y_test))

#指標の履歴表示（loss） 
plt.plot(history.history['loss'])
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()

#テストデータからの予測値および正解
Y_pred = model.predict(X_test).flatten()
compare = pd.DataFrame(np.array([Y_test,Y_pred]).T)
compare.columns = ['正解','予測値']
print(compare[:20])

# 散布図を描画
plt.scatter(Y_test, Y_pred)
plt.title(AA)
plt.ylabel('true')
plt.xlabel('predict')
plt.show()
