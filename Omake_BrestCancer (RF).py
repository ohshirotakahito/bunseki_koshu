#!/usr/bin/python
# coding: UTF-8

#モジュール読み出し
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import datasets
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier


# sklearnのbrest_cancerデータの読み込み
breast = datasets.load_breast_cancer()

X=breast.data
Y=breast.target

print(X)


# データの標準化
X = preprocessing.scale(X)

# ラベルをone-hot-encoding形式に変換
#Y = np_utils.to_categorical(Y)


# 訓練データとテストデータに分割
X_train,X_test,Y_train,Y_test =train_test_split(X,Y,test_size=0.2)

# 訓練データとテストデータのshapeをチェック
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
#(455, 30) (114, 30) (455, 2) (114, 2)出力層2，入力 input_shapreを30   

# モデル構築
model =  RandomForestClassifier(random_state=0,n_estimators=100)   

# モデル学習
model.fit(X_train, Y_train) 

#モデルをもちいた予想
predict_classes = model.predict(X_test)

true_classes = Y_test
matx=confusion_matrix(true_classes, predict_classes)

# 予測結果と、正解（本当の答え）がどのくらい合っていたかを表す混合行列の標準化
x_sm=(sum(matx[0]))
y_sm=(sum(matx[1]))
x_arry=matx[0]/x_sm
y_arry=matx[1]/y_sm

#混同行列（％）のデータとカラムラベル挿入
n_matx=x_arry,y_arry
MX2=pd.DataFrame(n_matx, index=[u'predict_Non-cancer',
                          u'predict_Cancer'], columns=[u'read_Non Cancer', u'real_Cancer'])

#混同行列（数）のデータとカラムラベル挿入
MX1=pd.DataFrame(matx, index=[u'predict_Non-cancer',
                          u'predict_Cancer'], columns=[u'read_Non Cancer', u'real_Cancer'])

#モデルをもちいた予想の可視化
fig, ax = plt.subplots(figsize=(4, 3)) # 混合行列のカラムの大きさ設定
sns.heatmap(MX1, annot=True, fmt="d")
ax.set_ylim(len(matx), 0)# 混合行列の軸の下限を設定し，値がみえるようにする（バグ）
ax.set_title('Confusion_Matrix')

#モデルをもちいた予想の可視化
fig, ax = plt.subplots(figsize=(4, 3)) # 混合行列のカラムの大きさ設定
sns.heatmap(MX2, annot=True, fmt="1.3")# fmtでデータの表示桁数
ax.set_ylim(len(matx), 0)# 混合行列の軸の下限を設定し，値がみえるようにする（バグ）
ax.set_title('normalized confusion matrix')


