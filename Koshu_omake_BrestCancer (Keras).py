#!/usr/bin/python
# coding: UTF-8

#モジュール読み出し
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import datasets
from sklearn import preprocessing

from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense, Dropout

# sklearnのbrest_cancerデータの読み込み
breast = datasets.load_breast_cancer()

X=breast.data
Y=breast.target

print(X)


# データの標準化
X = preprocessing.scale(X)

# ラベルをone-hot-encoding形式に変換
Y = np_utils.to_categorical(Y)


# 訓練データとテストデータに分割
X_train,X_test,Y_train,Y_test =train_test_split(X,Y,test_size=0.2)

# 訓練データとテストデータのshapeをチェック
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
#(455, 30) (114, 30) (455, 2) (114, 2)出力層2，入力 input_shapreを30   

# モデル構築
def build_model():
    model = Sequential()
    model.add(Dense(16,activation='relu',input_shape=(30,)))
    model.add(Dropout(0.2))
    model.add(Dense(8,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(2,activation='softmax'))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

#K-Fold検証法(https://manareki.com/k_fold)
kf = KFold(n_splits=3, shuffle=True)#n_splitsは分割回数(３分割)
all_loss=[]
all_val_loss=[]
all_acc=[]
all_val_acc=[]
ep=10

for train_index, val_index in kf.split(X_train,Y_train):

    train_data=X_train[train_index]
    train_label=Y_train[train_index]
    val_data=X_train[val_index]
    val_label=Y_train[val_index]

    model=build_model()
    history=model.fit(train_data,
                      train_label,
                      epochs=ep,
                      batch_size=8,
                      validation_data=(val_data,val_label))

    loss=history.history['loss']
    val_loss=history.history['val_loss']
    acc=history.history['accuracy']
    val_acc=history.history['val_accuracy']

    all_loss.append(loss)
    all_val_loss.append(val_loss)
    all_acc.append(acc)
    all_val_acc.append(val_acc)
    
# =============================================================================
#     #モデルをもちいた予想
#     predict_classes = model.predict_classes(train_data, batch_size=32)
#     true_classes = np.argmax(train_label,1)
#     matx=confusion_matrix(true_classes, predict_classes)
#     
#     #混同行列（数）のデータとカラムラベル挿入
#     MX1=pd.DataFrame(matx, index=[u'predict_Non-cancer',
#                           u'predict_Cancer'], columns=[u'read_Non Cancer', u'real_Cancer'])
#     
#     #モデルをもちいた予想の可視化
#     fig, ax = plt.subplots(figsize=(4, 3)) # 混合行列のカラムの大きさ設定
#     sns.heatmap(MX1, annot=True, fmt="d")
#     ax.set_ylim(len(matx), 0)# 混合行列の軸の下限を設定し，値がみえるようにする（バグ）
#     ax.set_title('Confusion_Matrix')
# 
# =============================================================================

ave_all_loss=[
    np.mean([x[i] for x in all_loss]) for i in range(ep)]
ave_all_val_loss=[
    np.mean([x[i] for x in all_val_loss]) for i in range(ep)]
ave_all_acc=[
    np.mean([x[i] for x in all_acc]) for i in range(ep)]
ave_all_val_acc=[
    np.mean([x[i] for x in all_val_acc]) for i in range(ep)]


#指標の履歴表示（loss）    
plt.plot(history.history['loss'])
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()

#指標の履歴表示（accuracy）
plt.plot(history.history['accuracy'])
plt.title('accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()

#モデルをもちいた予想
predict_classes = model.predict_classes(X_test, batch_size=32)

true_classes = np.argmax(Y_test,1)
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


