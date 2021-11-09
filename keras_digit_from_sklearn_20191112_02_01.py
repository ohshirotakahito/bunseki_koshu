##MNIST 手書き数字
#参考ページ
#http://aidiary.hatenablog.com/category/Keras?page=1478696865

#モジュール読み出し
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing

from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D
from keras.optimizers import Adam
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.utils import plot_model
from keras.callbacks import TensorBoard
from keras.datasets import cifar10
from keras.utils import np_utils



#入力層が64ユニット、隠れ層が16ユニット、出力層が10ユニットの多層パーセプトロンを構築した

##データの読み込み
# sklearnのdigitデータの読み込み (1797, 8, 8) 8×8サイズの数字データ1797点
digits = datasets.load_digits()

##目的変数と観光変数の分離
X = digits.data#8×8のimageデータの平滑化した一次行列
Y = digits.target#ラベル
#Z=digits.images#8×8のimageデータの二次行列
print(X.shape)

#一次元行列から二次元行列への変換
#for j in range(10):
#    Z=np.reshape(X[j],(8,8))
#
#print(Z)

# =============================================================================
# # 数字データの平均化した値を画像で出力（可視化：＃１－１０のデータ）
# mean_images = np.zeros((10,8,8))
# fig = plt.figure(figsize=(10,5))
# for i in range(10):
#     mean_images[i] = W[Y==i].mean(axis=0)
#     ax = fig.add_subplot(2, 5, i+1)
#     ax.axis('off')
#     ax.set_title('train.{0} (n={1})'.format(i, len(W[Y==i])))
#     ax.imshow(mean_images[i],cmap=plt.cm.gray, interpolation='none')
# plt.show()
# =============================================================================

# データの正規化
X = preprocessing.scale(X)

# ラベルをone-hot-encoding形式に変換
Y = np_utils.to_categorical(Y)
# 0 => [1, 0, 0]
# 1 => [0, 1, 0]
# 2 => [0, 0, 1]
    #risのラベルは文字列だがsklearnのデータセットでは0, 1, 2のように数値ラベルに変換されている。
    #これをニューラルネットで扱いやすいone-hotエンコーディング型式に変換する。
    #one-hotエンコーディングは、特定のユニットのみ1でそれ以外は0のようなフォーマットのこと。
    #この変換は、keras.utils.np_utils の to_categorical() に実装されている

# 訓練データとテストデータに分割
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8)
#shapeは，numpyのクラスndarryの変数　ndarray.shape

# 訓練データとテストデータのshapeをチェック
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
#print(x_test[0])
#print(x_test[0].shape)
#(1437, 64) (360, 64) (1437, 10) (360, 10)

# データの可視化(一つのデータを可視化)
# =============================================================================
plt.imshow(X_train[0], cmap='gray')
plt.colorbar()
plt.savefig('X_train[0]')
plt.show()
# =============================================================================

#Sequentialモデルのレイヤーの構築"""多層パーセプトロンモデルを構築"""
model = Sequential()
model.add(Dense(16, input_shape=(64, )))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Dropout(0.25))
model.add(Activation('softmax'))
#入力層が64ユニット、隠れ層が16ユニット、出力層が10ユニットの多層パーセプトロンを構築した


# モデル構築（訓練プロセスの定義）
model.compile(optimizer='adam',##学習の最適化法を決定(勾配法)
                  loss='categorical_crossentropy',#損失関数(誤差関数)を決定
                  metrics=['accuracy'])
##Kerasのオプティマイザの共通パラメータ
#SGD(確率的勾配降下法オプティマイザ)
#RMSprop(RMSPropオプティマイザ)
#Adagrad(Adagradオプティマイザ)
#Adadelta(Adadeltaオプティマイザ)
#Adam(Adamオプティマイザ)

# モデル訓練
history=model.fit(X_train, Y_train, nb_epoch=5, batch_size=1, verbose=1)

# モデル評価
loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)
print("Accuracy = {:.2f}".format(accuracy))
    
plt.plot(history.history['loss'])
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()

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

#モデルをもちいた予想の可視化
##Confusion Matrixカラムラベル
idd=[]
cll=[]

#Confusion Matrixカラムラベル要素抽出
Z=(set(true_classes))#存在する要素
PM=list(Z)#setをlistに変換
#print(PM)

#Confusion Matrixカラムラベルリスト作成(wineだけ)
for i in range(len(PM)):
    k=PM[i]
    p1="predict_"+str(k)
    p2="real_"+str(k)
    #x=input_list(T)
    idd.append(p1)
    cll.append(p2)

#混同行列（数）のデータとカラムラベル挿入
MX1=pd.DataFrame(matx, index=idd, columns=cll)

#モデルをもちいた予想の可視化
fig, ax = plt.subplots(figsize=(7, 6)) # 混合行列のカラムの大きさ設定
sns.heatmap(MX1, annot=True, fmt="d")
ax.set_ylim(len(matx), 0)# 混合行列の軸の下限を設定し，値がみえるようにする（バグ）
ax.set_title('Confusion_Matrix')

##混合行列の正規化
matx_s=[]
cx=matx.shape[0]

for i in range(cx):
    ss=sum(matx[i])
    matx_i=matx[i]/ss
    matx_i = matx_i.tolist()
    matx_s.append(matx_i)
    
#混同行列（正規化）のデータとカラムラベル挿入
MX2=pd.DataFrame(matx_s, index=idd, columns=cll)

#モデルをもちいた予想の可視化
fig, ax = plt.subplots(figsize=(7, 6)) # 混合行列のカラムの大きさ設定
sns.heatmap(MX2, annot=True, fmt="1.3")# fmtでデータの表示桁数
ax.set_ylim(len(matx_s), 0)# 混合行列の軸の下限を設定し，値がみえるようにする（バグ）
ax.set_title('Normalized Confusion matrix')
