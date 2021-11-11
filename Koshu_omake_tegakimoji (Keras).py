#参考ページ
#http://aidiary.hatenablog.com/category/Keras?page=1478696865

#モジュール読み出し
import cv2
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
from tensorflow.python.keras.models import load_model

#Sequentialモデルのレイヤーの構築"""多層パーセプトロンモデルを構築"""
def build_multilayer_perceptron():
    model = Sequential()
    model.add(Dense(16, input_shape=(64, )))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Dropout(0.25))
    model.add(Activation('softmax'))
    return model

#入力層が64ユニット、隠れ層が16ユニット、出力層が10ユニットの多層パーセプトロンを構築した

##データの読み込み
# sklearnのdigitデータの読み込み (1797, 8, 8) 8×8サイズの数字データ1797点
digits = datasets.load_digits()
X=digits.data#8×8のimageデータの平滑化した一次行列
Y=digits.target#ラベル
Z=digits.images#8×8のimageデータの二次行列
print(X.shape)

#一次元行列から二次元行列への変換
#for j in range(10):
#    Z=np.reshape(X[j],(8,8))
#
#print(Z)

# =============================================================================
# # 数字データの平均化した値を画像で出力（可視化：＃１－１０のデータ）
#mean_images = np.zeros((10,8,8))
#fig = plt.figure(figsize=(10,5))
#for i in range(10):
#     mean_images[i] = W[Y==i].mean(axis=0)
#     ax = fig.add_subplot(2, 5, i+1)
#     ax.axis('off')
#     ax.set_title('train.{0} (n={1})'.format(i, len(W[Y==i])))
#     ax.imshow(mean_images[i],cmap=plt.cm.gray, interpolation='none')
#plt.show()
# =============================================================================

# データの標準化
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
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8)
         #shapeは，numpyのクラスndarryの変数　ndarray.shape

# 訓練データとテストデータのshapeをチェック
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
#print(x_test[0])
#print(x_test[0].shape)

# データの可視化(一つのデータを可視化)
#plt.imshow(x_train[0], cmap='gray')
#plt.colorbar()
#plt.savefig('x_train[0]')
#plt.show()

# モデル構築（訓練プロセスの定義）
model = build_multilayer_perceptron()#上記のdefより
model.compile(optimizer='adam',##学習の最適化法を決定(勾配法)
                  loss='categorical_crossentropy',#損失関数(誤差関数)を決定
                  metrics=['accuracy'])

# モデル訓練
history=model.fit(x_train, y_train, epochs=5, batch_size=1, verbose=1)

# モデルの保存
model.save('test.h5')

# モデル評価
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
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
predict_classes = model.predict_classes(x_test, batch_size=32)
true_classes = np.argmax(y_test,1)
w=confusion_matrix(true_classes, predict_classes)

#モデルをもちいた予想の可視化
fig, ax = plt.subplots(figsize=(4, 3)) # 混合行列のカラムの大きさ設定
sns.heatmap(w, annot=True, fmt="d")
ax.set_ylim(len(w), 0)# 混合行列の軸の下限を設定し，値がみえるようにする（バグ）
ax.set_title('Confusion_Matrix_')

model = load_model('test.h5')
model.summary()
 
def predict_digit(filename):
    #画像を読み込む
    my_img = cv2.imread(filename)
    plt.imshow(my_img, cmap='gray')
    plt.show()
    #画像を膨張させる
    #my_img = cv2.dilate(my_img,kernel,iterations = 1)
    #plt.imshow(my_img, cmap='gray')
    #plt.show()
    #グレースケールに変換する
    my_img = cv2.cvtColor(my_img, cv2.COLOR_BGR2GRAY)
    # 16 * 16のサイズに変換する
    my_img = cv2.resize(my_img,(32,32))
    plt.imshow(my_img, cmap='gray')
    plt.show()
    # 16 * 16のサイズに変換する
    my_img = cv2.resize(my_img,(16,16))
    plt.imshow(my_img, cmap='gray')
    plt.show()
    # 8 * 8のサイズに変換する
    my_img = cv2.resize(my_img,(8,8))
    plt.imshow(my_img, cmap='gray')
    plt.show()
    #白黒反転する
    my_img = 15 - my_img // 16
    plt.imshow(my_img, cmap='gray')
    plt.show()
    #二次元を一次元に変換
    my_img = my_img.reshape((-1,64))
    model.predict_classes(my_img)
    res=model.predict_classes(my_img)
    print(model.predict_classes(my_img))
    return res[0]

n = predict_digit("eight.png")
print("eight.png = " + str(n))