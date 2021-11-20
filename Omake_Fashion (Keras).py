##MNIST ファッション
#2つのタプル:
#x_train, x_test: shape (num_samples, 28, 28) の白黒画像データのuint8配列．
#y_train, y_test: shape (num_samples,) のカテゴリラベル(0-9の整数)のuint8配列．

#参考ページ
#https://qiita.com/Kuma_T/items/4449f008cad18fbb7f1a

#モジュール読み出し
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.metrics import confusion_matrix
from keras.utils import np_utils

##データの読み込み
fashion_mnist = tf.keras.datasets.fashion_mnist

#訓練データとテストデータに分割
(X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()
# (60000, 28, 28) 8×8サイズの数字データ

# 訓練データとテストデータのshapeをチェック
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
#x_train=sum(x_train)
#(60000, 28, 28) (10000, 28, 28) (60000,) (10000,)入力28x28　出力10
D1 = X_train.shape[1]
#D2 = Y_train.shape[1]
print('input_index:',D1)

#データインデックス
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# =============================================================================
# 0#Tシャツ/トップスT-shirt/top
# 1#ズボンTrouser
# 2#プルオーバーPullover
# 3#ドレス
# 4#コート
# 5#サンダル
# 6#シャツ
# 7#スニーカー
# 8#バッグ
# 9#アンクルブーツ
# =============================================================================


# データの可視化(一つのデータを可視化)
plt.imshow(X_train[0], cmap='gray')
plt.colorbar()
plt.savefig('X_train[0]')
plt.show()

#データの可視化(すべてのデータを可視化)
#plt.figure(figsize=(12,15))
#for i in range(25):
#    plt.subplot(5, 5, i+1)
#    plt.title("Label: " + str(i))
#    plt.imshow(x_train[i].reshape(28,28), cmap=None)


# データの正規化
X_train, X_test = X_train / 255.0, X_test / 255.0

#y_test = np_utils.to_categorical(y_test)
#y_train = np_utils.to_categorical(y_train)

#Sequentialモデルのレイヤーの構築
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),#2次元データの1次元データに平滑化
  tf.keras.layers.Dense(512, activation=tf.nn.relu),#二層目のHidden Layer活性化関数の指定
  tf.keras.layers.Dropout(0.2),#過学習を防ぐためのドロップアウト率指定
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)#活性化関数の指定
])

#"""訓練プロセスの定義"""
model.compile(optimizer='adam',##学習の最適化法を決定(勾配法)
              loss='sparse_categorical_crossentropy',#損失関数を決定
              metrics=['accuracy'])
##Kerasのオプティマイザの共通パラメータ
#SGD(確率的勾配降下法オプティマイザ)
#RMSprop(RMSPropオプティマイザ)
#Adagrad(Adagradオプティマイザ)
#Adadelta(Adadeltaオプティマイザ)
#Adam(Adamオプティマイザ)

ep=10
# 学習(訓練)の実行
history=model.fit(X_train,
                  Y_train,
                  epochs=ep,
                  batch_size=8)

# モデルの評価（損失値の計算）
model.evaluate(X_test,  Y_test, verbose=1)

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

# ラベルをone-hot-encoding形式に変換
Y_test = np_utils.to_categorical(Y_test)

#モデルをもちいた予想
predict_classes = model.predict(X_test, batch_size=32)
predict_classes = np.argmax(predict_classes,1)
true_classes = np.argmax(Y_test,1)
matx=confusion_matrix(true_classes, predict_classes)
#print(matx)

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
    kk=class_names[k]
    p1="predict_"+kk
    p2="real_"+kk
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
sns.heatmap(MX2, annot=True, fmt="1.2")# fmtでデータの表示桁数
ax.set_ylim(len(matx_s), 0)# 混合行列の軸の下限を設定し，値がみえるようにする（バグ）
ax.set_title('Normalized Confusion matrix')