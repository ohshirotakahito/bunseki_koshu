#参考ページ
#http://aidiary.hatenablog.com/category/Keras?page=1478696865

#モジュール読み出し
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import np_utils
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix

#Sequentialモデルのレイヤーの構築"""多層パーセプトロンモデルを構築"""
def build_multilayer_perceptron():
    model = Sequential()
    model.add(Dense(16, input_shape=(4, )))
    model.add(Activation('relu'))
    model.add(Dense(3))
    model.add(Activation('softmax'))
    return model

if __name__ == "__main__":
    # Irisデータをロード
    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target

    # データの標準化
    X = preprocessing.scale(X)

    # ラベルをone-hot-encoding形式に変換
    # 0 => [1, 0, 0]
    # 1 => [0, 1, 0]
    # 2 => [0, 0, 1]
    Y = np_utils.to_categorical(Y)

    # 訓練データとテストデータに分割
    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, train_size=0.8)
    print(train_X.shape, test_X.shape, train_Y.shape, test_Y.shape)

    # モデル構築
    model = build_multilayer_perceptron()
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # モデル訓練
    history=model.fit(train_X, train_Y, epochs=50, batch_size=1, verbose=1)

    # モデル評価
    loss, accuracy = model.evaluate(test_X, test_Y, verbose=0)
    print("Accuracy = {:.2f}".format(accuracy))

predict_classes = model.predict(test_X, batch_size=32)
predict_classes = np.argmax(predict_classes,1)
true_classes = np.argmax(test_Y,1)
matx=confusion_matrix(true_classes, predict_classes)
  
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
fig, ax = plt.subplots(figsize=(5, 4)) # 混合行列のカラムの大きさ設定
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
fig, ax = plt.subplots(figsize=(5, 4)) # 混合行列のカラムの大きさ設定
sns.heatmap(MX2, annot=True, fmt="1.3")# fmtでデータの表示桁数
ax.set_ylim(len(matx_s), 0)# 混合行列の軸の下限を設定し，値がみえるようにする（バグ）
ax.set_title('Normalized Confusion matrix')
