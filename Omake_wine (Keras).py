# -*- coding: utf-8 -*-
# =============================================================================
# 1. Title: Wine Quality 
# 
# 2. Sources
#    Created by: Paulo Cortez (Univ. Minho), Antonio Cerdeira, Fernando Almeida, Telmo Matos and Jose Reis (CVRVV) @ 2009
#    
# 3. Past Usage:
# 
#   P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. 
#   Modeling wine preferences by data mining from physicochemical properties.
#   In Decision Support Systems, Elsevier, 47(4):547-553. ISSN: 0167-9236.
# 
#   In the above reference, two datasets were created, using red and white wine samples.
#   The inputs include objective tests (e.g. PH values) and the output is based on sensory data
#   (median of at least 3 evaluations made by wine experts). Each expert graded the wine quality 
#   between 0 (very bad) and 10 (very excellent). Several data mining methods were applied to model
#   these datasets under a regression approach. The support vector machine model achieved the
#   best results. Several metrics were computed: MAD, confusion matrix for a fixed error tolerance (T),
#   etc. Also, we plot the relative importances of the input variables (as measured by a sensitivity
#   analysis procedure).
#  
# 4. Relevant Information:
# 
#    The two datasets are related to red and white variants of the Portuguese "Vinho Verde" wine.
#    For more details, consult: http://www.vinhoverde.pt/en/ or the reference [Cortez et al., 2009].
#    Due to privacy and logistic issues, only physicochemical (inputs) and sensory (the output) variables 
#    are available (e.g. there is no data about grape types, wine brand, wine selling price, etc.).
# 
#    These datasets can be viewed as classification or regression tasks.
#    The classes are ordered and not balanced (e.g. there are munch more normal wines than
#    excellent or poor ones). Outlier detection algorithms could be used to detect the few excellent
#    or poor wines. Also, we are not sure if all input variables are relevant. So
#    it could be interesting to test feature selection methods. 
# 
# 5. Number of Instances: red wine - 1599; white wine - 4898. 
# 
# 6. Number of Attributes: 11 + output attribute
#   
#    Note: several of the attributes may be correlated, thus it makes sense to apply some sort of
#    feature selection.
# 
# 7. Attribute information:
# 
#    For more information, read [Cortez et al., 2009].
# 
#    Input variables (based on physicochemical tests):
#    1 - fixed acidity
#    2 - volatile acidity
#    3 - citric acid
#    4 - residual sugar
#    5 - chlorides
#    6 - free sulfur dioxide
#    7 - total sulfur dioxide
#    8 - density
#    9 - pH
#    10 - sulphates
#    11 - alcohol
#    Output variable (based on sensory data): 
#    12 - quality (score between 0 and 10)
# 
# =============================================================================
#モジュール読み出し
#import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv

#モジュール読み出し
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense, Dropout

##データの読み出し
#テキストデータに変換
# =============================================================================
# txt_data = open("testdata/winequality-red.txt","r")
# array=[]
# df=[]
# 
# line=[]
# for line in txt_data:
#     line=line.strip()#改行を消去
#     line=line.split()#タブで区切る
#     line = np.array(line[1:], dtype=np.float)#strをfloatに変換
#     df.append(line)
# 
# txt_data.close()
# df=df[1:]
# dh=[]
# 
# print(df[:20])

#df = np.array(df)
# =============================================================================
with open("testdata/winequality-red.txt", 'r') as f:
     wines = list(csv.reader(f, delimiter="\t"))
     array= np.array(wines[1:], dtype=np.float)
#print(array[:3],type(array))

# #array = array.T#転置する
#データリスト
title=["1#fixed_acidity",
       "2#volatile_acidity",
       "3#citric_acid",
       "4#residual_sugar",
       "5#chlorides",
       "6#free_sulfur_dioxide",
       "7#total_sulfur_dioxide",
       "8#density",
       "9#pH",
       "10#sulphates",
       "11#alcohol",
       "11#B",
       "12#quality"]

print('input_data_number:',len(array))

# =============================================================================
# 1#fixed_acidity
# 2#volatile_acidity
# 3#citric_acid
# 4#residual_sugar
# 5#chlorides
# 6#free_sulfur_dioxide
# 7#total_sulfur_dioxide
# 8#density
# 9#pH
# 10#sulphates
# 11#alcohol
#     Output variable (based on sensory data): 
# 12#quality (score between 0 and 10)
# =============================================================================
#予想する目的変数の選択
T=12#quality
AA=title[T]


#Xは環境変数（0-T-1, (T+1)-21）, Y(目的変数)は5のインデックス
X = []
Y = []
a = []
    
for i in range(len(array)):
    a = array[i]
    a = a.tolist()#type'ndarryをlistにする'tolistを用いると既存のndarrayをリストへ変換
    b = a[1:T-1]
    b1 = a[T+1:21]
    b[len(b):len(b)] = b1
    c = a[T]
    X.append(b)
    Y.append(c)

Y = np.array(Y)#listをndarrayに変換

# データの標準化
#print(X)  
X = preprocessing.scale(X)
#print(X)
 
# ラベルをone-hot-encoding形式に変換
#print(Y)
Y = np_utils.to_categorical(Y)
#print(Y,type(Y))

# 訓練データとテストデータに分割
 ##inputパラメータはtest_size
X_train,X_test,Y_train,Y_test =train_test_split(X,Y,test_size=0.3)
 
# 訓練データとテストデータのshapeをチェック
#print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)  
#(1119, 10) (480, 10) (1119, 9) (480, 9)出力層9，入力 input_shapeを10

#Y_train=list(Y_train)
#入力データのshapeをチェック
D1 = X_train.shape[1]
#print(type(Y_train.shape))
D2 = Y_train.shape[1]
print('input_index:',D1)
print('out_index:',D2)
 
# モデル構築
def build_model():
    model = Sequential()
    model.add(Dense(128,activation='relu',input_shape=(D1,)))
    model.add(Dropout(0.25))
    model.add(Dense(64,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(D2,activation='softmax'))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
##Kerasのオプティマイザの共通パラメータ
#SGD(確率的勾配降下法オプティマイザ)
#RMSprop(RMSPropオプティマイザ)
#Adagrad(Adagradオプティマイザ)
#Adadelta(Adadeltaオプティマイザ)
#Adam(Adamオプティマイザ)

#K-Fold検証法(https://manareki.com/k_fold)
kf = KFold(n_splits=3, shuffle=True)#n_splitsは分割回数(３分割)
all_loss=[]
all_val_loss=[]
all_acc=[]
all_val_acc=[]
ep=50

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
predict_classes = model.predict(X_test, batch_size=32)
predict_classes = np.argmax(predict_classes,1)
true_classes = np.argmax(Y_test,1)
matx=confusion_matrix(true_classes, predict_classes)

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
