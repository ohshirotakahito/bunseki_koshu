#!/usr/bin/python
# coding: UTF-8
# =============================================================================
# This dataset is based on "Bank Marketing" UCI dataset (please check the description at: http://archive.ics.uci.edu/ml/datasets/Bank+Marketing).
#    The data is enriched by the addition of five new social and economic features/attributes (national wide indicators from a ~10M population country), published by the Banco de Portugal and publicly available at: https://www.bportugal.pt/estatisticasweb.
#    This dataset is almost identical to the one used in [Moro et al., 2014] (it does not include all attributes due to privacy concerns). 
#    Using the rminer package and R tool (http://cran.r-project.org/web/packages/rminer/), we found that the addition of the five new social and economic attributes (made available here) lead to substantial improvement in the prediction of a success, even when the duration of the call is not included. Note: the file can be read in R using: d=read.table("bank-additional-full.csv",header=TRUE,sep=";")
#    
#    The zip file includes two datasets: 
#       1) bank-additional-full.csv with all examples, ordered by date (from May 2008 to November 2010).
#       2) bank-additional.csv with 10% of the examples (4119), randomly selected from bank-additional-full.csv.
#    The smallest dataset is provided to test more computationally demanding machine learning algorithms (e.g., SVM).
# 
#    The binary classification goal is to predict if the client will subscribe a bank term deposit (variable y).
# 
# =============================================================================
#モジュール読み出し
#import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#モジュール読み出し
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense, Dropout

#データの読み出し
df = pd.read_csv("testdata/bank-additional.txt")
ac=df.columns#カラム
ai=df.index#インデックス

array = df.values
array = array.T

#データリスト
age=[]
job=["admin.",
     "blue-collar",
     "entrepreneur",
     "housemaid",
     "management",
     "retired",
     "self-employed",
     "services",
     "student",
     "technician",
     "unemployed",
     "unknown"]
marital=["divorced",
            "married",
            "single",
            "unknown"]
education=["basic.4y",
           "basic.6y",
           "basic.9y",
           "high.school",
           "illiterate",
           "professional.course",
           "university.degree",
           "unknown"]
default=["no",
         "yes",
         "unknown"]
housing=["no",
         "yes",
         "unknown"]
loan=["no",
         "yes",
         "unknown"]
contact=["cellular",
         "telephone"]
month=["jan", 
       "feb", 
       "mar", 
       "apr",
       "may",
       "jun",
       "jul",
       "aug",
       "sep",
       "oct",
       "nov",
       "dec"]
day_of_week=["mon",
             "tue",
             "wed",
             "thu",
             "fri"]
duration=[]
campaign=[]
pdays=[]
previous=[]
poutcome=["failure",
          "nonexistent",
          "success"]
emp=[]
conspriceidx=[]
consconffidx=[]
euribor3m=[]
nr=[]
y=["no",
   "yes"]

#データインデックス
input_list=[age, job, marital, education, default, housing, loan,
       contact, month, day_of_week, duration, campaign, pdays,
       previous, poutcome, emp, conspriceidx,
       consconffidx, euribor3m, nr, y]
input_listx=["age", "job", "marital", "education", "default", "housing", "loan",
       "contact", "month", "day_of_week", "duration", "campaign", "pdays",
       "previous", "poutcome", "emp", "conspriceidx",
       "consconffidx", "euribor3m", "nr", "y"]


#k=1:job,2:marital,3:education,4:default,5:housing,
#6:loan,7:contact,8:month,9:day_of_week, 14:poutcome,20:y

#数値に変換するインデックス番号
nz=[1,2,3,4,5,6,7,8,9,14,20]

#数値に変換する
for k in nz:
    a = array[k]#job
    p=input_list[k]
    q=input_listx[k]
    r=[]
    for i in a:
        for j, e in enumerate(p):
            if i==e:
                r.append(j)
    array[k]=r

array = array.T
#print(array)

#家のローン(5:housing)があるかどうか？
#1:job,2:marital,3:education,4:default,5:housing,
#6:loan,7:contact,8:month,9:day_of_week, 14:poutcome,20:y
T=3

#Xは環境変数（0-T-1, (T+1)-21）, Y(目的変数)は5のインデックス
X = []
Y = []
    
for k in range(len(array)):
    a = array[k]
    a = a.tolist()#type'ndarryをlistにする'tolistを用いると既存のndarrayをリストへ変換
    b = a[0:T-1]
    b1 = a[T+1:21]
    b[len(b):len(b)] = b1
    c = a[T]
    X.append(b)
    Y.append(c)
    
#print(Y)

# =============================================================================
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
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)  
#(3089, 19) (1030, 19) (3089, 3) (1030, 3)出力層3，入力 input_shapreを19
D1 = X_train.shape[1]
D2 = Y_train.shape[1]
#print(D1,type(D1),D2,type(D2))
print('input_index:',D1)
#19 <class 'int'> 8 <class 'int'>

# モデル構築
def build_model():
    model = Sequential()
    model.add(Dense(128,activation='relu',input_shape=(D1,)))
    model.add(Dropout(0.3))
    model.add(Dense(64,activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(D2,activation='softmax'))
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

#Confusion Matrixカラムラベルリスト作成
for i in range(len(PM)):
    l=input_list[T]
    k=PM[i]
    p1="predict_"+l[k]
    p2="real_"+l[k]
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
