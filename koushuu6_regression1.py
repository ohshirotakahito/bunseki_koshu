# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 20:04:57 2021

@author: komoto
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = load_boston() #sklearn.datasetsの中のボストンの住宅価格をロード

print(data)

#0列目のデータ(犯罪件数)と住宅価格を図示
plt.plot(data.data[:,0], data.target ,'o')
plt.ylabel('Price' )
plt.xlabel(data.feature_names[0])
plt.show()

## 全データのうち2割をテスト用、残りを学習用に分割

X_train, X_test, y_train, y_test = train_test_split(data.data, data.target,
                                                    test_size=0.2)

#線形回帰モデルの呼び出し
reg = LinearRegression()

#線形回帰モデルの学習
reg.fit(X_train,y_train)

#X_testの住宅価格の予想
y_pred = reg.predict(X_test)
 
#予想価格と実際の価格の図示
plt.plot(y_test,y_pred,'o')
plt.plot([0, 50], [0, 50], '--k') #y=x に点線を表示
plt.xlabel('True Price / $1000')
plt.ylabel('Predicted Price / $1000')
plt.show()