# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 18:19:15 2021

@author: komoto
"""

#整数の定義
a = 2
print('a=',a)

#浮動小数の定義
b = 3.5
print('b=',b)

#計算
c = a+b
print('c=',c)

#関数の引数内で計算
print(a+b)

#文字列の定義
c = 'test'
print(c)

#文字列の足し算,掛算
c = 'test'+'test'
print(c)
c = 'test'*3
print(c)


#bool型の定義
t = True
f = False

print('t',t)
print('f',f)

#boolの演算
print(t and f)
print(t or f)

#リストの定義
a = [1,4,6,5,7,10,3]
print('a',a)

#要素の取得
print('a[3]',a[3])##この例ではPythonでは0から数え始めるので前から4番目の要素が取得される

#リストの演算
print('a*2',a*2)


#ライブラリのインポート
import numpy as np

#numpyのndarray型の定義
y = np.array([1,4,6,5,7,10,3])
x = np.array([1,1.5,2,3,3.3,3.5,4])

#ndarrayの演算
print('x+1',x+1)
print('x*2',x*2)
print('x+y',x+y)
print('x*y',x*y)

#グラフ描画用のライブラリのインポート
import matplotlib.pyplot as plt

#グラフの描画
plt.plot(x,y)

#グラフの表示
plt.show()

#数学で用いられる計算
x = np.linspace(0,4,100) #0から4まで100分割した配列
y1 = np.exp(x)/50
y2 = np.log10(x)
y3 = np.sin(x/0.2)

plt.plot(x,y1,label='y1')
plt.plot(x,y2,label='y2')
plt.plot(x,y3,label='y3')
plt.legend()
plt.show()

#2次元配列(行列)の定義
a = np.array([[1,4],[1,0]])
b = np.array([[0.5,-0.86],[0.86,0.5]])
x = np.array([0.5,0.86])

print('a',a)
print('b',b)
print('x',x)

#行列の計算
print('b+a',b+a)
print('ba',np.dot(b,a) )
print('bx',np.dot(b,x) )




