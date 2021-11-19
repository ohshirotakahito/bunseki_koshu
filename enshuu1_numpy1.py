# -*- coding: utf-8 -*-
"""
以下のプログラムを出力結果を予想した後に実行してください
@author: komoto
"""
import numpy as np

##1次元配列
a = np.array([1,3,5,7,9])

print('a[1]')
print(a[1] )

print('a[3]')
print(a[3] )

print('a[:2]')
print(a[:2] )
print('a[2:]')
print(a[2:] )

##二次元配列

b = np.array([[1,2,3],
              [4,5,6],
              [7,8,9] ])

print('b[0,0]')
print(b[0,0])

print('b[1]')
print(b[1])

print('b[:,1]')
print(b[:,1])

print('b[1:,1:]')
print(b[1:,1:])

## bool型(TrueもしくはFlase)を利用した要素抽出
print('a[np.array([False,False,True,True,True])] ')
print(a[np.array([False,False,True,True,True])] )

print('a > 4')
print(a > 4)

print('a[a>4]')
print(a[a>4])

