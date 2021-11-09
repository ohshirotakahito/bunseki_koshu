# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 18:44:43 2021

@author: komoto
"""
import numpy as np

x = 3
a = np.array([1,2,3])

#関数の定義
def func(x): #ｘに3を足して12掛け、値を返す関数
    t = (x+3)*12
    return t

#関数の呼び出し
print('x',func(x) )
print('a',func(a) )

# class
# わからなかったら深く考えずによく理解しないままでもよい

# classの定義
class class1(object):
    def __init__(self):# コンストラクタ:objが呼び出されるとまず実行される
        self.parameter1 = 1　
        self.parameter2 = 2

    def calculate(self,x):
        return self.parameter1 * x
    
    def analyze(self,x):
        return self.calculate(x)*self.parameter2
    
    def printHello(self):
        print('Hello')
        return 1

# classの呼び出し
obj = class1()

# class1内の関数を実行
obj.printHello()
#　
print('parameter 1',obj.parameter1)
print(obj.calculate(a) )
print(obj.analyze(a) )

# attributeの上書き
obj.parameter1 = 3 
print('parameter 1',obj.parameter1)
print(obj.calculate(a) )

