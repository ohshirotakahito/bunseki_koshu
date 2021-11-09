# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 16:43:02 2021

@author: komoto
"""

#基本的な文法

#Pythonではプログラミングの構造を指定するためにインデント(字下げを用います。)
#ifやforの次の行はTabを1回押します。
#同じだけインデントが下がっているところがif文の分岐構造やforの繰り返し構造を示します。

#if文

a = 2
if a == 2:
    print('a=2')
else:
    print('a≠2')
    
a = 1
if a == 2:
    print('a=2')
else:
    print('a≠2')

#for文
elements = ['H','He','Li','Be','B','C','N','O']
for element in elements:
    print(element)   
    
s = 0
for i in range(1,11):
    s = s +i
print(s)


"""
---------------演習課題----------------------------
1から100までの順に数字を示すプログラムを作成してください。
ただし、3で割り切れるときは,Fizz,5で割り切れるときはBuzz,
3でも５でも割り切れるときはFizzBuzz
と表記してください。
a を b で割った余りは a % b で求められます。 
以下のような出力が正解になります。
1, 2, Fizz, 4, Buzz, Fizz, 7, 8, Fizz, Buzz, 11, Fizz, 13, 14, Fizz Buzz, 16,
 17, Fizz, 19, Buzz, Fizz, 22, 23, Fizz, Buzz, 26, Fizz, 28, 29, Fizz Buzz, 31,
 32, Fizz, 34, Buzz, Fizz, ...
"""


#答えの一例　時間が足りない？

#　答え1
for i in range(1,101):
    a = i
    if i % 3 == 0:
        a = 'Fizz'
    if i % 5 == 0:
        a = 'Buzz'
    if i % 3 ==0 and i % 5 == 0:
        a ='FizzBuzz'
    print(a)

#　答え2
for i in range(1,101):
    if i % 3 == 0:
        if i % 5 == 0:
            print('FizzBuzz')
        else:
            print('Fizz')
    elif i % 5 == 0:
        print('Buzz')
    else:
        print(i)

[print('FizzBuzz' if x % 15 == 0 else 'Fizz' if x%3 == 0 else 'Buzz' if x % 5== 0 else x)  for x in range(1,101)]

