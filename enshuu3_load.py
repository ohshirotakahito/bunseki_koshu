# -*- coding: utf-8 -*-
"""
演習
1.次のプログラムを実行して、GraphData1.csvのグラフを確認してください。
2.プログラムを一部変更して、グラフを描画してください
i)x,yは1000点のデータからなります。ｙの800点以降のデータを0にしてください。
ii)ｘの値が50以下の範囲のみ青く描画したグラフを追加してください
　　plt.plot()の'r'を'b'とすることでグラフを青くすることができます

c.f. enshuu_numpy1.py, koushuu1_dtype.py
"""

import numpy as np
import matplotlib.pyplot as plt
import os

##パスの指定
file = os.path.join(os.getcwd(),'testdata','GraphData1.csv')
#windowsなら file = '.\\testdata\\Graphdata1.csv'
#macなら file = './testdata/Graphdata1.csv'
#のように書いてもよい

##
data = np.loadtxt(file,delimiter=',',skiprows=1) # 1000行2列の行列がロードされる
x = data[:,0] #dataの0列目
y = data[:,1] #dataの1列目

###########################################
#　↓　ここに i)の処理を追加


############################################

plt.plot(x,y,'r')#'r'は色が赤の実線であることを示す
plt.xlim(0,200)
plt.xlabel('x')
plt.ylabel('y')

###########################################
#　↓　ここに ii)の処理を追加


###########################################


plt.show()

print('グラフが描画されました。\nspyder右にあるプロットタブを選択してグラフを確認してください。')
