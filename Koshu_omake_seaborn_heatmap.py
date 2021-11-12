import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import pyplot

# 航空機のデータを読み込み
flights = sns.load_dataset("flights")
 
# ピボットを生成
flights = flights.pivot("month", "year", "passengers")
 
# プロットサイズを決める
pyplot.figure(figsize=(12, 5))

# X軸に年、Y軸に月、色の濃さで便数を表す
fig, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(flights, annot=True, fmt="d",center=250)
ax.set_ylim(len(flights), 0)
plt.show()