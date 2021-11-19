# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 15:39:06 2021

@author: komoto
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

fontsize = 18
plt.rcParams['xtick.major.width'] = 2.0#x軸主目盛り線の線幅
plt.rcParams['ytick.major.width'] = 2.0#y軸主目盛り線の線幅
plt.rcParams['font.size'] = fontsize #フォントの大きさ
plt.rcParams['axes.linewidth'] = 2.0# 軸の線幅edge linewidth。囲みの太さ    
plt.ylabel('True Class',fontsize = fontsize)
plt.xlabel('Predicted Class',fontsize = fontsize)

def plot_confusion_matrix(y_test,y_pred,cm=np.array([]),
                          cmstd=np.array([]),labels=[],title='',
                          vmax=0,vmin=0,
                          fontsize = 18,
                          normalize = False,
                          cmap=plt.cm.bwr,normalizedcm=True,
                          savename=''):
    print('test')
    if not cm.any():
        cm = metrics.confusion_matrix(y_test, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
    classes,count = np.unique(y_pred,return_counts=True)
    size = len(classes)
    if not vmax:
        if normalize:
            vmax = 2/size
        else:
            vmax = len(y_test)/size

    threshold = [0.8*vmax,0.2*vmax]

    plt.imshow(cm, interpolation='nearest', cmap=cmap,vmax=vmax,vmin=vmin)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(size)
    plt.xticks(tick_marks, classes, rotation=0)
    if labels:
        plt.yticks(tick_marks, labels, rotation=0)
    else:
        plt.yticks(tick_marks, classes, rotation=0)

    fmt = '.2f'

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if len(cmstd):
                text = format(cm[i, j], fmt) + '\n±' + format(cmstd[i, j], fmt)
            else:
                text = format(cm[i, j], fmt)
            plt.text(j, i, text,
                     horizontalalignment="center",
                     verticalalignment="center",
                     color="white" if cm[i, j] > threshold[0] or cm[i,j] < threshold[1] else "black",
                     fontweight= "normal" if cm[i, j] <1./size else "extra bold",
                     fontname = 'Arial',
                     fontsize = fontsize
                     )

    
    plt.tight_layout()
    if savename:
        plt.savefig(savename,dpi=600)
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.show()

    return cm