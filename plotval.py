# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:hydrogen
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.2'
#       jupytext_version: 1.2.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
from sklearn.utils.multiclass import unique_labels
from itertools import zip_longest
import numpy as np
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from itertools import cycle
import matplotlib.font_manager as fm
 
myfont = fm.FontProperties(fname='simhei.ttf') # 设置字体


#解决中文显示问题
#plt.rcParams['font.sans-serif'] = ['KaiTi'] # 指定默认字体
#plt.rcParams['font.sans-serif'] = ['SimHei']
#plt.rcParams['font.serif'] = ['SimHei']
#plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题


def draw_roc(class_names, y_true, y_pred):
  fpr = dict()
  tpr = dict()
  roc_auc = dict()
  for i, c in enumerate(class_names):
    #print(y_true[:, i])
    #print(y_pred[:, i])
    fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
  # Compute micro-average ROC curve and ROC area
  fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred.ravel())
  roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
  # Compute macro-average ROC curve and ROC area
  n_classes = len(class_names)
  # First aggregate all false positive rates
  all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

  # Then interpolate all ROC curves at this points
  mean_tpr = np.zeros_like(all_fpr)
  for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

  # Finally average it and compute AUC
  mean_tpr /= n_classes

  fpr["macro"] = all_fpr
  tpr["macro"] = mean_tpr
  roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

  # Plot all ROC curves
  lw = 2
  plt.figure()
  plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

  plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

  colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
  for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(class_names[i], roc_auc[i]))

  plt.plot([0, 1], [0, 1], 'k--', lw=lw)
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Some extension of Receiver operating characteristic to multi-class')
  plt.legend(loc="lower right")
  plt.savefig('ROC_Curve.jpg', dpi=400)
  plt.show()
  
def calculate_score(class_names, y_true, y_pred):
  '''
  :class_names(list): ['normal', 'abnormal']
  :y_true(1-D): (batchsize, )
  :y_pred(2-D): (batchsize, predict_output)
  :return(tuple): class scores history, mean socre
  '''
  score_list = []
  if not hasattr(calculate_score, 'aurocs'):
    aurocs = {}
    for c in class_names:
      aurocs[c] = []
  for i, c in enumerate(class_names):
    try:
      score = roc_auc_score(y_true[:,i], y_pred[:,i])
    except ValueError:
      score = 0
    aurocs[c].append(score)
    score_list.append(score)
  return aurocs, np.mean(score_list)

def confusion_matrix(y_true, y_pred):
  assert len(y_true) == len(y_pred), 'y_true must be equal to y_pred shape[0]'
  t = []
  p = []
  for e_true, e_pred in zip(y_true, y_pred):
    for label_true, label_pred in zip_longest(e_true, e_pred, fillvalue=e_true[0]):
      t.append(label_true)
      p.append(label_pred)
  npt = np.asarray(t)
  npp = np.asarray(p)
  n_labels = len(set(npt))
  sample_weight = np.ones((npt.shape[0],), dtype=np.int64)
  cm = coo_matrix((sample_weight, (npt, npp)), shape=(n_labels, n_labels), dtype=np.int64).toarray()
  return npt,npp,cm

def plot_confusion_matrix(y_true, y_pred, classes,
                          savefile=False,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    :y_true(2-D array): true label array, i.e. [[1,0],[0,1],[0]]
    :y_pred(2-D array): predict label array, i.e. [[1,0],[0,1],[2]]
    :classes(1-D array): true label class names, i.e. ['classname1','classname2']
    :savefile(string): save to absolute file path
    :normalize(boolean): normalize or not
    :title: the name of the figure
    """
    ft_size = 16
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    y_true, y_pred, cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)
    fig, ax = plt.subplots(figsize=(8,6))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    #ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes)
           #title=title,
           #ylabel='True label',
           #xlabel='Predicted label')
    plt.title(title,fontsize=ft_size)
    plt.xticks(fontsize=ft_size)
    plt.yticks(fontsize=ft_size)
    plt.xlabel('Predicted label', fontsize=ft_size)
    plt.ylabel('True label', fontsize=ft_size)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",fontsize=ft_size,
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    if savefile!=False:
        plt.savefig(savefile, dpi=400)
    return ax
