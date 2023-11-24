import os
import sys

import numpy as np
import pandas as pd

from tqdm import tqdm, tnrange
from time import time

import multiprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import KFold

from evaluate import *

import warnings
warnings.filterwarnings('ignore')
np.seterr(divide='ignore', invalid='ignore')


# read a txt file containing a list, one item per line
def readListFile(filename):
  file = open(filename, 'r')
  tmp = [x.strip() for x in file.readlines()]
  file.close()
  return np.array(tmp)


# pretty print of results for a model
def pprint(model, pooled, macro, macrow, time):
    print("## {0}:{1}pooled:\t{2:.4f}\t\tmacro:\t{3:.4f}\t\tmacrow:\t{4:.4f}\t\tTime:\t{5:.4f}".format(
      model,'\t' if len(model) > 7 else '\t\t',pooled, macro, macrow, time)
    )

# %%

cpu = multiprocessing.cpu_count()
seed = 220824

path = "ndata/"
labels = pd.read_csv("{0}/labels.csv".format(path)) # labels - true gene-function associations
labels_order = labels.columns
gene_list = pd.read_csv("{0}/genes.csv".format(path), names=["idx"])
gene_list = gene_list.idx.tolist()

results = pd.DataFrame(columns=["time","micro","macro","macrow"])

data = pd.DataFrame()
for clust in next(os.walk('ndata/clust'))[1]:
  clf = [x for x in os.listdir('ndata/clust/{0}'.format(clust)) if 'GO:'+x.split('.')[0][2:] in labels_order]
  for file in tqdm(clf):
    tmp = pd.read_csv('ndata/clust/{0}/{1}'.format(clust,file))
    data['{0}_{1}'.format(clust,file.split('.')[0][2:])] = tmp.label
    # tmp.columns = ['{0}_{1}'.format(clust,file.split('.')[0][2:])]
    # data = pd.concat([data, tmp['{0}_{1}'.format(clust,file.split('.')[0][2:])]], axis=1)

# %%

# load both models and evaluation class
estimator = RandomForestClassifier(n_estimators=100, min_samples_split=5, n_jobs=-1, random_state=seed) # Random forest classifier
e = Evaluate()

# array for results
pooled, etime = list(), list()
macro, macrow = list(), list()

"""
k-fold, the lcl approach is applied for each fold independently and the mean
of the performance for each fold is the result
"""
N = 5
fold_idx = 1
kfold = KFold(n_splits=N, shuffle=True, random_state=seed) # set the same folds for each model
for train_index, test_index in tqdm(kfold.split(labels), total=N, ascii="-#", desc="Training cross-validation"): # set the same folds for each model

  # prediction for the current fold (each fold is independent)
  pred = np.zeros((len(test_index), len(labels_order))) # results for random forest multilabel classifier

  """ create y and x dataset for the current hierarchy """
  y = labels.copy() # labels for all functions in level (i.e., children)
  y_train, y_test = y.loc[train_index], y.loc[test_index] # train-test split of y
  X_train, X_test = data.loc[train_index], data.loc[test_index] # train-test split for random forest

  """
  training and prediction for the current level. A special case arise for
  random forest when there is only ony child in the level, in that case the
  output of the rf is different
  """
  # random forest
  s_time = time()
  estimator.fit(X_train, y_train)
  _pred = estimator.predict_proba(X_test)
  for cidx, x, cls in zip(range(len(labels.columns)), _pred, estimator.classes_):
    pred[:,cidx] = x[:,0] if cls[0] == 1 else 1 - x[:,0]
  f_time = time() - s_time

  """
  storing predictions and performance measures for the current fold.
  Predictions are saved in a different file for each fold to avoid loosing
  the fold indexes
  """
  pred_df = pd.DataFrame(pred, columns=labels_order)
  pred_df.index = test_index
  pred_df.to_csv("npred/prob/{0}.csv".format(fold_idx))
  fold_idx += 1

  """
  compute the performance of the whole hierarchy
  """
  measures = e.multiclass_classification_measures(pred, y_test)
  macro.append(measures[1])
  macrow.append(measures[2])
  pooled.append(measures[3])
  etime.append(f_time)

  results.loc[fold_idx-2] = [f_time,measures[3],measures[1],measures[2]]
  pprint("Fold {0}:".format(fold_idx-1),measures[3],measures[1],measures[2],f_time)

"""
final performance of the hierarchy is computed, mean of the performance for
each fold. Then, the performance measures are saved
"""
# print(pooled)
pprint("Final: ", np.mean(pooled), np.mean(macro), np.mean(macrow), np.mean(etime))
results.loc[fold_idx-2] = [np.mean(etime),np.mean(pooled), np.mean(macro), np.mean(macrow)]
results.to_csv("npred/global.csv", index=False)
