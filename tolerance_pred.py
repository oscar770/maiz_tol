import os
import sys

import numpy as np
import pandas as pd

from tqdm import tqdm, tnrange
from time import time

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import KFold, StratifiedKFold

from sklearn import metrics

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.pipeline import make_pipeline

from matplotlib import pyplot as plt

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

# gtol = pd.read_csv("data/resistant.csv", names=['gene'])
# gtol.gene = gtol.gene.astype('string')
# gtol = gtol.gene.tolist()
# genes = pd.DataFrame()
# genes['id'] = readListFile('ndata/ngenes.txt')
# genes['tol'] = [1 if x in gtol else 0 for x in genes.id.tolist()]
# genes.to_csv('ndata/tolerance.csv', index=False)

seed = 220824

path = "ndata/"
labels = pd.read_csv("{0}/labels.csv".format(path)) # labels - true gene-function associations
labels_order = labels.columns
tol = pd.read_csv("{0}/tolerance.csv".format(path)).tol # labels - tolerance genes
gene_list = readListFile('ndata/ngenes.txt')

results = pd.DataFrame(columns=["fold","balanced_accuracy_score","average_precision_score","f1_score","precision_score","recall_score","roc_auc_score","tn","fp","fn","tp"])

data_clust = pd.DataFrame()
for clust in tqdm(next(os.walk('ndata/clust'))[1], total=10):
    clf = [x for x in os.listdir('ndata/clust/{0}'.format(clust)) if 'GO:'+x.split('.')[0][2:] in labels_order]
    for file in clf:
        tmp = pd.read_csv('ndata/clust/{0}/{1}'.format(clust,file))
        data_clust['{0}_{1}'.format(clust,file.split('.')[0][2:])] = tmp.label

data_assoc = pd.DataFrame(np.zeros(labels.shape), columns=labels_order)
for fold in range(1,6):
  tmp = pd.read_csv('npred/prob/{0}.csv'.format(fold), index_col=0)
  data_assoc.loc[tmp.index] = tmp.values

data = pd.concat([data_assoc, data_clust], axis=1)
# %%

# load both models and evaluation class
estimator = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed) # Random forest classifier
estimator = XGBClassifier(importance_type="gain", eval_metric=metrics.precision_score, tree_method="hist", objective="binary:logistic", random_state=seed) # Random forest classifier
pred = np.zeros(len(gene_list)) # results for random forest multilabel classifier
smote = SMOTE(sampling_strategy='minority', random_state=seed)

"""
k-fold, the lcl approach is applied for each fold independently and the mean
of the performance for each fold is the result
"""
N = 5
fold_idx = 1
kfold = StratifiedKFold(n_splits=N, shuffle=True, random_state=seed) # set the same folds for each model
for train_index, test_index in tqdm(kfold.split(data,tol), total=N, ascii="-#", desc="Training cross-validation"): # set the same folds for each model

    # prediction for the current fold (each fold is independent)

    """ create y and x dataset for the current hierarchy """
    y = tol.copy() # labels for all functions in level (i.e., children)
    y_train, y_test = y.loc[train_index], y.loc[test_index] # train-test split of y
    X_train, X_test = data.loc[train_index], data.loc[test_index] # train-test split for random forest

    """
    training and prediction for the current level. A special case arise for
    random forest when there is only ony child in the level, in that case the
    output of the rf is different
    """
    # random forest
    clf = make_pipeline(smote,estimator)
    clf.fit(X_train, y_train)
    _pred = clf.predict_proba(X_test)
    pred[test_index] = _pred[:,0] if clf.classes_[0] == 1 else 1 - _pred[:,0]

    """
    storing predictions and performance measures for the current fold.
    Predictions are saved in a different file for each fold to avoid loosing
    the fold indexes
    """
    pred_df = pd.DataFrame(pred[test_index], columns=['tolerance'])
    pred_df.index = test_index
    pred_df.to_csv("npred/prob/tol_{0}.csv".format(fold_idx))

    y_pred = pred[test_index]
    y_pred_bin = pred[test_index].copy()
    y_pred_bin[y_pred<0.5] = 0
    y_pred_bin[y_pred>=0.5] = 1
    """
    compute the performance of the whole hierarchy
    """
    tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_pred_bin).ravel()
    m = [
      fold_idx,
      metrics.balanced_accuracy_score(y_test, y_pred_bin),
      metrics.average_precision_score(y_test, y_pred),
      metrics.f1_score(y_test, y_pred_bin),
      metrics.precision_score(y_test, y_pred_bin),
      metrics.recall_score(y_test, y_pred_bin),
      metrics.roc_auc_score(y_test, y_pred),
      tn, fp, fn, tp,
    ]

    results.loc[fold_idx-1] = m
    fold_idx += 1

pred_df = pd.DataFrame(pred, columns=['tolerance'])
pred_df.to_csv("npred/prob/tol.csv", index=False)

"""
final performance of the hierarchy is computed, mean of the performance for
each fold. Then, the performance measures are saved
"""
pred_bin = pred.copy()
pred_bin[pred_bin<0.5] = 0
pred_bin[pred_bin>=0.5] = 1
tn, fp, fn, tp = metrics.confusion_matrix(tol, pred_bin).ravel()
m = [
  'final',
  metrics.balanced_accuracy_score(tol, pred_bin),
  metrics.average_precision_score(tol, pred),
  metrics.f1_score(tol, pred_bin),
  metrics.precision_score(tol, pred_bin),
  metrics.recall_score(tol, pred_bin),
  metrics.roc_auc_score(tol, pred),
  tn, fp, fn, tp,
]

topn = pd.DataFrame()
topn['gene'] = gene_list
topn['prob'] = pred
topn['tol'] = tol
topn = topn.sort_values(by=['prob'], ascending=False)
topn.to_csv("npred/tol_prob.csv", index=False)
prec = list()
x = [x for x in range(10,301,10)]
for n in x:
  _topn = topn.head(n)
  prec.append(_topn.tol.sum()/_topn.shape[0])

fig, ax  = plt.subplots(figsize=(10,6))
plt.plot(x, prec, '--o')
plt.xlabel('N')
plt.ylabel('Precision')
plt.grid(axis='both')
plt.tight_layout()
plt.savefig('npred/prec.pdf', dpi=600, format='pdf')

results.loc[fold_idx-1] = m
results.to_csv("npred/tol.csv", index=False)
