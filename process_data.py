import os
import json
import multiprocessing

import numpy as np
import igraph as ig
import pandas as pd
import networkx as nx

from tqdm import tqdm
from collections import deque
from matplotlib import pyplot as plt

from HBN import *

from goatools.obo_parser import GODag
from goatools.semantic import deepest_common_ancestor, common_parent_go_ids
from goatools.godag.go_tasks import get_go2parents
from goatools.gosubdag.gosubdag import GoSubDag
from goatools.gosubdag.plot.gosubdag_plot import GoSubDagPlot

# %%

godag = GODag("data/go-basic.obo")
file = open("data/genes.txt", 'r')
genes = [int(x.strip()) for x in file.readlines()]
gIDs = dict([(x,i) for i,x in enumerate(genes)])
file.close()

# %%

tol_all = pd.read_csv("data/resistant.csv", header=None).values.tolist()
tol = np.intersect1d(genes,tol_all)
tolIDs = [gIDs[t] for t in tol]
# intersection 161 genes

# %%

data_gcn = pd.read_csv("data/edgelist.csv")
g2t_new = pd.read_csv("data/ENTREZ_GENE_ID2GOTERM_BP_ALL.txt","\t",header=None,names=["Entrez","GOb"])
g2t_new = g2t_new[g2t_new.Entrez.isin(genes)].reset_index(drop=True)
g2t_new["ID"] = g2t_new.Entrez.apply(lambda x: gIDs[x])
g2t_new["GO"] = g2t_new.GOb.apply(lambda x: x.split('~')[0])

g2t_new.drop("GOb", axis=1, inplace=True)
g2t_new.drop("Entrez", axis=1, inplace=True)
print("GO terms: {0}".format(len(g2t_new.GO.unique())))
print("Gene-GO: {0}".format(g2t_new.shape[0]))

# %%
# Old gene-term associations - source unknown

def array2list(str):
  str = str.replace("\'",'\"')
  return json.loads(str)

g2t_old = pd.read_csv("data/functions_old.csv", header=None, names=["gID", "tList"])
g2t_old_it = g2t_old.to_dict('records')

data = list()
for line in g2t_old_it:
  for term in line["tList"].split():
    data.append((line["gID"], term))
g2t_old = pd.DataFrame(data, columns=["ID","GO"])
g2t_old = g2t_old.drop_duplicates().reset_index(drop=True)
print("GO terms: {0}".format(len(g2t_old.GO.unique())))
print("Old Gene-GO: {0}".format(g2t_old.shape[0]))

# %%
# Concatenate both dataframes and keep only biological processes

g2t = pd.concat([g2t_new, g2t_old])
g2t = g2t.drop_duplicates().reset_index(drop=True)
g2t = g2t[g2t.GO.isin(godag) & (g2t.GO != "GO:0008150")].reset_index(drop=True)
# g2t.to_csv("data/gene_term.csv", index=False)

# %%

g2t = pd.read_csv("data/gene_term.csv")
terms = g2t.GO.unique()
print("Number of terms (non-obsolet): {0}".format(len(terms)))
print("Number of relations: {0}".format(len(g2t)))

# %%
# Get terms ancestral relations for hierarchy creation

isa = list()
for t in tqdm(terms):
  q = deque()
  for p in godag[t].parents:
    q.append((t, p.id))

  while len(q) > 0:
    c, p = q.pop()
    if p != "GO:0008150":
      isa.append((c,p))
      for gp in godag[p].parents:
        q.append((p, gp.id))

isa = pd.DataFrame(isa, columns=['Child','Parent'])
isa = isa.drop_duplicates().reset_index(drop=True)
# isa.to_csv("data/isa.csv", index=False)

all_terms = np.union1d(np.union1d(isa.Child, isa.Parent), terms)
term_def = pd.DataFrame()
term_def["Term"] = all_terms
term_def["Desc"] = [godag[t].name for t in all_terms]
# term_def.to_csv("data/term_def.csv", index=False)

print('Number of terms: {0}'.format(len(all_terms)))
print('Number of relations: {0}'.format(len(isa)))

# %%

ng = len(genes)
terms = all_terms.copy()
nt, tIDs = len(terms), dict([(t,i) for i,t in enumerate(terms)])

# go by go matrix
# nt:number of terms, tIDs:term index map
go_by_go = np.zeros((nt,nt))
for edge in tqdm([tuple(x) for x in isa.to_numpy()]):
  u, v = tIDs[edge[0]], tIDs[edge[1]]
  go_by_go[u,v] = 1

# compute the transitive closure of the ancestor of a term (idx)
def ancestors(term):
  tmp = np.nonzero(go_by_go[term,:])[0]
  ancs = list()
  while len(tmp) > 0:
    tmp1 = list()
    for i in tmp:
      ancs.append(i)
      tmp1 += np.nonzero(go_by_go[i,:])[0].tolist()
    tmp = list(set(tmp1))
  return ancs

# gene by go matrix
gene_by_go = np.zeros((ng,nt))
for edge in tqdm([tuple(x) for x in g2t.to_numpy()]):
  u, v = edge[0], tIDs[edge[1]]
  gene_by_go[u,v] = 1
  gene_by_go[u,ancestors(v)] = 1

print()
print('**Final data**')
print('Genes: \t\t{0:8}'.format(ng))
print('Gene annot.: \t{0:8}'.format(np.count_nonzero(gene_by_go)))
print('Co-expression: \t{0:8.0f}'.format(len(data_gcn)))
print('GO terms: \t{0:8}'.format(nt))
print('GO hier.: \t{0:8.0f}'.format(np.sum(go_by_go)))

# %%
ntolIDs = list()
for x in tqdm(tolIDs):
  fx = np.nonzero(gene_by_go[x,:])[0]
  if len(fx) > 0:
    ntolIDs.append(x)
print(len(ntolIDs))

# %%

ntol = list(ntolIDs)

for x in tqdm(ntolIDs):
  fx = np.nonzero(gene_by_go[x,:])[0]
  # print(fx)
  for i in range(ng):
    if i not in ntol:
      fi = np.nonzero(gene_by_go[i,:])[0]
      if len(np.intersect1d(fx, fi)) == len(fx):
        # print(fi)
        ntol.append(i)
    # break
  # break

print(len(set(ntol)))

# %%

ntol = list(set(ntol))
file = open('ndata/ngenes.txt', 'w')
file.write('\n'.join(['{0}'.format(genes[x]) for x in ntol]))
file.close()

# %%

ntol = list(set(ntol))
a = gene_by_go[ntol,:].sum(axis=0)
ntol_terms = np.nonzero(a)[0]

file = open('ndata/nterms.txt', 'w')
file.write('\n'.join(['{0}'.format(terms[x]) for x in np.nonzero(a)[0]]))
file.close()

# %%

tol_gcn = data_gcn[(data_gcn.source.isin(ntol)) & (data_gcn.target.isin(ntol))].copy()
tol_gcn.columns = ["source","target","weight"]
tol_gcn.to_csv("ndata/ngcn.csv", index=False)
print(len(tol_gcn))
print(len(set(tol_gcn.source.tolist()+tol_gcn.target.tolist())))

# %%

g = nx.from_pandas_edgelist(tol_gcn, edge_attr='weight', create_using=nx.Graph())
print(nx.info(g))
nx.is_connected(g)

# %%

ngene_by_go = gene_by_go[np.ix_(ntol, ntol_terms)].copy()
np.savetxt('ndata/ngene_term.txt', ngene_by_go, fmt='%d', delimiter=',')

ngo_by_go = go_by_go[np.ix_(ntol_terms, ntol_terms)].copy()
np.savetxt('ndata/nterm_hier.txt', ngo_by_go, fmt='%d', delimiter=',')

# %%

def create_path(path):
  try:
    os.makedirs(path)
  except:
    pass

for fl in tqdm(['n{0}'.format(x) for x in range(10,101,10)]):
  clust = os.listdir('clustering/{0}'.format(fl))
  for c in clust:
    nc = 'GO:' + c.split('.')[0][2:]
    if nc in terms[ntol_terms]:
      cdf = pd.read_csv('clustering/{0}/{1}'.format(fl,c))
      cdf =cdf.loc[ntol].copy()
      create_path('ndata/clust/{0}'.format(fl[1:]))
      cdf.to_csv('ndata/clust/{0}/{1}'.format(fl[1:],c), index=False)

# %%
len(ntol_terms)
# %%
# Create file with list of functions per gene

data = list()
for i in tqdm(range(len(ntol))):
  tlist = np.nonzero(ngene_by_go[i,:])[0]
  if len(tlist) > 0:
    data.append((i,terms[ntol_terms[tlist]]))
termList = pd.DataFrame(data, columns=["ID","tList"])
termList.to_csv("ndata/ngene_term_list.csv", index=False)

# %%

def list2file(l, name):
  file = open(name, 'w')
  file.write('\n'.join([str(x) for x in l]))
  file.close()

def create_path(path):
  try: os.makedirs(path)
  except: pass

# %%

# roots
for i in range(len(ntol_terms)):
  if ngo_by_go[i,:].sum() == 0:
    print(i,terms[ntol_terms[i]], ngene_by_go[:,i].sum())

# %%

# hgenes = np.nonzero(gene_by_go[:,root])[0]
# hterms = terms[x] # terms to predict in hierarchy
#
# # Conver DAG to tree, will be used for prediction
# tree = mst(hgenes, x, gene_by_go.copy(), go_by_go.copy())
# hg2g = np.zeros((len(hterms),len(hterms)))
# for i, idx in enumerate(x):
#   parents = direct_pa(idx, x, tree)
#   parents = [np.where(x == p)[0][0] for p in parents]
#   hg2g[i, parents] = 1
#
# q = deque()
# q.append((0,0)) # parent, level
# parents, level = 0, 0
#
# lcn = list()
# lcpn = list()
# lcl, lastl, pterms, cterms = list(), 0, list(), list()
#
# while len(q) > 0:
#   pos, l = q.popleft()
#   children = np.nonzero(hg2g[:,pos])[0]
#
#   # lcl order of prediction
#   if lastl != l:
#     lastl = l
#     lcl.append("{0}= {1}".format(','.join(pterms), ','.join(cterms)))
#     pterms, cterms = list(), list()
#   pterms.append(hterms[pos])
#   cterms += list(hterms[children])
#
#   if len(children) > 0: # is a parent
#     lcpn.append(("{0}= {1}".format(hterms[pos], ','.join(hterms[children])))) # save lcpn order of prediction
#
#     parents += 1
#     for c in children:
#       lcn.append(("{0}= {1}".format(hterms[pos], hterms[c]))) # save lcn order of prediction
#       q.append((c,l+1))
#
#   level = max(level, l)
#
# train_times.loc[j] = [hterms[0], len(x), len(hgenes), len(x)-1, parents, level, 1]
#
# path = "data/{0}".format(hterms[0].replace(':',''))
# create_path(path)
# list2file(lcn, "{0}/lcn.txt".format(path))
# list2file(lcpn, "{0}/lcpn.txt".format(path))
# list2file(lcl, "{0}/lcl.txt".format(path))

# %%

sh_df = pd.DataFrame()
for i, trm in enumerate(terms[ntol_terms]):
  # sh_df[trm] = pd.Series(ngene_by_go[:,i])
  sh_df = pd.concat([sh_df, pd.DataFrame(ngene_by_go[:,i], columns=[trm])], axis=1)
path = "ndata/"
list2file(ntol, '{0}/genes.csv'.format(path))
sh_df.to_csv('{0}/labels.csv'.format(path), index=False)
