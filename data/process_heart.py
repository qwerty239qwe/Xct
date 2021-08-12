import numpy as np
import pandas as pd
import scanpy as sc
import scipy
import numpy.core.defchararray as np_f

ada = sc.read_h5ad('./data/global.h5ad')

HVG_i = np.argsort(np.asarray(ada.var['vst.variance.standardized']))[-3000:]
ada = ada[:, HVG_i].copy()
# check critical genes in
for g in ['MIF', 'CD74', 'CCL3', 'CCR1', 'FN1']:
    print(g in ada.var_names)


np.unique(ada.obs['cell_states'])

cc = ada.obs['cell_states'].value_counts() #cell type count
print(cc)

for cellname in list(cc.index[:10]): #filter out top 10 cell types and subsample
    ada = ada[ada.obs['cell_states']!= cellname, :]
sc.pp.subsample(ada, n_obs = 39000, random_state=0, copy = False)

ada.obs['ident'] = np_f.replace(np.asarray(ada.obs['cell_states'], dtype=str), 'Ø', 'A')
ada.obs['ident'] = ada.obs['ident'].astype('category')

counts = scipy.sparse.csr_matrix.toarray(ada.raw[:, HVG_i].X)
data = scipy.sparse.csr_matrix.toarray(ada.X)
ada.layers['raw'] = counts
ada.layers['log1p'] = data

ada.write('.data/heart_global_processed.h5ad')