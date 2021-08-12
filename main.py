import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
import scipy
from scipy.optimize import least_squares
import warnings
warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

try:
    from sys import path as syspath
    from os import path as ospath
    syspath.append(ospath.join(ospath.expanduser("~"), './'))
    
    from pcNet import pcNet
    import dNN 
except ImportError:
    print('Module not found')


class Xct_metrics():
    '''require adata with layer 'raw' (counts) and 'log1p' (normalized), cell labels in obs 'ident' '''
    __slots__ = ('genes', 'DB', '_genes_index_DB')
    def __init__(self, adata, specis = 'Human'): 
        if not ('raw' and 'log1p' in adata.layers.keys()):
            raise NameError('require adata with count and normalized layers named \'raw\' and \'log1p\'')
        else:
            self.genes = adata.var_names
            self.DB = self.Xct_DB(specis = specis)
            self._genes_index_DB = self.get_index(DB = self.DB)

    
    def Xct_DB(self, specis = 'Human'):
        '''load omnipath DB for L-R pairs'''
        if specis == 'Mouse':
            pass
        elif specis == 'Human':
            LR = pd.read_csv('https://raw.githubusercontent.com/yjgeno/Xct/note/DB/omnipath_intercell_toUse_v1.csv')
        else:
            raise NameError('Current DB only supports \'Mouse\' and \'Human\'')
        LR_toUse = LR[['genesymbol_intercell_source', 'genesymbol_intercell_target']]
        LR_toUse.columns = ['ligand', 'receptor']
        del LR

        return LR_toUse
    
    def subset(self):
        '''subset adata var with only DB L and R'''
        genes = np.ravel(self.DB.values) 
        genes = np.unique(genes[genes != None])
        genes_use = self.genes.intersection(genes)
            
        return [list(self.genes).index(g) for g in genes_use]    #index in orig adata
    
    def get_index(self, DB):
        '''original index of DB L-R pairs in adata var'''
        g_LRs = DB.iloc[:, :2].values #L-R
        gene_list = [None] + list(self.genes) 

        gene_index = np.zeros(len(np.ravel(g_LRs)), dtype = int)
        for g in gene_list:
            g_index = np.asarray(np.where(np.isin(np.ravel(g_LRs), g)))
            if g_index.size == 0:
                continue
            else:
                for i in g_index:
                    gene_index[i] = gene_list.index(g) 
        genes_index_DB = np.array(gene_index).reshape(g_LRs.shape) #gene index refer to subset adata var + 1
        
        return genes_index_DB


    def get_metric(self, adata, verbose = False): #require normalized data
        '''compute metrics for each gene'''
        data_norm = scipy.sparse.csr_matrix.toarray(adata.X) if scipy.sparse.issparse(adata.X) else adata.X
        if verbose:
            print('(cell, feature):', data_norm.shape)
        
        if (data_norm % 1 != 0).any(): #check space: True for log (float), False for counts (int)
            mean = np.mean(data_norm, axis = 0)
            var = np.var(data_norm, axis = 0)
            mean[mean == 0] = 1e-12
            dispersion = var / mean    
            cv = np.sqrt(var) / mean

            return mean, var, dispersion, cv
        else:
            raise ValueError("require log data")
    
    def chen2016_fit(self, adata, plot = False, verbose = False): #require raw data 
        '''NB model fit mean vs CV'''
        data_raw = adata.layers['raw'] #.copy()
        if (data_raw % 1 != 0).any():
            raise ValueError("require counts (int) data")
        else:
            mean_raw = np.mean(data_raw, axis = 0)
            var_raw = np.var(data_raw, axis = 0)
            mean_raw[mean_raw == 0] = 1e-12
            cv_raw = np.sqrt(var_raw) / mean_raw
        
        xdata_orig = mean_raw #raw
        ydata_orig = np.log10(cv_raw) #log   
        rows = len(xdata_orig) #features

        r = np.invert(np.isinf(ydata_orig)) # filter -Inf
        ydata = ydata_orig[r] #Y
        xdata = xdata_orig[r] #X

        #poly fit: log-log
        z = np.polyfit(np.log10(xdata), ydata, 2) 

        def predict(z, x):
            return z[0]*(x**2) + z[1]*x + z[2]

        xSeq_log = np.arange(min(np.log10(xdata)), max(np.log10(xdata)), 0.005) 
        ySeq_log = predict(z, xSeq_log)  #predicted y

        #start point for fit
        #plt.hist(np.log10(xdata), bins=100)
        def h(i):
            a = np.log10(xdata) >= (xSeq_log[i] - 0.05)
            b = np.log10(xdata) < (xSeq_log[i] + 0.05)
            return np.sum((a & b))

        gapNum = [h(i) for i in range(0, len(xSeq_log))] #density histogram of xdata
        cdx = np.nonzero(np.array(gapNum) > rows*0.005)[0] #start from high density bin

        xSeq = 10 ** xSeq_log 

        #end pointy for fit
        yDiff = np.diff(ySeq_log, 1) #a[i+1] - a[i]
        ix = np.nonzero((yDiff > 0) & (np.log10(xSeq[0:-1]) > 0))[0] # index of such (X, Y) at lowest Y

        if len(ix) == 0:
            ix = len(ySeq_log) - 1 # use all
        else:
            ix = ix[0]

        #subset data for fit
        xSeq_all = 10**np.arange(min(np.log10(xdata)), max(np.log10(xdata)), 0.001) 
        xSeq = xSeq[cdx[0]:ix]
        ySeq_log = ySeq_log[cdx[0]:ix]

        if verbose:
            #print(ix, cdx[0])
            print('{} (intervals for fit) / {} (filtered -Inf) / {} (original) features for the fit'.format(ix-cdx[0], len(ydata), len(ydata_orig)))

        #lst fit
        def residuals(coeff, t, y):
            return y - 0.5 * (np.log10(coeff[1]/t + coeff[0])) # x: raw mean y:log(cv)

        x0 = np.array([0, 1], dtype=float) # initial guess a=0, b=1
        model = least_squares(residuals, x0, loss='soft_l1', f_scale= 0.01, args=(xSeq, ySeq_log))

        def predict_robust(coeff, x):
            return 0.5 * (np.log10(coeff[1]/x + coeff[0]))

        ydataFit = predict_robust(model.x, xdata_orig) #logCV

        def cv_diff(obs_cv, fit_cv): 
            obs_cv[obs_cv == 0] = 1e-12
            diff = np.log10(obs_cv) - fit_cv
            return diff #{key: v for key, v in zip(self.genes, diff)} 

        if plot:
            y_predict = predict_robust(model.x, xSeq) 
            plt.figure(figsize=(6, 5), dpi=80)   
            plt.scatter(np.log10(xdata), ydata, s=3, marker='o') # orig
            plt.plot(np.log10(xSeq), ySeq_log, c='black', label='poly fit') # poly fit
            plt.plot(np.log10(xSeq), y_predict, label='robust lsq', c='r') # robust nonlinear

            #ind = list(res[res['padj'] < fdr].index)[:ngenes] # index for filtered xdata, ydata
            #for n, i in zip(['CCL19', 'CCR7', 'CXCL12', 'CXCR4'], [371, 388, 592, 598]):
            #   plt.annotate(n, xy = (np.log10(xdata)[i], ydata[i]), xytext = (np.log10(xdata)[i]+1, ydata[i]+0.5),
            #                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
            plt.xlabel('log10(mean)')
            plt.ylabel('log10(CV)')
            plt.legend(loc='lower left')
            plt.show()
        
        diff = cv_diff(cv_raw, ydataFit)
        return diff #log CV difference
      
      
      
class Xct(Xct_metrics):

    def __init__(self, adata, CellA, CellB, pmt = False, build_GRN = False, save_GRN = False, pcNet_name = 'pcNet', mode = None, verbose = False): 
        '''build_GRN: if True to build GRN thru pcNet, if False to load built GRN files;
            save_GRN: save constructed 2 pcNet;
            pcNet_name: name of GRN (.csv) files, read/write;
            3 modes to construct correspondence: None, 'combinators', 'pairs'
            '''
        Xct_metrics.__init__(self, adata)
        self._cell_names = CellA, CellB
        self._metric_names = ['mean', 'var', 'disp', 'cv', 'cv_res']

        if not ('ident' in adata.obs.keys()):
            raise NameError('require adata with cell labels saved in \'ident\'')
        else:
            ada_A = adata[adata.obs['ident'] == CellA, :].copy()
            ada_B = adata[adata.obs['ident'] == CellB, :].copy()
        self._cell_numbers = ada_A.shape[0], ada_B.shape[0]
        self.genes_names = list(ada_A.var_names.astype(str)), list(ada_B.var_names.astype(str))
        self._X = ada_A.X, ada_B.X # input array for nn projection
        if verbose:
            print(f'for interactions from {self._cell_names[0]} to {self._cell_names[1]} = {self._cell_numbers[0]} to {self._cell_numbers[1]}...')
        
        self._metric_A = np.vstack([self.get_metric(ada_A), self.chen2016_fit(ada_A)]) #len 5
        self._metric_B = np.vstack([self.get_metric(ada_B), self.chen2016_fit(ada_B)])
        
        if not pmt:
            self.ref = self.fill_metric()
            self.genes_index = self.get_index(DB = self.ref)
        if build_GRN: 
            if verbose:
                print('building GRN...')
            self._net_A = pcNet(ada_A.X, nComp=5, symmetric=True)
            if verbose:
                print('GRN of Cell A has been built, start building GRN of Cell B...')
            self._net_B = pcNet(ada_B.X, nComp=5, symmetric=True)
            if verbose:
                print('GRN of Cell B has been built, building correspondence..')
            if save_GRN:
                np.savetxt(f'data/{pcNet_name}_A.csv', self._net_A, delimiter="\t")
                np.savetxt(f'data/{pcNet_name}_B.csv', self._net_B, delimiter="\t")
        else:
            try:
                self._net_A = np.genfromtxt(f'data/{pcNet_name}_A.csv', delimiter="\t")
                self._net_B = np.genfromtxt(f'data/{pcNet_name}_B.csv', delimiter="\t")
                print('GRNs loaded...')
            except ImportError:
                print('require pcNet_name where csv files saved in with tab as delimiter')

        self._w = self.build_w(queryDB = mode, scale = True) 
        if verbose:
            print('correspondence has been built...')      
        del ada_A, ada_B

    def __str__(self):
        info = f'Xct object with the interaction between cells {self._cell_names[0]} X {self._cell_names[1]} = {self._cell_numbers[0]} X {self._cell_numbers[1]}'
        if '_w' in dir(self):
            return info + f'\n# of genes = {len(self.genes_names[0])} X {len(self.genes_names[1])} \nCorrespondence = {self._w.shape[0]} X {self._w.shape[1]}'
        else:
            return info
               
    def fill_metric(self, ref_obj = None, verbose = False):
        '''fill the corresponding metrics for genes of selected pairs (L-R candidates)'''
        if ref_obj is None:
            genes_index = self._genes_index_DB
        else:
            #if isinstance(ref_obj, Xct):
            genes_index = ref_obj.genes_index
        #print(genes_index)
        
        index_L = genes_index[:, 0]
        index_R = genes_index[:, 1]

        df = pd.DataFrame()

        for metric_A, metric_B, metric in zip(self._metric_A, self._metric_B, self._metric_names):
            filled_L = []
            filled_R = []
            for i in index_L:
                if i == 0:
                    filled_L.append(0) #none expression
                else:
                    filled_L.append(np.round(metric_A[i-1], 11))
            filled_L = np.array(filled_L, dtype=float)

            for i in index_R:
                if i == 0:
                    filled_R.append(0)
                else:
                    filled_R.append(np.round(metric_B[i-1], 11))
            filled_R = np.array(filled_R, dtype=float)

            filled = np.concatenate((filled_L[:, None], filled_R[:, None]), axis=1)
            result = pd.DataFrame(data = filled, columns = [f'{metric}_L', f'{metric}_R'])
            df = pd.concat([df, result], axis=1)   
        #DB = skin.DB.reset_index(drop = True, inplace = False) 
           
        if ref_obj is None:
            df = pd.concat([self.DB, df], axis=1) # concat 1:1 since sharing same index
            mask1 = (df['mean_L'] > 0) & (df['mean_R'] > 0) # filter 0 for first LR
            df = df[mask1]
            
        else: 
            ref_DB = self.DB.iloc[ref_obj.ref.index, :].reset_index(drop = True, inplace = False) #match index
            df = pd.concat([ref_DB, df], axis=1)
            df.set_index(pd.Index(ref_obj.ref.index), inplace = True)
            
        #df.replace(to_replace={0:None}, inplace = True) #for geo mean: replace 0 to None
        
        #df.to_csv('df.csv', index=False)
        if verbose:
            print('Selected {} LR pairs'.format(df.shape[0]))

        return df

    def _candidates(self, df_filtered):
        '''selected L-R candidates'''
        candidates = [a+'_'+b for a, b in zip(np.asarray(df_filtered['ligand'],dtype=str), np.asarray(df_filtered['receptor'],dtype=str))]
        return candidates
    
    
    def score(self, ref_DB = None, method = 0, a = 1):
        '''L-R score'''
        if ref_DB is None:
            ref_DB = self.ref.copy()
        S0 = ref_DB['mean_L'] * ref_DB['mean_R'] 
        S0 /= np.percentile(S0, 80) 
        S0 = S0/(0.5 + S0)

        if method == 0:
            return S0  
        if method == 1:
            S = (ref_DB['mean_L']**2 + ref_DB['var_L']) + a*(ref_DB['mean_R']**2 + ref_DB['var_R'])
            S = S/(0.5 + S)
        if method == 2:
            S = ref_DB['disp_L'] * ref_DB['disp_R']
        if method == 3:
            S = ref_DB['cv_L'] + a*ref_DB['cv_R']
        if method == 4:
            ref_DB['cv_res_L'][ref_DB['cv_res_L'] < 0] = 0
            S = abs(ref_DB['cv_res_L'] * ref_DB['cv_res_R'])
            S = S/(0.5+S) + a*S0
        if method == 5:
            S = ref_DB['cv_res_L'] + a*ref_DB['cv_res_R']

        return S #.astype(float)

    def build_w(self, queryDB = 'full', scale = True, mu = 1): 
        '''build w: 3 modes, if 'full' will use all the corresponding scores'''
        # u^2 + var
        metric_A_temp = (np.square(self._metric_A[0]) + self._metric_A[1])[:, None] 
        metric_B_temp = (np.square(self._metric_B[0]) + self._metric_B[1])[None, :] 
        #print(metric_A_temp.shape, metric_B_temp.shape)
        w12 = metric_A_temp@metric_B_temp
        w12_orig = w12.copy()
        
        def zero_out_w(w, LR_idx):
            lig_idx = np.ravel(np.asarray(LR_idx[:, 0]))
            lig_idx = list(np.unique(lig_idx[lig_idx != 0]) - 1)
            rec_idx = np.ravel(np.asarray(LR_idx[:, 1]))
            rec_idx = list(np.unique(rec_idx[rec_idx != 0]) - 1)
            # reverse select and zeros LR that not in idx list
            mask_lig = np.ones(w.shape[0], dtype=np.bool)
            mask_lig[lig_idx] = 0
            mask_rec = np.ones(w.shape[1], dtype=np.bool)
            mask_rec[rec_idx] = 0
            
            w[mask_lig, :] = 0
            w[:, mask_rec] = 0 
            assert np.count_nonzero(w) == len(lig_idx) * len(rec_idx)    
            
            return w

        if queryDB not in ['full', 'comb', 'pairs']:
            raise NameError('queryDB using the keyword \'full\', \'comb\' or \'pairs\'')
        elif queryDB == 'full':
            pass
        elif queryDB == 'comb':
            # ada.var index of LR genes (the intersect of DB and object genes, no pair relationship maintained)
            LR_idx_toUse = self._genes_index_DB
            w12 = zero_out_w(w12, LR_idx_toUse)
        elif queryDB == 'pairs':
            # maintain L-R pair relationship, both > 0
            LR_idx_toUse = self._genes_index_DB[(self._genes_index_DB[:, 0] > 0) & (self._genes_index_DB[:, 1] > 0)]
            w12 = zero_out_w(w12, LR_idx_toUse)

        if scale:
            w12 = mu * ((self._net_A+1).sum() + (self._net_B+1).sum()) / (2 * w12_orig.sum()) * w12 #scale factor using w12_orig

        w = np.block([[self._net_A+1, w12],
            [w12.T, self._net_B+1]])

        return w

    def nn_projection(self, d = 2, n = 3000, lr = 0.001, plot_loss = False):
        '''manifold alignment by neural network'''
        x1_np = scipy.sparse.csr_matrix.toarray(self._X[0].T) if scipy.sparse.issparse(self._X[0]) else self._X[0].T #gene by cell
        x2_np = scipy.sparse.csr_matrix.toarray(self._X[1].T) if scipy.sparse.issparse(self._X[1]) else self._X[1].T
        
        projections, losses = dNN.train_and_project(x1_np, x2_np, d=d, w=self._w, n=n, lr=lr)
        if plot_loss:
            plt.figure(figsize=(6, 5), dpi=80)
            plt.plot(np.arange(len(losses))*100, losses)
            #plt.savefig('fig.png', dpi=80)
            plt.show()
        
        return projections, losses

    def _pair_distance(self, projections): #projections: manifold alignment, ndarray
        '''distances of each pair in latent space'''
        d = {}
        for i, l in enumerate(self.genes_names[0]):
            for j, r in enumerate(self.genes_names[1]):
                d[f'{l}_{r}'] = [(i, j), np.linalg.norm(projections[i, :] - projections[len(self.genes_names[0]) + j, :])]
        
        return d    

    def nn_output(self, projections):
        '''output info of each pair'''
        #manifold alignment pair distances
        print('computing pair-wise distances...')
        result_nn = self._pair_distance(projections) #dict
        print('manifold aligned # of pairs:', len(result_nn))

        #output df
        print('adding column \'rank\'...')
        df_nn = pd.DataFrame.from_dict(result_nn, orient='index', columns=['idx', 'dist']).sort_values(by=['dist'])
        df_nn['rank'] = np.arange(len(df_nn))
        
        print('adding column \'correspondence_score\'...')
        w12 = self._w[:self._net_A.shape[0], self._net_A.shape[1]:]
        correspondence_score = [w12[idx] for idx in np.asarray(df_nn['idx'])]
        df_nn['correspondence_score'] = correspondence_score
    
        return df_nn 
    
    def filtered_nn_output(self, df_nn, candidates):
        df_nn_filtered = df_nn.loc[candidates].sort_values(by=['rank']) #dist ranked L-R candidates
        print('manifold aligned # of L-R pairs:', len(df_nn_filtered))
        df_nn_filtered['rank_filtered'] = np.arange(len(df_nn_filtered))

        return df_nn_filtered

    def chi2_test(self, df_nn, df = 1, pval = 0.05, FDR = False): #input df_nn_filtered
        '''chi-sqaure left tail test to have enriched pairs'''
        if ('dist' and 'rank') in df_nn.columns:
            dist2 = np.square(np.asarray(df_nn['dist']))
            dist_mean = np.mean(dist2)
            FC = dist2 / dist_mean
            p = scipy.stats.chi2.cdf(FC, df = df) #left tail CDF    
            if FDR:
                from statsmodels.stats.multitest import multipletests
                rej, p, _, _ = multipletests(pvals = p, alpha = pval, method = 'fdr_bh')
                
            df_nn['p_val'] = p
            df_enriched = df_nn[df_nn['p_val'] < pval].sort_values(by=['rank'])
            print(f'\nTotal enriched: {len(df_enriched)} / {len(df_nn)}')
    
            return df_enriched
        else:
            raise NameError('require resulted dataframe with column \'dist\' and \'rank\'')
      
def scores(adata, ref_obj, method = 0, a = 1, n = 100):
    '''cell labels permutation'''
    result = []
    temp = adata.copy()
    
    for _ in range(n):
        labels_pmt = np.random.permutation(temp.obs['ident']) #pmt gloablly
        temp.obs['ident'] = labels_pmt
        pmt_obj = Xct(temp, ref_obj._cell_names[0], ref_obj._cell_names[1], pmt =True)
        df_pmt = pmt_obj.fill_metric(ref_obj = ref_obj)
        result.append(pmt_obj.score(ref_DB = df_pmt, method = method, a = a))
    
    return np.array(result).T
  
  
def pmt_test(orig_score, scores, p = 0.05):
    '''significant result for permutation test'''
    enriched_i, pvals, counts = ([] for _ in range(3))
    for i, dist in enumerate(scores):
        count = sum(orig_score[i] > value for value in dist)
        pval = 1- count/len(dist)
        pvals.append(pval)
        counts.append(count)
        
        if pval < p:
            enriched_i.append(i)           
    
    return enriched_i, pvals, counts
  
  
if __name__ == '__main__':
    ada = sc.datasets.paul15()[:, :100] # raw counts
    ada.obs = ada.obs.rename(columns={'paul15_clusters': 'ident'})
    ada.layers['raw'] = np.asarray(ada.X, dtype=int)
    sc.pp.log1p(ada)
    ada.layers['log1p'] = ada.X.copy()

    obj = Xct(ada, '14Mo', '15Mo', build_GRN = True, save_GRN = True, pcNet_name = 'Net_for_Test', mode = 'full')
    print('building Xct object...')
    print(obj)
    obj_load = Xct(ada, '14Mo', '15Mo', build_GRN = False, pcNet_name = 'Net_for_Test', mode = 'full')
    print('Testing loading...')
    projections, losses = obj.nn_projection(n = 500, plot_loss = False)
    df_nn = obj.nn_output(projections)
    df_enriched = obj.chi2_test(df_nn)
    print(df_enriched.head())


