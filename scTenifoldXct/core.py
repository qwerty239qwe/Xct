from typing import List
import os
from os import PathLike
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from anndata._core.views import ArrayView
import anndata
import scipy
from scipy import sparse

from scTenifold import cal_pcNet
from .nn import ManifoldAlignmentNet
from .stat import null_test, chi2_test

sc.settings.verbosity = 0


class GRN:
    def __init__(self,
                 name: str = None,
                 data: anndata.AnnData = None,
                 GRN_file_dir: PathLike = None,
                 rebuild_GRN: bool = False,
                 verbose: bool = True,
                 **kwargs):
        self.kws = kwargs
        if GRN_file_dir is not None:
            self._pc_net_file_name = (Path(GRN_file_dir) / Path(f"pcnet_{name}"))
        # load pcnet
        if rebuild_GRN:
            if verbose:
                print(f'building GRN {name}...')
            self._net = cal_pcNet(data.to_df().T, nComp=5, symmetric=True, **kwargs)
            self._gene_names = data.var_names.copy(deep=True)
            if verbose:
                print(f'GRN of {name} has been built')

            if GRN_file_dir is not None:
                os.makedirs('./data', exist_ok = True) # create dir 'data'
                sparse.save_npz(self._pc_net_file_name, self._net)
        else:
            if verbose:
                print(f'loading GRN {name}...')
            if GRN_file_dir is not None:
                self._gene_names = pd.Index(pd.read_csv(Path(GRN_file_dir) / Path("gene_name.tsv"), sep='\t')["gene_name"])
                self._net = sparse.load_npz(self._pc_net_file_name)

    @classmethod
    def from_sparse(cls, name, sparse_matrix, gene_names):
        obj = cls(name)
        obj.set_value(sparse_matrix, gene_names)
        return obj

    @classmethod
    def load(cls, dir_name, pcnet_name):
        return cls(name=pcnet_name, GRN_file_dir=dir_name)

    @property
    def net(self) -> sparse.coo_matrix:
        return self._net

    @property
    def gene_names(self):
        return self._gene_names

    def set_value(self, sparse_matrix: sparse.coo_matrix, gene_names):
        if sparse_matrix.shape[0] != sparse_matrix.shape[1]:
            raise ValueError("sparse_matrix should be a square sparse matrix"
                             f"({sparse_matrix.shape[0]} != {sparse_matrix.shape[1]})")
        if sparse_matrix.shape[0] != len(gene_names):
            raise ValueError(f"gene_names should have the same length as the sparse_matrix "
                             f"({sparse_matrix.shape[0]} != {len(gene_names)})")

        self._net = sparse_matrix
        self._gene_names = gene_names

    def save(self, dir_name):
        Path(dir_name).mkdir(parents=True, exist_ok=True)
        sparse.save_npz(self._pc_net_file_name, self._net)
        pd.DataFrame({"gene_name": self._gene_names}).to_csv(Path(dir_name) / Path("gene_name.tsv"), sep='\t')

    def subset_in(self, values):
        bool_ind = self._gene_names.isin(values)
        self._net = self._net.tocsr()[bool_ind, bool_ind]
        self._gene_names = self._gene_names[bool_ind]

    def concat(self, grn, axis=0):
        if axis not in [0, 1]:
            raise ValueError("axis should be either 0 or 1.")
        concat_method = sparse.vstack if axis == 0 else sparse.hstack
        self._net = concat_method([self._net, grn.net])
        self._gene_names = pd.Index(pd.concat([self.gene_names.to_series(), grn.gene_names.to_series()]))


def concat_grns(grns, axis=0):
    """
    concat multiple GRNs
    :param grns:
    :param axis:
    :return:
    """
    if axis not in [0, 1]:
        raise ValueError("axis should be either 0 or 1.")
    concat_method = sparse.vstack if axis == 0 else sparse.hstack
    obj = GRN()
    obj.set_value(concat_method([grn.net for grn in grns]),
                  gene_names=pd.Index(pd.concat([grn.gene_names.to_series() for grn in grns])))
    return obj


class scTenifoldXct:
    def __init__(self,
                 data,
                 cell_names: List[str],
                 obs_label: str,  # ident
                 species: str,
                 GRN_file_dir: PathLike = None,
                 rebuild_GRN: bool = False,
                 query_DB: str = None,
                 alpha: float = 0.55,
                 mu: float = 1.,
                 scale_w: bool = True,
                 n_dim: int = 3,
                 verbose=True):

        if species.lower() not in ["human", "mouse"]:
            raise ValueError("species must be human or mouse")

        if query_DB is not None and query_DB not in ['comb', 'pairs']:
            raise ValueError('queryDB using the keyword None, \'comb\' or \'pairs\'')

        self._metrics = ["mean", "var"]
        self.verbose = verbose
        self._cell_names = cell_names
        self._cell_data_dic, self._cell_metric_dict = {}, {}
        self._genes = {}
        for name in self._cell_names:
            self.load_data(data, name, obs_label)
        self._species = species
        self._LRs = self._load_db_data(Path(__file__).parent.parent / Path("data/LR.csv"),
                                       ['ligand', 'receptor'])
        # self._TFs = self._load_db_data() # is this an unused db data?

        # fill metrics
        self._LR_metrics = self.fill_metric()
        self._candidates = self._get_candidates(self._LR_metrics)

        self._net_A = GRN(name=self._cell_names[0],
                          data=self._cell_data_dic[self._cell_names[0]],
                          GRN_file_dir=GRN_file_dir,
                          rebuild_GRN=rebuild_GRN,
                          verbose=verbose)
        self._net_B = GRN(name=self._cell_names[1],
                          data=self._cell_data_dic[self._cell_names[1]],
                          GRN_file_dir=GRN_file_dir,
                          rebuild_GRN=rebuild_GRN,
                          verbose=verbose)
        if verbose:
            print('building correspondence...')

        # cal w
        self._w, w12_shape = self._build_w(alpha=alpha,
                                           query_DB=query_DB,
                                           scale_w=scale_w,
                                           mu=mu)

        self._nn_trainer = ManifoldAlignmentNet(self._get_data_arr(),
                                                w=self._w,
                                                n_dim=n_dim,
                                                layers=None,
                                                verbose=verbose)

        self._aligned_result = self._nn_trainer.nn_aligned_dist(gene_names_x=self._genes[self._cell_names[0]],
                                                                gene_names_y=self._genes[self._cell_names[1]],
                                                                w12_shape=w12_shape)

    @property
    def nn_trainer(self):
        return self._nn_trainer

    @property
    def aligned_dist(self):
        return self._aligned_result

    def knock_out(self, ko_gene_list):
        gene_idx = pd.concat([self._genes[self._cell_names[0]].to_series(),
                              self._genes[self._cell_names[1]].to_series()])
        assert len(gene_idx) == self._net_A.net.shape[0] == self._net_B.net.shape[0]

        bool_idx = gene_idx.isin(ko_gene_list)
        self._w = self._w.tolil()
        self._w[bool_idx, :] = 0
        self._w[:, bool_idx] = 0
        self._w = self._w.tocoo()

    def load_data(self, data, cell_name, obs_label):
        if isinstance(data, anndata.AnnData):
            self._genes[cell_name] = data.var_names
            self._cell_data_dic[cell_name] = data[data.obs[obs_label] == cell_name, :]
            self._cell_metric_dict[cell_name] = {}
            self._cell_metric_dict[cell_name] = self._get_metric(self._cell_data_dic[cell_name], cell_name)

    def _load_db_data(self, file_path, subsets):
        df = pd.read_csv(file_path)
        df = df.loc[:, subsets] if subsets is not None else df
        if self._species == "mouse":
            for c in df.columns:
                df[c] = df[c].str.capitalize()

        return df

    def _get_metric(self, adata: ArrayView, name):  # require normalized data
        '''compute metrics for each gene'''
        data_norm = adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X.copy()  # adata.layers['log1p']
        if self.verbose:
            print('(cell, feature):', data_norm.shape)
        if (data_norm % 1 != 0).any():  # check space: True for log (float), False for counts (int)
            mean = np.mean(data_norm, axis=0)  # .toarray()
            var = np.var(data_norm, axis=0)  # .toarray()
            return {"mean": dict(zip(self._genes[name], mean)),
                    "var": dict(zip(self._genes[name], var))} # , dispersion, cv
        raise ValueError("require log data")

    def fill_metric(self):
        val_df = pd.DataFrame()
        for c in self._LRs.columns:
            for m in self._metrics:
                val_df[f"{m}_L"] = self._LRs[c].map(self._cell_metric_dict[self._cell_names[0]]).fillna(0.)
                val_df[f"{m}_R"] = self._LRs[c].map(self._cell_metric_dict[self._cell_names[1]]).fillna(0.)
        df = pd.concat([self._LRs, val_df], axis=1)  # concat 1:1 since sharing same index
        df = df[(df['mean_L'] > 0) & (df['mean_R'] > 0)]  # filter 0 (none or zero expression) of LR
        if self.verbose:
            print('Selected {} LR pairs'.format(df.shape[0]))

        return df

    def _get_candidates(self, df_filtered):
        '''selected L-R candidates'''
        candidates = [a + '_' + b for a, b in zip(np.asarray(df_filtered['ligand'], dtype=str),
                                                  np.asarray(df_filtered['receptor'], dtype=str))]
        return candidates

    @staticmethod
    def _zero_out_w(w, mask_lig, mask_rec):
        w = w.tolil()
        w[mask_lig, :] = 0
        w[:, mask_rec] = 0
        assert np.count_nonzero(w) == sum(mask_lig) * sum(mask_rec)
        return w.tocoo()

    def _build_w(self, alpha, query_DB=None, scale_w=True, mu: float = 1.) -> (sparse.coo_matrix, (int, int)):
        '''build w: 3 modes, default None will not query the DB and use all pair-wise corresponding scores'''
        # (1-a)*u^2 + a*var
        ligand, receptor = self._cell_names[0], self._cell_names[1]

        metric_a_temp = ((1 - alpha) * np.square(self._cell_metric_dict[ligand]["mean"]) +
                         alpha * (self._cell_metric_dict[ligand]["var"]))[:, None]
        metric_b_temp = ((1 - alpha) * np.square(self._cell_metric_dict[receptor]["mean"]) +
                         alpha * (self._cell_metric_dict[receptor]["var"]))[:, None]

        # print(metric_A_temp.shape, metric_B_temp.shape)

        # make it sparse to reduce mem usage
        w12 = sparse.coo_matrix(metric_a_temp @ metric_b_temp)
        del metric_a_temp
        del metric_b_temp

        if scale_w:
            w12_orig_sum = w12.sum()
        if query_DB is not None:
            if query_DB == 'comb':
                # ada.var index of LR genes (the intersect of DB and object genes, no pair relationship maintained)
                used_row_index = np.isin(self._genes[ligand], self._LRs["ligand"])
                used_col_index = np.isin(self._genes[receptor], self._LRs["receptor"])
            elif query_DB == 'pairs':
                # maintain L-R pair relationship, both > 0
                selected_LR = self._LR_metrics[(self._LR_metrics[f"mean_L"] > 0) & (self._LR_metrics[f"mean_R"] > 0)]
                used_row_index = np.isin(self._genes[ligand], selected_LR["ligand"])
                used_col_index = np.isin(self._genes[receptor], selected_LR["receptor"])
            else:
                raise ValueError("queryDB must be: [None, \'comb\' or \'pairs\']")

            w12 = self._zero_out_w(w12, used_row_index, used_col_index)
        if scale_w:
            w12 = mu * ((self._net_A.net.sum() + self._net_A.net.shape[0] * self._net_A.net.shape[1]) +
                        (self._net_B.net.sum() + self._net_B.net.shape[0] * self._net_B.net.shape[1])) / (
                        2 * w12_orig_sum) * w12  # scale factor using w12_orig

        if self.verbose:
            print(f"Trying to concatenate pcnet using {self._net_A}")
        w = sparse.vstack([sparse.hstack([self._net_A.net + 1, w12]),
                           sparse.hstack([w12.T, self._net_B.net + 1])])

        return w, w12.shape

    def _get_data_arr(self):  # change the name (not always count data)
        '''return a list of counts in numpy array, gene by cell'''
        data_arr = [cell_data.X.T.toarray() if scipy.sparse.issparse(cell_data.X) else cell_data.X.T   # gene by cell
                     for _, cell_data in self._cell_data_dic.items()]
        return data_arr  # a list

    def train_nn(self,
                 n_steps=1000,
                 lr=0.01,
                 verbose=True,
                 plot_losses: bool = True,
                 losses_file_name: str = None,
                 **optim_kwargs
                 ):
        projections, losses = self._nn_trainer.train(n_steps=n_steps, lr=lr, verbose=verbose, **optim_kwargs)
        if plot_losses:
            self._nn_trainer.plot_losses(losses_file_name)

        return projections, losses

    def plot_losses(self, **kwargs):
        self._nn_trainer.plot_losses(**kwargs)

    def add_names_to_nets(self):

        # TODO: modify this -> plot_pcnet
        '''for graph visualization'''
        self._net_A = pd.DataFrame(self._net_A, columns = self.genes_names[0], index = self.genes_names[0])
        self._net_B = pd.DataFrame(self._net_B, columns = self.genes_names[1], index = self.genes_names[1])
        print('completed.')

    def null_test(self,
                  filter_zeros: bool = True,
                  pct=0.05,
                  plot_result=False):
        return null_test(self._aligned_result, self._candidates,
                         filter_zeros=filter_zeros,
                         pct=pct,
                         plot=plot_result)

    def chi2_test(self,
                  dof=1,
                  pval=0.05,
                  cal_FDR=True,
                  plot_result=False,
                  ):
        return chi2_test(df_nn=self._aligned_result, df=dof,
                         pval=pval,
                         FDR=cal_FDR,
                         candidates=self._candidates,
                         plot=plot_result)


if __name__ == '__main__':
    import nn
    import scanpy as sc

    ada = sc.datasets.paul15()[:, :100] # raw counts
    ada.obs = ada.obs.rename(columns={'paul15_clusters': 'ident'})
    ada.layers['raw'] = np.asarray(ada.X, dtype=int)
    sc.pp.log1p(ada)
    ada.layers['log1p'] = ada.X.copy()

    # obj = Xct(ada, '14Mo', '15Mo', specis="Mouse", build_GRN = True, save_GRN = True, pcNet_name = 'Net_for_Test', queryDB = None, verbose = True)
    # print(obj)
    # obj_load = Xct(ada, '14Mo', '15Mo', build_GRN = False, pcNet_name = 'Net_for_Test', queryDB = None, verbose = True)
    # print('Testing loading...')

    # df1 = obj.fill_metric()
    # candidates = get_candidates(df1)
    # counts_np = get_counts_np(obj)
    # projections, losses = nn.train_and_project(counts_np, obj._w, dim = 2, steps = 1000, lr = 0.001)

    # df_nn = nn_aligned_dist(obj, projections)
    # df_enriched = chi2_test(df_nn, df = 1, FDR = False, candidates = candidates)
    # print(df_enriched.head())

