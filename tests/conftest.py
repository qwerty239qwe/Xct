import pytest

import numpy as np
import scanpy as sc
import scipy

from scTenifoldXct.core import Xct, get_candidates
from scTenifoldXct.core import get_counts_np


@pytest.fixture(scope="session")
def ada_skin():
    ada = sc.read_h5ad("../data/LS.h5ad")
    data = scipy.sparse.csr_matrix.toarray(ada.X)
    counts = np.asarray(np.expm1(data), dtype=int)
    ada.layers['raw'] = counts
    ada.layers['log1p'] = data
    HVG_i = np.argsort(np.asarray(ada.var['vst.variance.standardized']))[-3000:]
    return ada[:, HVG_i]


@pytest.fixture(scope="session")
def xct_skin(ada_skin):
    return Xct(ada_skin, 'Inflam. FIB', 'Inflam. DC', build_GRN = False, pcNet_name = 'skin_net', verbose = True)


@pytest.fixture(scope="session")
def df_skin(xct_skin):
    return xct_skin.fill_metric()


@pytest.fixture()
def candidates_skin(df_skin):
    candidates = get_candidates(df_skin)
    return candidates


# small dataset
@pytest.fixture(scope="session")
def xct():
    ada = sc.datasets.paul15()[:, :100]  # raw counts
    ada.obs = ada.obs.rename(columns={'paul15_clusters': 'ident'})
    ada.layers['raw'] = np.asarray(ada.X, dtype=int)
    sc.pp.log1p(ada)
    ada.layers['log1p'] = ada.X.copy()
    return Xct(ada, '14Mo', '15Mo', build_GRN=True, save_GRN=False, pcNet_name='Net_for_Test', queryDB=None, verbose=True)


@pytest.fixture(scope="session")
def candidates(xct):
    df1 = xct.fill_metric()
    return get_candidates(df1)


@pytest.fixture(scope="session")
def counts_np(xct):
    return get_counts_np(xct)