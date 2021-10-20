import pytest

import numpy as np
import scanpy as sc

from scTenifoldXct.main import Xct
from scTenifoldXct.main import get_counts_np


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
    return xct.get_candidates(df1)


@pytest.fixture(scope="session")
def counts_np(xct):
    return get_counts_np(xct)