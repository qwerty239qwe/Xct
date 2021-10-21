import pytest

from scTenifoldXct.main import get_counts_np
from scTenifoldXct.dNN import ManifoldAlignmentNet


@pytest.fixture(scope="package")
def ma_net(xct, counts_np):
    ma = ManifoldAlignmentNet(counts_arr=counts_np, w=xct._w, layers=None, n_dim=3, verbose=True)
    ma.train(n_steps=100)
    ma.plot_losses(file_name="temp.png")
    return ma


@pytest.fixture(scope="package")
def ma_net_skin(xct_skin):
    counts = get_counts_np(xct_skin)
    ma = ManifoldAlignmentNet(counts_arr=counts, w=xct_skin._w, layers=None, n_dim=3, verbose=True)
    ma.train(n_steps=100)
    ma.plot_losses(file_name="temp.png")
    return ma