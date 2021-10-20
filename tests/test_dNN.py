import pytest

from scTenifoldXct.dNN import ManifoldAlignmentNet


def test_ManifoldAlignmentNet(xct, counts_np):
    ma = ManifoldAlignmentNet(counts_arr=counts_np, w=xct._w, layers=None, n_dim=3, verbose=True)
    ma.train(n_steps=100)
    ma.plot_losses(file_name="temp.png")