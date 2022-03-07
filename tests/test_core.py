import pytest
from scTenifoldXct.core import GRN


def test_xct_obj_attrs(xct_skin):
    assert isinstance(xct_skin.net_A, GRN), "net_A should be a GRN object"
    assert isinstance(xct_skin.net_B, GRN), "net_B should be a GRN object"
    assert xct_skin.net_B.shape == xct_skin.net_A.shape


@pytest.fixture(scope="session")
def df_nn_skin(xct_skin):
    xct_skin.train_nn(n_steps= 1000, lr = 0.001)
    return xct_skin.null_test(pct=0.025, plot_result=True)


@pytest.fixture(scope="session")
def df_nn_paul15(xct_paul15):
    xct_paul15.train_nn(n_steps= 1000, lr = 0.001)
    return xct_paul15.null_test(pct=0.025, plot_result=True)