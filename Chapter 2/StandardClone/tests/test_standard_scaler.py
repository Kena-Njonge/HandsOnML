import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler as SkSS

from package.standard_scaler import StandardScaler 

def test_against_sklearn():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(5, 4)).astype(float)      

    mine = StandardScaler().fit(X.copy())
    ref  = SkSS().fit(X.copy())

    # means and stds should match
    assert np.allclose(np.array(mine.means), ref.mean_, atol=1e-9)
    assert np.allclose(np.array(mine.stds),  ref.scale_, atol=1e-9)

    # transformed matrices should be the same
    Z_mine = mine.transform(X)
    Z_ref  = ref.transform(X)
    assert np.allclose(Z_mine, Z_ref, atol=1e-9)

    # inverse-transform round-trip
    X_hat = mine.inverse_transform(Z_mine)
    assert np.allclose(X_hat, X, atol=1e-9)
