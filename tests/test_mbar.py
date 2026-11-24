import numpy as np
import pytest

from reweightmeld.mbar import bayes_available, build_u_kn, kish_ess, run_bayesmbar, run_mbar, weights_from_f
from reweightmeld.walkers import walker_ids


def _harmonic(ks, n, seed):
    rng = np.random.default_rng(seed)
    x = [rng.normal(0, 1 / np.sqrt(k), n) for k in ks]
    E = np.stack([[[0.5 * ks[k] * x[j][s] ** 2 for j in range(len(ks))] for k in range(len(ks))] for s in range(n)])
    return x, E


def test_mbar_harmonic_two_states():
    """Two 1D harmonic states (k = 1 and 4, kT): reweighting to state 0 gives <x^2>_0 = 1 and
    f_1 - f_0 = 0.5 ln 4."""
    ks = np.array([1.0, 4.0])
    x, E = _harmonic(ks, 20000, 0)
    u_kn, N_k, st, sl = build_u_kn(E, np.arange(len(E)))
    res = run_mbar(u_kn, N_k)
    assert abs((res["f_k"][1] - res["f_k"][0]) - 0.5 * np.log(4.0)) < 0.02
    xs = np.concatenate(x)
    assert abs((res["w"] * xs ** 2).sum() - 1.0) < 0.03
    assert np.allclose(weights_from_f(u_kn, N_k, res["f_k"]), res["w"], atol=1e-10)
    assert sl.tolist() == [0] * 20000 + [1] * 20000


def test_build_u_kn_columns_and_thinning():
    E = np.arange(4 * 2 * 2, dtype=float).reshape(4, 2, 2)      # (S, k, j)
    u_kn, N_k, st, sl = build_u_kn(E, np.arange(10, 14), keep=[[0, 2], [1, 3]])
    assert N_k.tolist() == [2, 2]
    assert st.tolist() == [10, 12, 11, 13] and sl.tolist() == [0, 0, 1, 1]
    assert np.allclose(u_kn[:, 0], E[0, :, 0]) and np.allclose(u_kn[:, 3], E[3, :, 1])


def test_kish_ess():
    assert kish_ess(np.ones(10)) == pytest.approx(10)
    assert kish_ess(np.array([1.0, 0, 0])) == pytest.approx(1)


def test_walker_ids_follow_meld_convention():
    perms = np.array([[1, 0, 2], [0, 2, 1]])
    assert walker_ids(perms).tolist() == [[1, 0, 2], [1, 2, 0]]


@pytest.mark.skipif(not bayes_available(), reason="BayesMBAR not installed")
def test_bayesmbar_agrees_with_mbar():
    ks = np.array([1.0, 2.0, 4.0])
    _, E = _harmonic(ks, 1000, 1)
    u_kn, N_k, _, _ = build_u_kn(E, np.arange(len(E)))
    m = run_mbar(u_kn, N_k)
    b = run_bayesmbar(u_kn, N_k, n_posterior=200)
    df_m, df_b = m["f_k"] - m["f_k"][0], b["f_mean"] - b["f_mean"][0]
    assert np.allclose(df_m, df_b, atol=3 * b["f_std"].max() + 1e-3)
    W = np.stack([weights_from_f(u_kn, N_k, f) for f in b["F_samples"]])
    assert np.allclose(W.mean(0), m["w"], atol=5e-4)
