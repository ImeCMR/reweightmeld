"""MBAR / BayesMBAR reweighting of a MELD H,T-REMD window to one ladder slot (the target state).

States = the R ladder slots (constant alphas). Sample n = (stage s, slot j) with
    u_kn[k, n] = energy_matrix[s][k, j]      (kT; includes minus MELD's log-prior of the discrete parameters,
                                              so this is MBAR on the extended (x, y) space)
All stages of the window are used (MBAR stays consistent on correlated samples). Correlation is measured along
walkers and used for the walker-corrected ESS and to thin the BayesMBAR input.
"""
import logging

import numpy as np
from pymbar import MBAR

from reweightmeld.walkers import walker_ids, walker_inefficiency

log = logging.getLogger(__name__)


def build_u_kn(E, stages, keep=None):
    """E: (S, R, R) energy matrices of `stages`; keep: optional per-slot indices into `stages`.
    Returns u_kn (R, N), N_k (R,), sample stage (N,), sample slot (N,); samples are ordered by slot."""
    S, R, _ = E.shape
    cols, st, sl, N_k = [], [], [], np.zeros(R, dtype=int)
    for j in range(R):
        idx = np.arange(S) if keep is None else np.asarray(keep[j])
        cols.append(E[idx, :, j].T)                         # (R, n_j): structure in slot j in every state k
        st.append(np.asarray(stages)[idx])
        sl.append(np.full(len(idx), j))
        N_k[j] = len(idx)
    return np.concatenate(cols, axis=1), N_k, np.concatenate(st), np.concatenate(sl)


def kish_ess(w):
    w = np.asarray(w, dtype=float)
    return float(w.sum() ** 2 / (w ** 2).sum())


def run_mbar(u_kn, N_k, target=0):
    mbar = MBAR(u_kn, N_k, solver_protocol="robust")
    W = mbar.weights()                                       # (N, K), columns normalised
    overlap = mbar.compute_overlap()["matrix"]
    return {"f_k": mbar.f_k.copy(), "w": W[:, target].copy(), "overlap": overlap}


def _logsumexp(a, axis):
    m = a.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(a - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def weights_from_f(u_kn, N_k, f_k, target=0):
    """MBAR weights of all samples in state `target` for given free energies f_k."""
    log_denom = _logsumexp(np.log(N_k)[:, None] + f_k[:, None] - u_kn, axis=0)
    logw = -u_kn[target] - log_denom
    logw -= logw.max()
    w = np.exp(logw)
    return w / w.sum()


def bayes_available():
    try:
        import bayesmbar  # noqa: F401
        return True
    except ImportError:
        return False


def run_bayesmbar(u_kn, N_k, n_posterior=200, target=0, seed=0):
    """Posterior over f_k with BayesMBAR (uniform prior). None if BayesMBAR is not installed."""
    try:
        from bayesmbar import BayesMBAR
    except ImportError:
        log.warning("BayesMBAR not installed (pip install 'reweightmeld[bayes]'); skipping posterior weights")
        return None
    bm = BayesMBAR(u_kn, N_k, prior="uniform", sample_size=n_posterior, random_seed=seed, verbose=False)
    F = np.asarray(bm.F_samples)[:n_posterior]               # (n_post, K)
    return {"F_samples": F, "f_mean": np.asarray(bm.F_mean),
            "f_std": np.asarray(bm.F_std)[target]}            # F_std: K x K matrix of std(F_i - F_j)


def reweight_window(run, stages, target=0, bayes=False, n_posterior=200, seed=0):
    """MBAR weights of every (stage, slot) frame of a constant-alpha window in slot `target`.
    Returns a dict of arrays and diagnostics (see reweightmeld.ensemble.WindowResult)."""
    stages = np.asarray(stages, dtype=int)
    a = run.alphas(stages)
    if np.abs(a - a[0]).max() > 0:
        raise ValueError("alphas are not constant over the requested stage window")
    E = run.energy_matrices(stages)
    if not np.allclose(np.einsum("skk->sk", E), run.energies(stages)):
        raise ValueError("energy_matrix diagonal != energies: unexpected MELD data layout")
    wid = walker_ids(run.permutation_vectors(stages))
    g = walker_inefficiency(E, wid, target)
    g_med = float(np.median(g))
    u_kn, N_k, st, sl = build_u_kn(E, stages)
    res = run_mbar(u_kn, N_k, target)
    w = res["w"]
    out = {
        "stages": stages, "stage": st, "slot": sl, "weight": w, "N_k": N_k, "f_k": res["f_k"],
        "alphas": a[0], "overlap_offdiag": np.diag(res["overlap"], 1), "g_walker": g,
        "ess_kish": kish_ess(w),
        "ess": kish_ess(w) / g_med,                          # Kish ESS / median walker inefficiency
        "n_walkers_in_target": int(len(np.unique(wid[:, target]))),
        "frac_weight_target": float(w[sl == target].sum()),
        "u_target_reweighted": float((w * u_kn[target]).sum()),   # should match the plain target-slot mean
        "u_target_direct": float(u_kn[target][sl == target].mean()),
    }
    if bayes:
        stride = max(1, int(np.ceil(g_med)))
        thin = [np.arange(0, len(stages), stride)] * E.shape[1]
        u_thin, N_thin, _, _ = build_u_kn(E, stages, thin)
        b = run_bayesmbar(u_thin, N_thin, n_posterior=n_posterior, target=target, seed=seed)
        if b is not None:
            W = np.stack([weights_from_f(u_kn, N_k, f, target) for f in b["F_samples"]])
            out.update(weight_bayes=W.mean(0), weight_bayes_std=W.std(0), f_k_bayes=b["f_mean"],
                       f_k_bayes_std=b["f_std"], bayes_stride=stride)
    return out
