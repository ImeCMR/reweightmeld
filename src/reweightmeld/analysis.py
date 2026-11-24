"""Structure analysis of (reweighted) ensembles: CA-distance PCA, conformational states, fluctuations, contacts.

All functions take mdtraj trajectories; distances in Angstrom. The CA-distance features (|i - j| >= 3) are
invariant to rotation and translation, so no superposition is needed for the PCA.
"""
import numpy as np
import mdtraj


def ca(traj):
    return traj.atom_slice(traj.topology.select("name CA"))


def ca_distances(traj, min_sep=3):
    L = ca(traj).n_atoms
    pairs = np.array([(i, j) for i in range(L) for j in range(i + min_sep, L)])
    return mdtraj.compute_distances(ca(traj), pairs) * 10.0


def pca_fit(X, n_comp=10):
    """(mean, components (D, n_comp), explained-variance fractions) of the rows of X."""
    mu = X.mean(0)
    _, s, Vt = np.linalg.svd(X - mu, full_matrices=False)
    ev = s ** 2 / (s ** 2).sum()
    return mu, Vt[:n_comp].T, ev[:n_comp]


def kmeans(X, k, seed=0, n_init=10, iters=100):
    """Plain k-means with k-means++ starts, best of n_init runs. Returns (labels, centres)."""
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_init):
        C = X[[rng.integers(len(X))]]
        for _ in range(1, k):
            d = ((X[:, None] - C[None]) ** 2).sum(-1).min(1)
            C = np.vstack([C, X[rng.choice(len(X), p=d / d.sum())]])
        for _ in range(iters):
            lab = ((X[:, None] - C[None]) ** 2).sum(-1).argmin(1)
            Cn = np.array([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
            if np.allclose(Cn, C):
                break
            C = Cn
        inertia = float(((X - C[lab]) ** 2).sum())
        if best is None or inertia < best[0]:
            best = (inertia, lab, C)
    return best[1], best[2]


def states(sets, reference, n_states=5, n_pc=10, seed=0):
    """PCA fitted on sets[reference], k-means states on its first n_pc PCs; every set projected and assigned.

    sets: {name: trajectory}. Returns dict with P (name -> (n, n_pc) projections), labels (name -> state),
    centres, explained (variance fractions), representatives (frame index in the reference set per state)."""
    X = {k: ca_distances(t) for k, t in sets.items()}
    mu, V, ev = pca_fit(X[reference], n_pc)
    P = {k: (x - mu) @ V for k, x in X.items()}
    lab, cen = kmeans(P[reference], n_states, seed=seed)
    labels = {k: ((p[:, None] - cen[None]) ** 2).sum(-1).argmin(1) for k, p in P.items()}
    labels[reference] = lab
    reps = []
    for j in range(n_states):
        idx = np.where(lab == j)[0]
        reps.append(int(idx[np.argmin(((P[reference][idx] - cen[j]) ** 2).sum(1))]) if len(idx) else -1)
    return {"P": P, "labels": labels, "centres": cen, "explained": ev, "representatives": reps,
            "populations": {k: np.bincount(l, minlength=n_states) / len(l) for k, l in labels.items()}}


def rmsf_ca(traj, reference_frame):
    """CA RMSF (A) after superposition on a reference frame (single-frame trajectory)."""
    c = ca(traj)
    c.superpose(ca(reference_frame))
    return mdtraj.rmsf(c, None) * 10.0


def radius_of_gyration(traj):
    return mdtraj.compute_rg(traj.atom_slice(traj.topology.select("not element H"))) * 10.0


def contact_probability(traj, cutoff=8.0):
    """(L, L) upper-triangular probability that CA(i) - CA(j) < cutoff A."""
    c = ca(traj)
    L = c.n_atoms
    pairs = np.array([(i, j) for i in range(L) for j in range(i + 1, L)])
    d = mdtraj.compute_distances(c, pairs) * 10.0
    M = np.zeros((L, L))
    M[pairs[:, 0], pairs[:, 1]] = (d < cutoff).mean(0)
    return M
