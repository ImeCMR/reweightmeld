"""A synthetic MELD run in MELD's data-store layout (Data/Blocks/block_*.nc + Data/trajectory.pdb).

R = 3 replicas of a 6-residue CA-only chain. The x coordinate of the first atom, measured from its reference
position, is a 1D harmonic coordinate with spring constant KS[k] in state k (kT units):
    energy_matrix[k, j, s] = 0.5 * KS[k] * x[j, s]**2,     energies[j, s] = energy_matrix[j, j, s]
so reweighting to slot 0 must give <x^2> = 1 / KS[0]. Stage 0 has no permutation vector (as in MELD); the
alphas change at stage ALPHA_CHANGE; stages are split over blocks of BLOCK stages, each stage s stored at
time index s of its block.
"""
import netCDF4
import numpy as np
import pytest

KS = np.array([1.0, 1.6, 2.5])
N_STAGES, BLOCK, ALPHA_CHANGE = 1200, 500, 100
N_RES = 6


def _ref_positions():
    """(N_RES, 3) nm: a slightly kinked chain with 0.38 nm CA-CA spacing."""
    return np.array([[0.38 * i, 0.05 * (i % 2), 0.0] for i in range(N_RES)])


def write_pdb(path, xyz_nm):
    """MELD-style trajectory.pdb: no MODEL records, no element column, frames separated by END."""
    with open(path, "w") as fh:
        for f, frame in enumerate(xyz_nm):
            fh.write(f"REMARK stage {f + 1}\n")
            for i, (x, y, z) in enumerate(frame * 10.0):
                fh.write(f"ATOM  {i + 1:5d}  CA  ALA {i + 1:5d}    {x:8.3f}{y:8.3f}{z:8.3f}\n")
            fh.write("TER\nEND\n")


def make_run(root, seed=0, exchange=True):
    rng = np.random.default_rng(seed)
    R = len(KS)
    ref = _ref_positions()
    x = np.stack([rng.normal(0, 1 / np.sqrt(KS[j]), N_STAGES) for j in range(R)])   # (R, S): x in slot j
    E = 0.5 * KS[:, None, None] * x[None] ** 2                                         # (R_k, R_j, S)
    pos = np.broadcast_to(ref, (R, N_STAGES, N_RES, 3)).copy()
    pos += rng.normal(0, 0.02, pos.shape)
    pos[:, :, 0, 0] = ref[0, 0] + x                                                     # harmonic coordinate
    pos = pos.transpose(0, 2, 3, 1)                                                     # (R, A, 3, S)
    alphas = np.tile(np.linspace(0, 1, R)[:, None], (1, N_STAGES))
    alphas[:, :ALPHA_CHANGE] *= 0.9
    perms = np.tile(np.arange(R)[:, None], (1, N_STAGES))
    if exchange:                                  # random neighbour swaps (any valid permutation will do)
        for s in range(N_STAGES):
            i = rng.integers(R - 1)
            perms[[i, i + 1], s] = perms[[i + 1, i], s]
    data = root / "Data" / "Blocks"
    data.mkdir(parents=True)
    for b, lo in enumerate(range(0, N_STAGES, BLOCK)):
        hi = min(lo + BLOCK, N_STAGES)
        with netCDF4.Dataset(data / f"block_{b:06d}.nc", "w") as d:
            d.createDimension("n_replicas", R)
            d.createDimension("n_atoms", N_RES)
            d.createDimension("cartesian", 3)
            d.createDimension("timesteps", None)
            d.createDimension("n_discrete_parameters", 1)
            v = {n: d.createVariable(n, t, dims) for n, t, dims in (
                ("positions", "f8", ("n_replicas", "n_atoms", "cartesian", "timesteps")),
                ("alphas", "f8", ("n_replicas", "timesteps")),
                ("energies", "f8", ("n_replicas", "timesteps")),
                ("permutation_vectors", "i8", ("n_replicas", "timesteps")),
                ("energy_matrix", "f8", ("n_replicas", "n_replicas", "timesteps")),
                ("discrete_parameters", "i8", ("n_replicas", "n_discrete_parameters", "timesteps")))}
            v["positions"][..., lo:hi] = pos[..., lo:hi]
            v["alphas"][:, lo:hi] = alphas[:, lo:hi]
            v["energies"][:, lo:hi] = np.einsum("kks->ks", E[..., lo:hi])
            v["energy_matrix"][..., lo:hi] = E[..., lo:hi]
            v["discrete_parameters"][..., lo:hi] = 5
            p_lo = max(lo, 1)                     # stage 0: no permutation vector (masked)
            v["permutation_vectors"][:, p_lo:hi] = perms[:, p_lo:hi]
    write_pdb(root / "Data" / "trajectory.pdb", [ref, ref])
    return {"x": x, "E": E, "pos": pos, "alphas": alphas, "perms": perms}


@pytest.fixture(scope="session")
def meld_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("meld_run")
    truth = make_run(root)
    return root, truth
