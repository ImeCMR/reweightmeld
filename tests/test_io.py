import numpy as np
import pytest

from conftest import ALPHA_CHANGE, BLOCK, N_RES, N_STAGES
from reweightmeld.io import MeldRun, read_pdb_frame
from reweightmeld.windows import alpha_change_stages, constant_alpha_segments, frozen_window


def test_reads_layout_across_blocks(meld_run):
    root, t = meld_run
    run = MeldRun(root)
    assert (run.n_replicas, run.n_atoms) == (3, N_RES)
    assert np.array_equal(run.stages, np.arange(N_STAGES))
    st = np.array([3, BLOCK - 1, BLOCK, N_STAGES - 1])             # straddles the block boundary
    assert np.allclose(run.energy_matrices(st), t["E"][..., st].transpose(2, 0, 1))
    assert np.allclose(run.energies(st), np.einsum("kks->sk", t["E"][..., st]))
    assert np.allclose(run.alphas(st), t["alphas"][:, st].T)
    slots = np.array([0, 2, 1, 2])
    assert np.allclose(run.positions(st, slots), t["pos"][slots, :, :, st])


def test_permutation_stage0_is_identity(meld_run):
    root, t = meld_run
    p = MeldRun(root).permutation_vectors([0, 1, 2])
    assert np.array_equal(p[0], np.arange(3))
    assert np.array_equal(p[1:], t["perms"][:, 1:3].T)


def test_missing_stage_raises(meld_run):
    with pytest.raises(KeyError):
        MeldRun(meld_run[0]).alphas([N_STAGES + 5])


def test_windows(meld_run):
    run = MeldRun(meld_run[0])
    assert alpha_change_stages(run).tolist() == [ALPHA_CHANGE]
    w = frozen_window(run, min_start=0)
    assert (w[0], w[-1]) == (ALPHA_CHANGE + 1, N_STAGES - 1)        # first stage after the change dropped
    assert frozen_window(run, min_start=201)[0] == 201
    segs = constant_alpha_segments(run, min_start=1, min_len=50)
    assert [(s[0], s[-1]) for s in segs] == [(1, ALPHA_CHANGE - 1), (ALPHA_CHANGE + 1, N_STAGES - 1)]


def test_topology_and_pdb_frames(meld_run):
    run = MeldRun(meld_run[0])
    top = run.topology()
    assert top.n_residues == N_RES and top.n_atoms == N_RES
    assert read_pdb_frame(run.run_dir / "Data" / "trajectory.pdb", 1).n_atoms == N_RES
    with pytest.raises(ValueError):
        read_pdb_frame(run.run_dir / "Data" / "trajectory.pdb", 5)
    tr = run.trajectory([10, 11], [0, 1])
    assert tr.n_frames == 2 and np.allclose(tr.xyz, run.positions([10, 11], [0, 1]))
