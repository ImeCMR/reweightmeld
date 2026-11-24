import json

import numpy as np
import pytest

from conftest import ALPHA_CHANGE, KS, N_STAGES
from reweightmeld import MeldRun, Reweighting, reweight


@pytest.fixture(scope="module")
def rw(meld_run):
    return reweight(MeldRun(meld_run[0]), target=0, start_stage=1, log=None)


def test_reweight_recovers_target(meld_run, rw):
    root, t = meld_run
    x = t["x"][rw.slot, rw.stage]
    assert abs((rw.weight * x ** 2).sum() - 1 / KS[0]) < 0.08
    assert rw.weight.sum() == pytest.approx(1.0)
    w = rw.windows[0]
    assert (w["stages"][0], w["stages"][-1]) == (ALPHA_CHANGE + 1, N_STAGES - 1)
    assert w["u_target_reweighted"] == pytest.approx(w["u_target_direct"], abs=0.05)
    assert w["f_k"][1] - w["f_k"][0] == pytest.approx(0.5 * np.log(KS[1] / KS[0]), abs=0.05)
    assert rw.perm_stages[0] == 1 and len(rw.perm_stages) == N_STAGES - 1


def test_segments_pool(meld_run):
    rs = reweight(MeldRun(meld_run[0]), start_stage=1, segments=True, min_segment_len=50, log=None)
    assert len(rs.windows) == 2 and rs.mix.sum() == pytest.approx(1.0)
    assert rs.weight.sum() == pytest.approx(1.0)
    assert sorted(np.unique(rs.segment).tolist()) == [0, 1]


def test_window_with_alpha_change_refused(meld_run):
    from reweightmeld.mbar import reweight_window
    with pytest.raises(ValueError):
        reweight_window(MeldRun(meld_run[0]), np.arange(50, 150))


def test_save_load_roundtrip(tmp_path, rw):
    files = rw.save(tmp_path / "sys")
    assert all((tmp_path / f.split("/")[-1]).exists() for f in files)
    back = Reweighting.load(tmp_path / "sys")
    for k in ("stage", "slot", "segment", "weight", "perm_stages", "perms", "walker_ids"):
        assert np.array_equal(getattr(back, k), getattr(rw, k)), k
    assert np.allclose(back.windows[0]["weight"], rw.windows[0]["weight"])
    assert np.allclose(back.windows[0]["g_walker"], rw.windows[0]["g_walker"])
    s = json.loads((tmp_path / "sys.summary.json").read_text())
    assert s["windows"][0]["ess"] == pytest.approx(rw.windows[0]["ess"])


def test_resample_draws_by_weight(meld_run, rw):
    run = MeldRun(meld_run[0])
    ens, idx = rw.resample(run, 50, seed=3)
    assert ens.n_frames == 50
    assert np.allclose(ens.xyz, run.positions(rw.stage[idx], rw.slot[idx]))
    assert np.array_equal(idx, rw.draw(50, seed=3))
