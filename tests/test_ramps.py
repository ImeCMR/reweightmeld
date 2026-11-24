"""Restraint time ramps read from a MELD-like Data/system.dat pickle, without MELD installed."""
import pickle
import shutil
import sys
import types
import warnings

import numpy as np
import pytest

from reweightmeld.ramps import time_ramps

FAKE = ("meld", "meld.system", "meld.system.scalers", "meld.system.restraints", "openmm")


def _fake_classes():
    """Classes with MELD / OpenMM module paths, registered so pickle can refer to them."""
    mods = {m: types.ModuleType(m) for m in FAKE}

    def cls(mod, name, body=None):
        c = type(name, (), dict(body or {}))
        c.__module__ = mod
        c.__qualname__ = name
        setattr(mods[mod], name, c)
        return c

    sc = "meld.system.scalers"
    C = types.SimpleNamespace(
        NonLinearRamp=cls(sc, "NonLinearRamp"), LinearRamp=cls(sc, "LinearRamp"),
        ConstantRamp=cls(sc, "ConstantRamp"), Switcher=cls(sc, "TimeRampSwitcher"),
        Restraint=cls("meld.system.restraints", "DistanceRestraint"),
        Manager=cls("meld.system.restraints", "RestraintManager"),
        OmmSystem=cls("openmm", "System", {"__getstate__": lambda self: "<xml/>",
                                           "__setstate__": lambda self, s: None}))
    return mods, C


def _write_system(path, build):
    mods, C = _fake_classes()
    saved = {m: sys.modules.get(m) for m in FAKE}
    sys.modules.update(mods)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(build(C)))
    finally:
        for m, old in saved.items():                    # reading must work without these modules
            if old is None:
                sys.modules.pop(m, None)
            else:
                sys.modules[m] = old


def _ramp(C, kind, t0, t1, w0=1e-3, w1=1.0, factor=None):
    r = getattr(C, kind)()
    r.t_start, r.t_end, r.w_start, r.w_end = float(t0), float(t1), float(w0), float(w1)
    if factor is not None:
        r.factor = float(factor)
    return r


def _system(C, ramps):
    rests = []
    for i, r in enumerate(ramps):
        x = C.Restraint()
        x.ramp, x.k, x.atoms = r, 250.0, np.array([i, i + 1])
        rests.append(x)
    m = C.Manager()
    m.groups, m.omm = [rests, (rests[0],)], C.OmmSystem()
    return {"restraints": m, "names": ("a", "b")}


def test_ramp_end(tmp_path):
    def build(C):
        nl = _ramp(C, "NonLinearRamp", 1, 200, factor=4)
        lin = _ramp(C, "LinearRamp", 0, 150, w0=0.005)
        sw = C.Switcher()
        sw.first_ramp, sw.second_ramp, sw.switching_time = lin, C.ConstantRamp(), 120.0
        return _system(C, [nl, nl, lin, sw])
    p = tmp_path / "Data" / "system.dat"
    _write_system(p, build)
    info = time_ramps(p)
    assert info["full_stage"] == 200 and not info["unknown"]
    types_ = {r["type"]: r for r in info["ramps"]}
    assert set(types_) == {"NonLinearRamp", "LinearRamp", "ConstantRamp", "TimeRampSwitcher"}
    assert types_["NonLinearRamp"]["t_end"] == 200 and types_["NonLinearRamp"]["factor"] == 4


def test_no_ramps(tmp_path):
    p = tmp_path / "Data" / "system.dat"
    _write_system(p, lambda C: _system(C, [C.ConstantRamp()]))
    assert time_ramps(p)["full_stage"] == 0


def test_info_and_warning(meld_run, tmp_path, capsys):
    from reweightmeld.cli import main
    from reweightmeld.ensemble import reweight
    from reweightmeld.io import MeldRun
    root, _ = meld_run
    run_dir = tmp_path / "run"
    shutil.copytree(root, run_dir)
    _write_system(run_dir / "Data" / "system.dat", lambda C: _system(C, [_ramp(C, "LinearRamp", 0, 150)]))
    main(["info", str(run_dir), "--start-stage", "1"])
    out = capsys.readouterr().out
    assert "ramps end (final weight) at stage 150" in out and "WARNING" in out
    with pytest.warns(UserWarning, match="ramps run until stage 150"):
        reweight(MeldRun(run_dir), start_stage=1, log=None)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        rw = reweight(MeldRun(run_dir), start_stage=151, log=None)
    assert rw.settings["ramp_end_stage"] == 150
