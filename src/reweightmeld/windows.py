"""Stage windows over which the MELD ladder (the alphas of every slot) is constant.

MBAR treats every ladder slot as a fixed thermodynamic state, so the alphas must not change inside a window.
MELD-adapt adapts the ladder on a doubling schedule (stages 100, 200, 400, ...); the stored alphas tell where
it really stopped.
"""
import numpy as np


def alpha_change_stages(run):
    """Stages whose alphas differ from the previous stored stage."""
    st = run.stages
    a = run.alphas(st)
    return st[1:][np.abs(np.diff(a, axis=0)).max(axis=1) > 0]


def frozen_window(run, min_start=201):
    """Stages after the last alpha change (and >= min_start) to the end of the run.
    The first stage after a change is dropped as well (MELDDiffuse convention)."""
    changes = alpha_change_stages(run)
    start = int(changes[-1]) + 1 if len(changes) else int(run.stages[0])
    start = max(start, int(min_start))
    if start > run.stages[-1]:
        raise ValueError(f"no stage left after {start}: the ladder changes until the end of the run")
    return np.arange(start, int(run.stages[-1]) + 1)


def constant_alpha_segments(run, min_start=201, min_len=50):
    """Every constant-alpha segment from min_start on (first stage after each change dropped), keeping those
    with at least min_len stages."""
    st = run.stages[run.stages >= min_start]
    al = run.alphas(st)
    cut = np.where(np.abs(np.diff(al, axis=0)).max(axis=1) > 0)[0] + 1
    segs = [seg[1:] if i else seg for i, seg in enumerate(np.split(st, cut))]
    return [seg for seg in segs if len(seg) >= min_len]
