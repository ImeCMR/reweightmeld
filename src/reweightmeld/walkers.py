"""Walkers: continuous trajectories followed through the replica exchanges with the permutation vectors."""
import numpy as np
from pymbar import timeseries


def walker_ids(perms):
    """perms: (S, R) permutation vectors of consecutive stages (MELD convention: after the exchange, slot i
    holds the structure that was in slot perms[i]). Returns (S, R): the walker id in every slot, walker w
    being the trajectory that started in slot w (MELD's extract_trajectory follow_structure: ids = ids[perm])."""
    ids = np.arange(perms.shape[1])
    out = np.empty_like(perms)
    for s in range(len(perms)):
        ids = ids[perms[s]]
        out[s] = ids
    return out


def walker_inefficiency(E, wid, target=0):
    """Statistical inefficiency g of the target-state reduced energy along each walker's trajectory.
    E: (S, R, R) energy matrices of the window, wid: (S, R) walker ids of the same stages."""
    R = E.shape[1]
    g = []
    for w in range(R):
        slot = np.argmax(wid == w, axis=1)                  # slot of walker w at every stage
        u = E[np.arange(len(E)), target, slot]
        g.append(timeseries.statistical_inefficiency(u))
    return np.array(g)
