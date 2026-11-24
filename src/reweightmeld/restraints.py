"""MELD distance-restraint files and their satisfaction on MELD (all-atom) structures.

File format (as read by MELD's setup scripts): one restraint per line, `resi atomi resj atomj [upper_A]`,
residues 1-based; a blank line closes a group. MELD enforces, per group, the best (lowest-energy) line, so a
group counts as satisfied when any of its lines is within its upper bound. Files without a distance column
(e.g. hydrophobe.dat, strand_pair.dat) need a default upper bound.
Atom names are used as written: MELD's own frames carry hydrogens with the same (Amber) names.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import mdtraj
import numpy as np


@dataclass
class DistanceGroup:
    lines: List[tuple]        # (res_i 0-based, atom_i, res_j 0-based, atom_j, upper A)
    kind: str                 # label, e.g. "noe", "contact"
    source: str               # file name


def _is_hbond(line):
    """Backbone H-bond line: O-N or O-H pair with upper bound <= 3.3 A."""
    return {line[1], line[3]} in ({"O", "N"}, {"O", "H"}) and line[4] <= 3.3


def read_distance_file(path, kind="noe", default_upper: Optional[float] = None) -> List[DistanceGroup]:
    """Groups of a MELD distance file. With kind="noe", groups whose lines are all backbone H-bond pairs are
    labelled "hbond" (as in MELDDiffuse)."""
    groups, cur = [], []
    for raw in list(Path(path).read_text().splitlines()) + [""]:
        c = raw.split()
        if not c:
            if cur:
                k = "hbond" if kind == "noe" and all(_is_hbond(l) for l in cur) else kind
                groups.append(DistanceGroup(cur, k, Path(path).name))
                cur = []
            continue
        if len(c) >= 5:
            upper = float(c[4])
        elif default_upper is not None:
            upper = float(default_upper)
        else:
            raise ValueError(f"{path}: no distance column and no default upper bound: {raw!r}")
        cur.append((int(c[0]) - 1, c[1], int(c[2]) - 1, c[3], upper))
    return groups


def parse_spec(spec: str):
    """'FILE:KIND[:UPPER]' -> (file, kind, upper or None), e.g. 'NOE.dat:noe', 'hydrophobe.dat:contact:5.0'."""
    parts = spec.split(":")
    if len(parts) == 1:
        return parts[0], "noe", None
    return parts[0], parts[1], (float(parts[2]) if len(parts) > 2 else None)


def load_restraints(files: Sequence, base_dir=".") -> List[DistanceGroup]:
    """files: (file, kind[, default_upper]) tuples or 'FILE:KIND[:UPPER]' strings, relative to base_dir."""
    out = []
    for f in files:
        f = parse_spec(f) if isinstance(f, str) else tuple(f) + (None,) * (3 - len(f))
        out += read_distance_file(Path(base_dir) / f[0], f[1], f[2])
    return out


class RestraintScorer:
    """Group satisfaction of distance restraints on trajectories of one topology."""

    def __init__(self, groups: List[DistanceGroup], topology: mdtraj.Topology):
        index = {(a.residue.index, a.name): a.index for a in topology.atoms}
        pairs, owner, upper, self.n_missing = [], [], [], 0
        for g_i, g in enumerate(groups):
            for (ri, ai, rj, aj, u) in g.lines:
                ia, ja = index.get((ri, ai)), index.get((rj, aj))
                if ia is None or ja is None:
                    self.n_missing += 1
                    continue
                pairs.append((ia, ja)); owner.append(g_i); upper.append(u)
        self.groups = groups
        self.pairs, self.owner, self.upper = np.array(pairs).reshape(-1, 2), np.array(owner, int), np.array(upper)
        self.kinds = np.array([g.kind for g in groups])
        self.separation = np.array([min(abs(l[0] - l[2]) for l in g.lines) for g in groups])

    def satisfied(self, traj: mdtraj.Trajectory) -> np.ndarray:
        """(frames, groups) bool: some line of the group within its upper bound."""
        d = mdtraj.compute_distances(traj, self.pairs) * 10.0
        sat = np.zeros((len(self.groups), traj.n_frames), bool)
        np.logical_or.at(sat, self.owner, (d <= self.upper[None]).T)
        return sat.T

    def describe(self, g_i):
        ri, ai, rj, aj, u = self.groups[g_i].lines[0]
        n = len(self.groups[g_i].lines)
        return f"{ri + 1} {ai} - {rj + 1} {aj} <= {u} A" + (f" (+{n - 1} more lines)" if n > 1 else "")
