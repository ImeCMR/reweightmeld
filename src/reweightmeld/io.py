"""Read a MELD run directly from its data store, Data/Blocks/block_*.nc (netCDF4). No MELD install needed.

Layout (MELD 0.6, checked against MELD's remd/leader.py, remd/ladder.py and runner/openmm_runner.py):
- every per-stage array is time-last; stage s is stored at time index s of the block that contains it;
- positions are in nm;
- energy_matrix[k, j, s] is the structure saved in ladder slot j at stage s, evaluated with the Hamiltonian
  and temperature of slot k, in kT, including minus MELD's log-prior of the discrete parameters:
      U(x_j; alpha_k, y_j) / (R T(alpha_k)) - log_prior(y_j; alpha_k)
  It is computed before the exchange, but the ladder swaps its columns with every accepted exchange, so it
  matches the post-exchange positions[j, ..., s] (its diagonal equals energies[j, s]);
- permutation_vectors[:, 0] is not written (stage 0 is the starting state); it is read as the identity.
"""
import glob
import os
import tempfile
from pathlib import Path

import mdtraj
import netCDF4
import numpy as np

# MELD / Amber residue names -> standard names
RESNAME_MAP = {"HIE": "HIS", "HID": "HIS", "HIP": "HIS", "HSD": "HIS", "HSE": "HIS",
               "CYX": "CYS", "CYM": "CYS", "ASH": "ASP", "GLH": "GLU", "LYN": "LYS"}


class MeldRun:
    """A MELD run directory (with Data/Blocks/block_*.nc and Data/trajectory.pdb)."""

    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.block_files = sorted(glob.glob(str(self.run_dir / "Data" / "Blocks" / "block_*.nc")))
        if not self.block_files:
            raise FileNotFoundError(f"no MELD blocks in {self.run_dir}/Data/Blocks")
        self._stage_block = {}
        for b, fp in enumerate(self.block_files):
            with netCDF4.Dataset(fp) as d:
                a = d.variables["alphas"][:]
                filled = ~np.ma.getmaskarray(a).any(axis=0)
                for s in np.where(filled)[0]:
                    self._stage_block.setdefault(int(s), b)
                if b == 0:
                    self.n_replicas = d.dimensions["n_replicas"].size
                    self.n_atoms = d.dimensions["n_atoms"].size
        self.stages = np.array(sorted(self._stage_block))

    def __repr__(self):
        return (f"MeldRun({str(self.run_dir)!r}: {self.n_replicas} replicas, {self.n_atoms} atoms, "
                f"stages {self.stages[0]}..{self.stages[-1]})")

    def _by_block(self, stages):
        out = {}
        for s in stages:
            if int(s) not in self._stage_block:
                raise KeyError(f"stage {int(s)} is not stored in {self.run_dir}")
            out.setdefault(self._stage_block[int(s)], []).append(int(s))
        return out

    def _read(self, var, stages, fill=None):
        """var[..., stages] with the time axis moved to the front; masked entries -> fill (if given)."""
        stages = np.asarray(stages, dtype=int)
        res = {}
        for b, ss in self._by_block(stages).items():
            with netCDF4.Dataset(self.block_files[b]) as d:
                lo, hi = min(ss), max(ss) + 1
                chunk = d.variables[var][..., lo:hi]
                if np.ma.isMaskedArray(chunk) and fill is not None:
                    chunk = chunk.filled(fill)
                chunk = np.asarray(chunk)
                for s in ss:
                    res[s] = chunk[..., s - lo]
        return np.stack([res[int(s)] for s in stages])

    def alphas(self, stages):
        return self._read("alphas", stages)                          # (S, R)

    def energies(self, stages):
        return self._read("energies", stages)                        # (S, R)

    def energy_matrices(self, stages):
        return self._read("energy_matrix", stages)                   # (S, R_k, R_j)

    def permutation_vectors(self, stages):
        """(S, R) int. Stages without a stored vector (stage 0) are returned as the identity."""
        p = self._read("permutation_vectors", stages, fill=-1).astype(int)
        bad = (p < 0).any(axis=1)
        p[bad] = np.arange(self.n_replicas)
        return p

    def discrete_parameters(self, stages):
        return self._read("discrete_parameters", stages)             # (S, R, P)

    def positions(self, stages, slots):
        """Positions (nm) for paired arrays of stages and slots -> (N, n_atoms, 3)."""
        stages, slots = np.asarray(stages, int), np.asarray(slots, int)
        out = np.empty((len(stages), self.n_atoms, 3), dtype=np.float32)
        blocks = np.array([self._stage_block[int(s)] for s in stages])
        for b in np.unique(blocks):
            idx = np.where(blocks == b)[0]
            lo, hi = stages[idx].min(), stages[idx].max() + 1
            with netCDF4.Dataset(self.block_files[b]) as d:
                chunk = np.asarray(d.variables["positions"][..., lo:hi])   # (R, A, 3, T)
            out[idx] = chunk[slots[idx], :, :, stages[idx] - lo]
        return out

    def time_ramps(self):
        """Restraint time ramps from Data/system.dat (see reweightmeld.ramps.time_ramps); None if the file is
        missing or cannot be read. The Hamiltonians are fixed from stage result["full_stage"] on."""
        if not hasattr(self, "_ramps"):
            from reweightmeld.ramps import time_ramps
            path = self.run_dir / "Data" / "system.dat"
            try:
                self._ramps = time_ramps(path) if path.exists() else None
            except Exception as err:                        # unexpected pickle content: report, do not fail
                self._ramps = None
                self._ramps_error = f"{type(err).__name__}: {err}"
        return self._ramps

    def topology(self, standard_names=False) -> mdtraj.Topology:
        """All-atom topology of the run (frame 1 of Data/trajectory.pdb). standard_names=True maps
        MELD / Amber residue names (HIE, CYX, ...) to standard ones; atom names are kept as written."""
        top = read_pdb_frame(self.run_dir / "Data" / "trajectory.pdb", 0).topology
        if standard_names:
            for res in top.residues:
                res.name = RESNAME_MAP.get(res.name, res.name)
        return top

    def trajectory(self, stages, slots, topology=None) -> mdtraj.Trajectory:
        """mdtraj trajectory of the (stage, slot) frames."""
        return mdtraj.Trajectory(self.positions(stages, slots), topology or self.topology())


def read_pdb_frame(pdb_path, k=0) -> mdtraj.Trajectory:
    """k-th frame of a PDB whose frames are separated by END records (MELD's Data/trajectory.pdb: no MODEL
    records, no element column). Atom names are kept as written (Amber names, e.g. HB2 / HB3)."""
    lines, frame = [], 0
    with open(pdb_path) as fh:
        for line in fh:
            if frame == k and line.startswith(("ATOM", "HETATM", "TER")):
                lines.append(line)
            if line.startswith("END") and not line.startswith("ENDMDL"):
                if frame == k:
                    break
                frame += 1
    if not lines:
        raise ValueError(f"frame {k} not found in {pdb_path}")
    fd, tmp = tempfile.mkstemp(suffix=".pdb")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.writelines(lines + ["END\n"])
        return mdtraj.load_pdb(tmp, standard_names=False)
    finally:
        os.remove(tmp)
