"""Reweighting pipeline: windows -> MBAR per window -> pooled weights; saving, loading and resampling.

    from reweightmeld import MeldRun, reweight
    run = MeldRun("/path/to/meld_run")
    rw = reweight(run, target=0, start_stage=201, bayes=True)
    rw.save("out/system")                       # out/system.reweight.npz, .perm.npz, .summary.json
    ens, idx = rw.resample(run, 1000, seed=0)   # mdtraj trajectory drawn by weight
"""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from reweightmeld.mbar import kish_ess, reweight_window
from reweightmeld.walkers import walker_ids
from reweightmeld.windows import alpha_change_stages, constant_alpha_segments, frozen_window

SUMMARY_KEYS = ("ess", "ess_kish", "n_walkers_in_target", "frac_weight_target", "u_target_reweighted",
                "u_target_direct")


@dataclass
class Reweighting:
    """Weights of every (stage, slot) frame of the reweighted window(s) in the target slot.

    stage, slot, segment, weight: (N,) arrays, weights sum to 1 (segments mixed by `mix`);
    weight_bayes / weight_bayes_std: BayesMBAR posterior mean / std (if computed);
    windows: per-window diagnostics (dicts, see mbar.reweight_window);
    perm_stages, perms, walker_ids: permutation vectors and walker id of every slot from stage 1 to the end of
    the last window (walkers followed through every stage)."""
    run_dir: str
    target: int
    stage: np.ndarray
    slot: np.ndarray
    segment: np.ndarray
    weight: np.ndarray
    mix: np.ndarray
    windows: List[Dict]
    perm_stages: np.ndarray
    perms: np.ndarray
    walker_ids: np.ndarray
    weight_bayes: Optional[np.ndarray] = None
    weight_bayes_std: Optional[np.ndarray] = None
    settings: Dict = field(default_factory=dict)

    @property
    def n_frames(self):
        return len(self.weight)

    def summary(self):
        """JSON-friendly summary (one row per window + totals)."""
        rows = []
        for i, w in enumerate(self.windows):
            rows.append({"window": i, "first": int(w["stages"][0]), "last": int(w["stages"][-1]),
                         "n_stages": int(len(w["stages"])), "alpha_target": float(w["alphas"][self.target]),
                         "min_neighbor_overlap": float(np.min(w["overlap_offdiag"])),
                         "mix": float(self.mix[i]), **{k: _py(w[k]) for k in SUMMARY_KEYS}})
        return {"run_dir": self.run_dir, "target_slot": self.target, "n_frames": int(self.n_frames),
                "kish_ess_pooled": kish_ess(self.weight), "bayes": self.weight_bayes is not None,
                "settings": self.settings, "windows": rows}

    # ------------------------------------------------------------------------------------------- persistence
    def save(self, prefix):
        """Write <prefix>.reweight.npz (frames and weights), <prefix>.perm.npz (permutation vectors, walker
        ids) and <prefix>.summary.json. Returns the list of files."""
        prefix = str(prefix)
        Path(prefix).parent.mkdir(parents=True, exist_ok=True)
        extra = {}
        if self.weight_bayes is not None:
            extra = {"weight_bayes": self.weight_bayes, "weight_bayes_std": self.weight_bayes_std}
        win = {}
        for i, w in enumerate(self.windows):
            for k, v in w.items():
                if isinstance(v, np.ndarray) and k not in ("stage", "slot", "weight", "weight_bayes",
                                                           "weight_bayes_std"):
                    win[f"window{i}_{k}"] = v
        np.savez(prefix + ".reweight.npz", stage=self.stage, slot=self.slot, segment=self.segment,
                 weight=self.weight, segment_mix=self.mix, target=self.target, **extra, **win)
        used = np.isin(self.perm_stages, np.concatenate([w["stages"] for w in self.windows]))
        np.savez(prefix + ".perm.npz", stage=self.perm_stages, permutation_vectors=self.perms,
                 walker_ids=self.walker_ids, used=used)
        with open(prefix + ".summary.json", "w") as fh:
            json.dump(self.summary(), fh, indent=1)
        return [prefix + s for s in (".reweight.npz", ".perm.npz", ".summary.json")]

    @classmethod
    def load(cls, prefix):
        """Reload what save() wrote (per-window scalar diagnostics come from the summary JSON)."""
        prefix = str(prefix)
        r = np.load(prefix + ".reweight.npz")
        p = np.load(prefix + ".perm.npz")
        s = json.loads(Path(prefix + ".summary.json").read_text())
        windows = []
        for row in s["windows"]:
            i = row["window"]
            w = {k: r[f"window{i}_{k}"] for k in ("stages", "N_k", "f_k", "alphas", "overlap_offdiag", "g_walker")
                 if f"window{i}_{k}" in r}
            w.update({k: row[k] for k in SUMMARY_KEYS})
            m = r["segment"] == i
            w.update(stage=r["stage"][m], slot=r["slot"][m], weight=r["weight"][m] / r["segment_mix"][i])
            if "weight_bayes" in r:
                w.update(weight_bayes=r["weight_bayes"][m] / r["segment_mix"][i],
                         weight_bayes_std=r["weight_bayes_std"][m] / r["segment_mix"][i])
            windows.append(w)
        return cls(run_dir=s["run_dir"], target=int(r["target"]), stage=r["stage"], slot=r["slot"],
                   segment=r["segment"], weight=r["weight"], mix=r["segment_mix"], windows=windows,
                   perm_stages=p["stage"], perms=p["permutation_vectors"], walker_ids=p["walker_ids"],
                   weight_bayes=r["weight_bayes"] if "weight_bayes" in r else None,
                   weight_bayes_std=r["weight_bayes_std"] if "weight_bayes_std" in r else None,
                   settings=s.get("settings", {}))

    # ------------------------------------------------------------------------------------------- structures
    def draw(self, n, seed=0, weights="mbar"):
        """Indices of n frames drawn with replacement with probability = weight ('mbar' or 'bayes')."""
        w = self.weight if weights == "mbar" else self.weight_bayes
        rng = np.random.default_rng(seed)
        return rng.choice(self.n_frames, size=n, p=w / w.sum(), replace=True)

    def resample(self, run, n, seed=0, topology=None, weights="mbar"):
        """(mdtraj.Trajectory, indices): n structures drawn by weight (all-atom, MELD topology)."""
        idx = self.draw(n, seed, weights)
        return run.trajectory(self.stage[idx], self.slot[idx], topology), idx

    def save_ensemble(self, run, prefix, n, seed=0):
        """Weighted resample written as <prefix>.ensemble.dcd/.pdb + <prefix>.ensemble_index.npy."""
        ens, idx = self.resample(run, n, seed)
        ens.save_dcd(str(prefix) + ".ensemble.dcd")
        ens[0].save_pdb(str(prefix) + ".ensemble.pdb")
        np.save(str(prefix) + ".ensemble_index.npy", idx)
        return ens, idx


def _py(v):
    return v.item() if isinstance(v, np.generic) else v


def reweight(run, target=0, start_stage=201, segments=False, min_segment_len=50, bayes=False, n_posterior=200,
             seed=0, log=print):
    """Reweight a MELD run to ladder slot `target`.

    segments=False: one window, the stages after the last alpha change (and >= start_stage).
    segments=True: every constant-alpha segment from start_stage on (>= min_segment_len stages), each
    reweighted to its own target slot, then pooled with mixing fractions proportional to their walker-corrected
    ESS. Pooling requires the same target alpha in every segment (checked); whether MELD-adapt's prior on the
    discrete parameters changed between segments cannot be checked: compare u_target_direct across windows."""
    wins = (constant_alpha_segments(run, start_stage, min_segment_len) if segments
            else [frozen_window(run, start_stage)])
    if not wins:
        raise ValueError("no usable stages: check start_stage / min_segment_len")
    res = []
    for i, w in enumerate(wins):
        if log:
            log(f"window {i}: stages {w[0]}..{w[-1]} ({len(w)} stages)")
        res.append(reweight_window(run, w, target=target, bayes=bayes, n_posterior=n_posterior, seed=seed))
    a0 = np.array([r["alphas"][target] for r in res])
    if np.ptp(a0) > 1e-8:
        raise ValueError(f"target-slot alpha differs between windows ({a0}); do not pool")
    mix = np.array([r["ess"] for r in res], float)
    mix /= mix.sum()
    has_b = all("weight_bayes" in r for r in res)
    perm_stages = np.arange(max(int(run.stages[0]), 1), int(wins[-1][-1]) + 1)
    perms = run.permutation_vectors(perm_stages)
    return Reweighting(
        run_dir=str(run.run_dir), target=target,
        stage=np.concatenate([r["stage"] for r in res]), slot=np.concatenate([r["slot"] for r in res]),
        segment=np.concatenate([np.full(len(r["weight"]), i) for i, r in enumerate(res)]),
        weight=np.concatenate([r["weight"] * c for r, c in zip(res, mix)]), mix=mix, windows=res,
        perm_stages=perm_stages, perms=perms, walker_ids=walker_ids(perms),
        weight_bayes=np.concatenate([r["weight_bayes"] * c for r, c in zip(res, mix)]) if has_b else None,
        weight_bayes_std=np.concatenate([r["weight_bayes_std"] * c for r, c in zip(res, mix)]) if has_b else None,
        settings={"start_stage": int(start_stage), "segments": bool(segments), "min_segment_len": int(min_segment_len),
                  "bayes": bool(bayes), "alpha_change_stages": alpha_change_stages(run).tolist()})
