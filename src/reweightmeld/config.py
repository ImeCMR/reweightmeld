"""Settings of a reweighting run: defaults < YAML config (-c) < command-line flags.

Example config (every key optional except run_dir and out_prefix, which may also come from the command line):

    run_dir: /path/to/meld_run
    out_prefix: out/2lpu
    target: 0               # ladder slot to reweight to
    start_stage: 201        # skip the restraint ramp of the setup scripts (1 if the run has none)
    segments: false         # true: every constant-alpha segment, pooled
    min_segment_len: 50
    bayes: true             # BayesMBAR posterior weights (needs reweightmeld[bayes])
    n_posterior: 200
    seed: 0
    resample: 1000          # weighted resample of structures (0 = none)
    report: true            # diagnostic + structure plots into <out_prefix>_plots/
    n_analysis: 2000
    n_states: 5
    restraints: ["NOE.dat:noe", "local_NOE.dat:noe", "hydrophobe.dat:contact:5.0"]   # relative to run_dir
"""
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class Settings:
    run_dir: Optional[str] = None
    out_prefix: Optional[str] = None
    target: int = 0
    start_stage: int = 201
    segments: bool = False
    min_segment_len: int = 50
    bayes: bool = False
    n_posterior: int = 200
    seed: int = 0
    resample: int = 0
    report: bool = False
    n_analysis: int = 2000
    n_states: int = 5
    restraints: List[str] = field(default_factory=list)

    @classmethod
    def load(cls, path=None, **overrides):
        """Defaults, updated by the YAML file at `path` (if given), then by every override that is not None."""
        values = {}
        if path:
            values = yaml.safe_load(Path(path).read_text()) or {}
            unknown = set(values) - {f.name for f in fields(cls)}
            if unknown:
                raise ValueError(f"{path}: unknown settings {sorted(unknown)}")
        values.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**values)

    def as_dict(self):
        return asdict(self)

    @property
    def plot_dir(self):
        return f"{self.out_prefix}_plots"
