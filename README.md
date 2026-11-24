# reweightmeld

MBAR / BayesMBAR reweighting of MELD H,T-REMD runs (including MELD-adapt) to their target state, with
reweighting diagnostics, weighted resampling of structures and ensemble analysis plots.

It reads the MELD data store (`Data/Blocks/block_*.nc`, `Data/trajectory.pdb`) directly with netCDF4. It needs
neither MELD, OpenMM nor MELDDiffuse.

## Installation

```bash
conda create -n reweightmeld_env -c conda-forge python=3.11 numpy netcdf4 mdtraj "pymbar>=4" matplotlib pyyaml
conda activate reweightmeld_env
pip install -e /path/to/reweightmeld              # or: pip install git+<repository URL>
pip install -e "/path/to/reweightmeld[bayes]"     # optional: BayesMBAR (JAX, CPU is enough)
pip install -e "/path/to/reweightmeld[test]" && pytest   # optional: tests (run from /path/to/reweightmeld)
```

If packages in `~/.local` (pip `--user` installs) shadow the environment's own, e.g. pytest fails with
`No module named 'platformdirs.pytest_plugin'`, run with `export PYTHONNOUSERSITE=1`.

## Command line

```bash
reweightmeld info /path/to/meld_run                     # replicas, stages, ladder adaptations, window used
reweightmeld run /path/to/meld_run -o out/sys --bayes --resample 1000 --report \
    --restraint NOE.dat:noe --restraint local_NOE.dat:noe
reweightmeld run -c configs/example_2lpu.yaml           # the same from a YAML file (flags override it)
reweightmeld resample out/sys -n 500 --seed 1           # another weighted resample from saved weights
reweightmeld report out/sys --n-states 4                # plots again from saved weights
```

| Option | Meaning (default) |
|---|---|
| `--target` | ladder slot to reweight to (0 = lowest temperature, full restraints) |
| `--start-stage` | earliest stage used (201: skips the restraint ramp of the usual setup scripts; use 1 if the run has none) |
| `--segments` | reweight every constant-alpha segment, each to its own target slot, then pool them by ESS (default: only the stages after the last ladder adaptation) |
| `--min-segment-len` | segments shorter than this are skipped (50) |
| `--bayes` | BayesMBAR posterior mean and std of every weight |
| `--resample N` | weighted resample of N all-atom structures (DCD + PDB) |
| `--report` | diagnostic and structure-analysis plots into `<prefix>_plots/` |
| `--restraint FILE:KIND[:UPPER]` | MELD distance-restraint file for the satisfaction plot (relative to the run directory). KIND = `noe` (pure backbone H-bond groups are labelled `hbond`) or `contact`; UPPER = upper bound in Å for files without a distance column, e.g. `hydrophobe.dat:contact:5.0` |

## Python

```python
from reweightmeld import MeldRun, reweight, Reweighting, report

run = MeldRun("/path/to/meld_run")
rw = reweight(run, target=0, start_stage=201, bayes=True)
rw.save("out/sys")                                  # .reweight.npz, .perm.npz, .summary.json
ens, idx = rw.resample(run, 1000, seed=0)           # mdtraj.Trajectory drawn by weight

rw = Reweighting.load("out/sys")                    # later
report.diagnostics(run, rw, "out/sys_plots", show=True)
report.structure(run, rw, "out/sys_plots", prefix="out/sys", restraints=["NOE.dat:noe"], show=True)
```

`notebooks/meld_reweighting.ipynb` runs the same steps in a notebook, with every plot shown inline and saved.

## What is computed

**States and samples.** Every ladder slot k is one thermodynamic state, which requires the ladder (the alphas)
to be constant over the window used. A sample is a frame n = (stage s, slot j). Its reduced energy in state k is
MELD's own energy-matrix entry, in kT:

$$
u_k(n) \;=\; \texttt{energy\_matrix}[k, j, s] \;=\; \frac{U\big(x_{s,j};\ \alpha_k,\ y_{s,j}\big)}{R\,T(\alpha_k)} \;-\; \log p_{\alpha_k}\big(y_{s,j}\big)
$$

- U is the full potential: force field, implicit solvent and the restraints scaled for α_k.
- y is the number of active restraints MELD-adapt sampled for that structure.
- The last term is MELD's unnormalised log-prior on y.

MELD saves the matrix after the exchange, with its columns swapped together with the structures. So column j
belongs to the structure saved in slot j, and the diagonal equals the stored `energies`; this is checked on every
run.

**Weights.** MBAR (pymbar 4, robust solver) on all frames of the window gives the weight of every frame in the
target state; the weights sum to 1. BayesMBAR (uniform prior) is optional: it runs on frames thinned by the median
walker inefficiency and gives a posterior mean and std of every weight.

**Walkers.** The permutation vectors are followed from stage 1 to give the walker (continuous trajectory) in every
slot at every stage. The statistical inefficiency g of the target-state energy along each walker gives the
walker-corrected ESS:

$$
\mathrm{ESS} \;=\; \frac{\mathrm{ESS}_{\text{Kish}}}{\operatorname{median}_w\, g_w},
\qquad
\mathrm{ESS}_{\text{Kish}} \;=\; \frac{\left(\sum_n w_n\right)^2}{\sum_n w_n^2}
$$

**Checks reported** (in `<prefix>.summary.json`, per window):
- `ess`, `ess_kish`;
- `u_target_reweighted` ≈ `u_target_direct`: the reweighted mean target energy should match the plain mean in the
  target slot;
- `min_neighbor_overlap`: MBAR overlap of neighbouring slots; values below ~0.03 mean a bottleneck in the ladder;
- `n_walkers_in_target`: how many different walkers visited the target slot;
- `frac_weight_target`: the share of the weight carried by the target slot's own frames.

**Plots** (`--report`):

| Plot | What to look for |
|---|---|
| 8a Ladder and window | alphas stop changing before the reweighted window (shaded) starts |
| 8b Weight by slot | most weight from the target slot and its neighbours |
| 8c Neighbour overlap | every neighbouring pair well above ~0.03 |
| 8d Walkers | several walkers visit the target slot; their statistical inefficiency |
| 8e Weight concentration | how many frames carry 50 / 90 / 99 / 99.9 % of the weight |
| 8f Weight over time | roughly flat; a trend means the window is not equilibrated |
| 8g BayesMBAR (with `--bayes`) | relative uncertainty of the weights |
| 9a–9b PCA and states | Cα-distance PCA fitted on the reweighted ensemble, k-means states and their populations, compared with the target slot alone and all slots unweighted; state representatives saved as `<prefix>.state<k>.pdb` |
| 9c RMSF and Rg | Cα fluctuations and radius of gyration |
| 9d Contact maps | Cα contact probability (< 8 Å), reweighted vs target slot only |
| 9e Restraint satisfaction (with `--restraint`) | fraction of groups satisfied per structure; per group vs sequence separation |

## Output files

| File | Content |
|---|---|
| `<prefix>.reweight.npz` | `stage`, `slot`, `segment`, `weight` of every frame; `segment_mix`, `target`; `weight_bayes(_std)` with `--bayes`; per-window arrays `window<i>_*` (stages, f_k, N_k, alphas, neighbour overlap, walker g) |
| `<prefix>.perm.npz` | `stage`, `permutation_vectors` (S, R), `walker_ids` (S, R), `used` (stage inside a reweighted window) |
| `<prefix>.summary.json` | settings and per-window diagnostics |
| `<prefix>.ensemble.dcd/.pdb`, `.ensemble_index.npy` | weighted resample; row of `reweight.npz` behind each frame |
| `<prefix>_plots/*.png`, `<prefix>.state<k>.pdb` | plots and state representatives |

To load any frame yourself: `run.positions([stage], [slot])` returns its coordinates in nm, and
`run.trajectory(stages, slots)` returns an mdtraj trajectory.

## Package layout

```
src/reweightmeld/
  io.py          MeldRun: blocks, energy matrices, alphas, permutation vectors, positions, topology
  windows.py     constant-alpha windows (last adaptation, segments)
  walkers.py     walker ids from permutation vectors, statistical inefficiency along walkers
  mbar.py        u_kn construction, MBAR, BayesMBAR, ESS, reweight_window()
  ensemble.py    reweight() pipeline, Reweighting (save / load / resample)
  restraints.py  MELD distance-restraint files and group satisfaction
  analysis.py    Cα-distance PCA, k-means states, RMSF, Rg, contact maps
  plots.py       figure functions
  report.py      diagnostics() and structure(): all figures for a run
  config.py      settings (defaults < YAML < command line)
  cli.py         `reweightmeld` command
tests/           synthetic MELD run with a known answer, unit and CLI tests
configs/         example settings
notebooks/       meld_reweighting.ipynb (the whole workflow, plots inline)
```

## Origin

The reweighting code is taken from MELDDiffuse (`melddiffuse.data.meld_io`, `melddiffuse.data.reweight`); it
gives the same weights on the same run.
