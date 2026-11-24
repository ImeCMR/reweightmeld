"""Reports: the diagnostic and structure-analysis figures for a reweighted MELD run.

    from reweightmeld import MeldRun, Reweighting, report
    run = MeldRun(run_dir); rw = Reweighting.load(prefix)
    report.diagnostics(run, rw, out_dir, show=True)                       # 8a-8g
    report.structure(run, rw, out_dir, prefix=prefix, restraints=[...])   # 9a-9e (+ state PDBs)

Every figure is saved as <out_dir>/<name>.png; show=True also displays it (Jupyter: inline), otherwise it is
closed. Both functions return {name: path} plus, for structure(), the computed numbers.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from reweightmeld import analysis, plots
from reweightmeld.restraints import RestraintScorer, load_restraints
from reweightmeld.windows import alpha_change_stages


def _emit(fig, out_dir, name, show, files):
    fig.tight_layout()
    path = Path(out_dir) / f"{name}.png"
    fig.savefig(path, dpi=150)
    files[name] = str(path)
    if show:
        plt.show()
    else:
        plt.close(fig)


def diagnostics(run, rw, out_dir, show=False):
    """Reweighting diagnostics: ladder and window (8a), weight by slot + neighbour overlap (8b-c), walkers (8d),
    weight concentration and weight over time (8e-f), BayesMBAR uncertainty (8g, if computed)."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    files, R, t = {}, run.n_replicas, rw.target
    st = run.stages
    _emit(plots.ladder(st, run.alphas(st), [w["stages"] for w in rw.windows], alpha_change_stages(run)),
          out_dir, "8a_ladder_window", show, files)
    multi = len(rw.windows) > 1
    for i, w in enumerate(rw.windows):
        lab = f"window {i}: " if multi else ""
        _emit(plots.weight_by_slot(w, t, R, lab), out_dir, f"8bc_weight_by_slot_overlap_window{i}", show, files)
        m = np.isin(rw.perm_stages, w["stages"])
        _emit(plots.walkers(rw.perm_stages[m], rw.walker_ids[m, t], w["g_walker"], w["ess_kish"], w["ess"], R, lab),
              out_dir, f"8d_walkers_window{i}", show, files)
    _emit(plots.weight_concentration(rw.weight, rw.stage, sum(w["ess"] for w in rw.windows)),
          out_dir, "8ef_weight_concentration_time", show, files)
    if rw.weight_bayes is not None:
        _emit(plots.bayes_uncertainty(rw.weight, rw.weight_bayes, rw.weight_bayes_std),
              out_dir, "8g_bayesmbar_uncertainty", show, files)
    return files


def structure(run, rw, out_dir, prefix=None, n_analysis=2000, n_states=5, restraints=(), restraint_dir=None,
              seed=0, show=False, log=print):
    """Structure analysis of the reweighted ensemble against two unweighted sets from the same window(s): the
    target slot alone and all slots pooled. PCA + states (9a), populations (9b), CA RMSF + Rg (9c), contact maps
    (9d), restraint satisfaction (9e, with `restraints`: (file, kind[, upper]) tuples or 'FILE:KIND[:UPPER]',
    relative to restraint_dir, default the run directory). State representatives -> <prefix>.state<k>.pdb."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    files, t = {}, rw.target
    rng = np.random.default_rng(seed + 1)
    top = run.topology()
    n = min(n_analysis, rw.n_frames)
    i_w = rng.choice(rw.n_frames, size=n, p=rw.weight / rw.weight.sum(), replace=True)
    i_t = np.where(rw.slot == t)[0]
    i_t = rng.choice(i_t, size=min(n, len(i_t)), replace=False)
    i_p = rng.choice(rw.n_frames, size=n, replace=False)
    names = ("reweighted (MBAR)", f"slot {t} only, unweighted", "all slots, unweighted")
    sets = {k: run.trajectory(rw.stage[i], rw.slot[i], top) for k, i in zip(names, (i_w, i_t, i_p))}
    colours = dict(zip(names, (plots.BLUE, plots.ORANGE, plots.GREY)))
    if log:
        log(f"structure analysis: {n} frames per set ({len(np.unique(i_w))} different frames in the reweighted draw)")

    S = analysis.states(sets, names[0], n_states=n_states, seed=seed)
    if prefix:
        for j, r in enumerate(S["representatives"]):
            if r >= 0:
                sets[names[0]][r].save_pdb(f"{prefix}.state{j}.pdb")
    _emit(plots.pca_states(S["P"], S["centres"], S["explained"], colours), out_dir, "9a_pca_states", show, files)
    _emit(plots.state_populations(S["populations"], colours,
                                  "State populations" + (f" (representatives: {Path(prefix).name}.state<k>.pdb)" if prefix else "")),
          out_dir, "9b_state_populations", show, files)

    ref = sets[names[0]][S["representatives"][0]]
    rmsf = {k: analysis.rmsf_ca(tr, ref) for k, tr in sets.items()}
    rg = {k: analysis.radius_of_gyration(tr) for k, tr in sets.items()}
    _emit(plots.rmsf_rg(rmsf, rg, colours), out_dir, "9c_rmsf_rg", show, files)
    Mw, Mt = analysis.contact_probability(sets[names[0]]), analysis.contact_probability(sets[names[1]])
    _emit(plots.contact_maps(Mw, Mt, "reweighted", names[1]), out_dir, "9d_contact_maps", show, files)

    out = {"files": files, "states": S, "rmsf": rmsf, "rg": rg}
    if restraints:
        groups = load_restraints(restraints, restraint_dir or run.run_dir)
        sc = RestraintScorer(groups, top)
        sat = {k: sc.satisfied(tr) for k, tr in sets.items()}
        _emit(plots.restraint_satisfaction(sat, sc.kinds, sc.separation, colours, names[0]),
              out_dir, "9e_restraint_satisfaction", show, files)
        pg = sat[names[0]].mean(0)
        worst = [(int(g), sc.kinds[g], float(pg[g]), sc.describe(g)) for g in np.argsort(pg)[:10]]
        out.update(restraint_scorer=sc, satisfaction=sat, least_satisfied=worst)
        if log:
            log(f"{len(groups)} groups, {len(sc.pairs)} lines resolved, {sc.n_missing} lines with unknown atoms skipped")
            log("least satisfied groups (reweighted): group, kind, fraction satisfied, first line")
            for g, kind, f, d in worst:
                log(f"  {g:5d} {kind:8s} {f:.2f}   {d}")
    return out
