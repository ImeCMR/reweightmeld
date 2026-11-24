"""Command line interface:  reweightmeld <command> ...

    info     RUN_DIR                          replicas, stages, ladder adaptations, the window that would be used
    run      [RUN_DIR] -o PREFIX [-c CONFIG]  MBAR (+ BayesMBAR) weights, permutation vectors, walker ids;
                                              optional weighted resample (--resample N) and plots (--report)
    resample PREFIX -n N                      weighted resample of structures from a saved reweighting
    report   PREFIX                           diagnostic and structure plots from a saved reweighting

Outputs of `run` (PREFIX = -o): PREFIX.reweight.npz, PREFIX.perm.npz, PREFIX.summary.json;
with --resample: PREFIX.ensemble.dcd/.pdb/_index.npy; with --report: PREFIX_plots/*.png, PREFIX.state<k>.pdb.
"""
import argparse
import json
import logging


def _add_report_args(p):
    p.add_argument("--n-analysis", type=int, default=None, help="structures drawn by weight for the analysis (2000)")
    p.add_argument("--n-states", type=int, default=None, help="conformational states (k-means on the PCA) (5)")
    p.add_argument("--restraint", action="append", default=None, metavar="FILE:KIND[:UPPER]",
                   help="MELD restraint file for the satisfaction plot, relative to the run directory; KIND = noe or "
                        "contact; UPPER = default upper bound (A) for files without a distance column. Repeatable.")


def main(argv=None):
    p = argparse.ArgumentParser(prog="reweightmeld", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    i = sub.add_parser("info", help="describe a MELD run")
    i.add_argument("run_dir")
    i.add_argument("--start-stage", type=int, default=201)

    r = sub.add_parser("run", help="reweight a MELD run")
    r.add_argument("run_dir", nargs="?", default=None)
    r.add_argument("-o", "--out-prefix", default=None)
    r.add_argument("-c", "--config", default=None, help="YAML settings (command-line flags take precedence)")
    r.add_argument("--target", type=int, default=None, help="ladder slot to reweight to (0)")
    r.add_argument("--start-stage", type=int, default=None, help="earliest stage used (201)")
    r.add_argument("--segments", action="store_true", default=None,
                   help="reweight every constant-alpha segment and pool them (default: last window only)")
    r.add_argument("--min-segment-len", type=int, default=None)
    r.add_argument("--bayes", action="store_true", default=None, help="BayesMBAR posterior weights")
    r.add_argument("--n-posterior", type=int, default=None)
    r.add_argument("--seed", type=int, default=None)
    r.add_argument("--resample", type=int, default=None, metavar="N", help="weighted resample of N structures")
    r.add_argument("--report", action="store_true", default=None, help="also write the plots")
    _add_report_args(r)

    s = sub.add_parser("resample", help="weighted resample from a saved reweighting")
    s.add_argument("prefix")
    s.add_argument("-n", type=int, required=True)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--weights", choices=["mbar", "bayes"], default="mbar")
    s.add_argument("--out", default=None, help="output prefix (default: PREFIX)")
    s.add_argument("--run-dir", default=None, help="override the run directory stored in the summary")

    g = sub.add_parser("report", help="plots from a saved reweighting")
    g.add_argument("prefix")
    g.add_argument("--run-dir", default=None, help="override the run directory stored in the summary")
    g.add_argument("--plot-dir", default=None, help="default: PREFIX_plots")
    g.add_argument("--diagnostics-only", action="store_true")
    g.add_argument("--seed", type=int, default=0)
    _add_report_args(g)                             # defaults: the values saved by `run`

    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    from reweightmeld.ensemble import Reweighting, reweight
    from reweightmeld.io import MeldRun

    if a.cmd == "info":
        from reweightmeld.windows import alpha_change_stages, frozen_window
        run = MeldRun(a.run_dir)
        ch = alpha_change_stages(run)
        print(run)
        print(f"ladder changed at {len(ch)} stages: {ch.tolist()}")
        w = frozen_window(run, a.start_stage)
        print(f"window used by `run` (start_stage {a.start_stage}): stages {w[0]}..{w[-1]} ({len(w)} stages, "
              f"{len(w) * run.n_replicas} frames)")
        return

    if a.cmd == "run":
        from reweightmeld.config import Settings
        st = Settings.load(a.config, run_dir=a.run_dir, out_prefix=a.out_prefix, target=a.target,
                           start_stage=a.start_stage, segments=a.segments, min_segment_len=a.min_segment_len,
                           bayes=a.bayes, n_posterior=a.n_posterior, seed=a.seed, resample=a.resample,
                           report=a.report, n_analysis=a.n_analysis, n_states=a.n_states, restraints=a.restraint)
        if not st.run_dir or not st.out_prefix:
            p.error("run needs a run directory and -o/--out-prefix (on the command line or in the config)")
        run = MeldRun(st.run_dir)
        print(run)
        rw = reweight(run, target=st.target, start_stage=st.start_stage, segments=st.segments,
                      min_segment_len=st.min_segment_len, bayes=st.bayes, n_posterior=st.n_posterior, seed=st.seed)
        rw.settings["cli"] = st.as_dict()
        for f in rw.save(st.out_prefix):
            print("wrote", f)
        print(json.dumps(rw.summary()["windows"], indent=1))
        if st.resample:
            _, idx = rw.save_ensemble(run, st.out_prefix, st.resample, st.seed)
            print(f"wrote {st.out_prefix}.ensemble.dcd/.pdb ({st.resample} structures, {len(set(idx))} different)")
        if st.report:
            _report(run, rw, st.plot_dir, st.out_prefix, st.n_analysis, st.n_states, st.restraints, st.seed)
        return

    rw = Reweighting.load(a.prefix)
    run = MeldRun(a.run_dir or rw.run_dir)
    if a.cmd == "resample":
        out = a.out or a.prefix
        _, idx = rw.save_ensemble(run, out, a.n, a.seed)
        print(f"wrote {out}.ensemble.dcd/.pdb ({a.n} structures, {len(set(idx))} different)")
    elif a.cmd == "report":
        saved = rw.settings.get("cli", {})          # settings of the `run` that made the weights
        _report(run, rw, a.plot_dir or f"{a.prefix}_plots", a.prefix,
                a.n_analysis or saved.get("n_analysis", 2000), a.n_states or saved.get("n_states", 5),
                a.restraint if a.restraint is not None else saved.get("restraints", []),
                a.seed, structure=not a.diagnostics_only)


def _report(run, rw, plot_dir, prefix, n_analysis, n_states, restraints, seed, structure=True):
    import matplotlib
    matplotlib.use("Agg")
    from reweightmeld import report
    files = report.diagnostics(run, rw, plot_dir)
    if structure:
        files.update(report.structure(run, rw, plot_dir, prefix=prefix, n_analysis=n_analysis, n_states=n_states,
                                      restraints=restraints, seed=seed)["files"])
    for f in files.values():
        print("wrote", f)


if __name__ == "__main__":
    main()
