import json

import numpy as np

from reweightmeld.cli import main


def test_cli_run_report_resample(meld_run, tmp_path, capsys):
    root, _ = meld_run
    (root / "NOE.dat").write_text("1 CA 4 CA 12.0\n\n2 CA 6 CA 12.0\n")
    cfg = tmp_path / "c.yaml"
    cfg.write_text(f"run_dir: {root}\nstart_stage: 1\nresample: 40\nreport: true\nn_analysis: 300\n"
                   f"n_states: 3\nrestraints: ['NOE.dat:noe']\n")
    pre = str(tmp_path / "out" / "sys")
    main(["run", "-c", str(cfg), "-o", pre, "--seed", "2"])
    s = json.loads(open(pre + ".summary.json").read())
    assert s["settings"]["cli"]["seed"] == 2 and s["settings"]["cli"]["n_states"] == 3
    for name in ("8a_ladder_window", "8bc_weight_by_slot_overlap_window0", "8d_walkers_window0",
                 "8ef_weight_concentration_time", "9a_pca_states", "9b_state_populations", "9c_rmsf_rg",
                 "9d_contact_maps", "9e_restraint_satisfaction"):
        assert (tmp_path / "out" / "sys_plots" / f"{name}.png").exists(), name
    assert np.load(pre + ".ensemble_index.npy").shape == (40,)
    assert (tmp_path / "out" / "sys.state0.pdb").exists()
    main(["resample", pre, "-n", "7", "--out", str(tmp_path / "r")])
    assert np.load(str(tmp_path / "r") + ".ensemble_index.npy").shape == (7,)
    main(["report", pre, "--diagnostics-only", "--plot-dir", str(tmp_path / "p")])
    assert (tmp_path / "p" / "8a_ladder_window.png").exists()
    main(["info", str(root), "--start-stage", "1"])
    assert "ladder changed at 1 stages" in capsys.readouterr().out
