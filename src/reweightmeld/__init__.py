"""reweightmeld: MBAR / BayesMBAR reweighting of MELD H,T-REMD runs to their target state.

Reads the MELD data store (Data/Blocks/block_*.nc) directly; needs neither MELD nor OpenMM.

    from reweightmeld import MeldRun, reweight, Reweighting
    run = MeldRun("/path/to/meld_run")
    rw = reweight(run, target=0, start_stage=201, bayes=True)
    rw.save("out/system")
"""
from reweightmeld.ensemble import Reweighting, reweight
from reweightmeld.io import MeldRun

__version__ = "0.1.0"
__all__ = ["MeldRun", "Reweighting", "reweight", "__version__"]
