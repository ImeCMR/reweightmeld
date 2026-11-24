"""Restraint time ramps of a MELD run, read from Data/system.dat without MELD or OpenMM.

MELD restraints can carry a time ramp (meld/system/scalers.py: LinearRamp, NonLinearRamp, ConstantRamp,
TimeRampSwitcher) that scales their force constant with the REMD stage: the runner calls
prepare_for_timestep(state, alpha, step) with the exchange step (meld/remd/leader.py, worker.py). While a ramp
is running, the Hamiltonian of every ladder slot changes from stage to stage, so those stages must not be
reweighted together with later ones. A ramp gives its final weight for every stage >= t_end, so the
Hamiltonians are fixed from

    full_stage = max over ramps of t_end            (TimeRampSwitcher: also its switching_time)

on. The system is a pickle of meld.system.meld_system.System. It is read with an unpickler that builds real
numpy / builtin objects and a stand-in object (keeping its attributes) for every MELD / OpenMM class, then the
object graph is searched for the ramp classes.
"""
import pickle
from pathlib import Path

RAMP_CLASSES = ("ConstantRamp", "LinearRamp", "NonLinearRamp", "TimeRampSwitcher")

# globals that are built for real; everything else becomes a stand-in
_REAL = {
    "builtins": {"set", "frozenset", "complex", "slice", "range", "bytearray", "bytes", "object", "list", "dict",
                 "tuple", "int", "float", "str", "bool"},
    "copyreg": {"_reconstructor"},
    "collections": {"OrderedDict", "defaultdict", "deque"},
    "_codecs": {"encode"},
    "numpy": {"ndarray", "dtype"},
    "numpy.core.multiarray": {"_reconstruct", "scalar"},
    "numpy._core.multiarray": {"_reconstruct", "scalar"},
}


class _Stub:
    """Stand-in for a class that is not imported: keeps constructor arguments, state and items."""
    _module = ""

    def __init__(self, *args, **kwargs):
        self.__dict__["_args"] = args

    def __setstate__(self, state):
        if isinstance(state, tuple) and len(state) == 2 and (state[0] is None or isinstance(state[0], dict)):
            state = {**(state[0] or {}), **(state[1] or {})}      # (dict, slots) state
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.__dict__["_state"] = state

    def __setitem__(self, key, value):
        self.__dict__.setdefault("_items", {})[key] = value

    def append(self, value):
        self.__dict__.setdefault("_list", []).append(value)

    def extend(self, values):
        self.__dict__.setdefault("_list", []).extend(values)


class _StubUnpickler(pickle.Unpickler):
    _cache = {}

    def find_class(self, module, name):
        if name in _REAL.get(module, ()):
            return super().find_class(module, name)
        key = (module, name)
        if key not in self._cache:
            self._cache[key] = type(name, (_Stub,), {"_module": module})
        return self._cache[key]


def load_stub_pickle(path):
    """Unpickle a MELD file (e.g. Data/system.dat) with stand-ins for MELD / OpenMM classes."""
    with open(path, "rb") as fh:
        try:
            return _StubUnpickler(fh).load()
        except UnicodeDecodeError:                      # Python 2 pickles, as MELD's vault does
            fh.seek(0)
            return _StubUnpickler(fh, encoding="latin1").load()


def _walk(root):
    seen, stack = set(), [root]
    while stack:
        obj = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if isinstance(obj, _Stub):
            yield obj
            stack.extend(obj.__dict__.values())
        elif isinstance(obj, dict):
            stack.extend(obj.keys())
            stack.extend(obj.values())
        elif isinstance(obj, (list, tuple, set, frozenset)):
            stack.extend(obj)


def _full_stage(r):
    """First stage at which the ramp gives its final weight (None if unknown)."""
    name = type(r).__name__
    if name == "ConstantRamp":
        return 0
    if name in ("LinearRamp", "NonLinearRamp"):
        return float(r.t_end)
    if name == "TimeRampSwitcher":                       # its inner ramps are found by the walk as well
        return float(r.switching_time)
    return None


def time_ramps(system_path):
    """Time ramps in a MELD system pickle.

    Returns {"ramps": [{"type", "t_start", "t_end", "w_start", "w_end", "factor", "switching_time",
    "n_objects"}...], "full_stage": first stage with every ramp at its final weight (0 without ramps),
    "unknown": names of ramp-like classes that are not understood (full_stage is then unreliable)}."""
    system = load_stub_pickle(system_path)
    rows, unknown, full = {}, set(), 0.0
    for obj in _walk(system):
        name = type(obj).__name__
        is_ramp = name in RAMP_CLASSES or (name.endswith("Ramp") and "scalers" in obj._module)
        if not is_ramp:
            continue
        stage = _full_stage(obj)
        if stage is None:
            unknown.add(f"{obj._module}.{name}")
            continue
        full = max(full, stage)
        d = obj.__dict__
        key = (name,) + tuple(d.get(k) for k in ("t_start", "t_end", "w_start", "w_end", "factor",
                                                  "switching_time"))
        if key in rows:
            rows[key]["n_objects"] += 1
        else:
            rows[key] = {"type": name, "t_start": d.get("t_start"), "t_end": d.get("t_end"),
                         "w_start": d.get("w_start"), "w_end": d.get("w_end"), "factor": d.get("factor"),
                         "switching_time": d.get("switching_time"), "n_objects": 1}
    return {"ramps": sorted(rows.values(), key=lambda r: (r["type"], r["t_end"] or 0)),
            "full_stage": int(full) if float(full).is_integer() else float(full), "unknown": sorted(unknown)}


def describe(info):
    """Human-readable lines for time_ramps() output."""
    if not info["ramps"] and not info["unknown"]:
        return ["restraint time ramps: none (Hamiltonians fixed from stage 0)"]
    out = ["restraint time ramps:"]
    for r in info["ramps"]:
        if r["type"] == "ConstantRamp":
            txt = "constant (weight 1)"
        elif r["type"] == "TimeRampSwitcher":
            txt = f"switches ramp at stage {r['switching_time']:g}"
        else:
            txt = (f"weight {r['w_start']:g} -> {r['w_end']:g} over stages {r['t_start']:g}..{r['t_end']:g}"
                   + (f", factor {r['factor']:g}" if r["factor"] is not None else ""))
        out.append(f"  {r['type']}: {txt}  ({r['n_objects']} ramp object(s))")
    for u in info["unknown"]:
        out.append(f"  {u}: unknown ramp class, end not determined")
    out.append(f"ramps end (final weight) at stage {info['full_stage']}: use start_stage >= {info['full_stage'] + 1}")
    return out
