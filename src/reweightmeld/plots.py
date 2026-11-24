"""Figures for reweighting diagnostics and structure analysis. Every function returns a matplotlib Figure.

The package style is applied only while a figure is built (matplotlib's global settings are left alone).
"""
import functools

import numpy as np
import matplotlib.pyplot as plt

from reweightmeld.mbar import kish_ess

BLUE, ORANGE, GREEN, GREY, RED = "#2a78d6", "#eb6834", "#1baf7a", "#7a7a75", "#b8323a"
STYLE = {"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
         "grid.color": "#e6e6e3", "axes.axisbelow": True, "legend.frameon": False}


def _styled(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with plt.rc_context(STYLE):
            return fn(*args, **kwargs)
    return wrapper


# ------------------------------------------------------------------------------------------- diagnostics
@_styled
def ladder(stages, alphas, windows, changes):
    """Alphas of every slot over all stages; reweighted window(s) shaded; adaptation stages dotted."""
    R = alphas.shape[1]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for j in range(R):
        ax.plot(stages, alphas[:, j], lw=0.8, color=plt.cm.viridis(j / max(R - 1, 1)))
    for w in windows:
        ax.axvspan(w[0], w[-1], color=BLUE, alpha=0.12)
    for c in changes:
        ax.axvline(c, color=GREY, lw=0.6, ls=":")
    ax.set(xlabel="MELD stage", ylabel="alpha of each slot (0 = target)",
           title=f"Ladder: {len(changes)} stages where alphas changed (dotted); shaded = reweighted window(s)")
    return fig


@_styled
def weight_by_slot(win, target, n_replicas, label=""):
    """Fraction of the target weight carried by each slot, and the overlap between neighbouring slots."""
    fw = np.bincount(win["slot"], weights=win["weight"], minlength=n_replicas)
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.6))
    a.bar(np.arange(n_replicas), fw, color=BLUE, width=0.8, label="MBAR")
    if "weight_bayes" in win:
        fb = np.bincount(win["slot"], weights=win["weight_bayes"], minlength=n_replicas)
        a.plot(np.arange(n_replicas), fb, "D", ms=4, mfc="white", mec="k", label="BayesMBAR posterior mean")
        a.legend()
    a.set(xlabel=f"replica slot ({target} = target)", ylabel="fraction of the target-state weight",
          title=f"{label}{fw[target]:.0%} of the weight from the target slot itself")
    ov = win["overlap_offdiag"]
    b.plot(np.arange(len(ov)), ov, "o-", color=BLUE, ms=3)
    b.axhline(0.03, color=RED, ls="--", lw=1, label="0.03")
    b.set(xlabel="neighbouring slot pair (k, k+1)", ylabel="MBAR overlap",
          title=f"weakest neighbour overlap {ov.min():.3f} (pair {int(np.argmin(ov))}-{int(np.argmin(ov)) + 1})")
    b.legend()
    return fig


@_styled
def walkers(stages, walker_in_target, g_walker, ess_kish, ess, n_replicas, label=""):
    """Walker in the target slot over time; statistical inefficiency of every walker."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [2, 1]})
    a.scatter(stages, walker_in_target, s=2, c=walker_in_target, cmap="tab20", vmin=0, vmax=max(n_replicas - 1, 1))
    a.set(xlabel="MELD stage", ylabel="walker in the target slot", ylim=(-0.5, n_replicas - 0.5),
          title=f"{label}{len(np.unique(walker_in_target))} of {n_replicas} walkers visited the target slot")
    b.bar(np.arange(len(g_walker)), g_walker, color=GREY, width=0.8)
    b.axhline(np.median(g_walker), color=BLUE, ls="--", lw=1, label=f"median g = {np.median(g_walker):.0f}")
    b.set(xlabel="walker", ylabel="statistical inefficiency g", yscale="log",
          title=f"ESS: Kish {ess_kish:.0f} -> walker-corrected {ess:.1f}")
    b.legend()
    return fig


@_styled
def weight_concentration(weight, stage, ess_corrected):
    """How many frames carry the weight; target weight over time."""
    ws = np.sort(weight)[::-1]
    cw = np.cumsum(ws)
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.6))
    a.plot(np.arange(1, len(ws) + 1), cw, color=BLUE)
    a.set_xscale("log")
    for q, c in ((0.5, "#86b6ef"), (0.9, "#5598e7"), (0.99, "#256abf"), (0.999, "#104281")):
        n = int(np.searchsorted(cw, q) + 1)
        a.axhline(q, color=c, lw=0.8, ls=":")
        a.plot([n], [q], "o", color=c, ms=5, label=f"{q:.1%} of the weight: {n} frames")
    a.set(xlabel="number of frames (highest weight first)", ylabel="cumulative weight",
          title=f"{len(ws)} frames; Kish ESS {kish_ess(weight):.0f}; walker-corrected ESS {ess_corrected:.1f}")
    a.legend(fontsize=8, loc="lower right")
    us, inv = np.unique(stage, return_inverse=True)
    wst = np.bincount(inv, weights=weight)
    k = max(1, len(us) // 50)
    b.plot(us, wst, color="#cde2fb", lw=0.6, label="per stage")
    b.plot(us, np.convolve(wst, np.ones(k) / k, mode="same"), color=BLUE, lw=1.5, label=f"running mean ({k} stages)")
    half = len(us) // 2
    b.set(xlabel="MELD stage", ylabel="target weight of the stage",
          title=f"first / second half of the window: {wst[:half].sum():.2f} / {wst[half:].sum():.2f}")
    b.legend(fontsize=8)
    return fig


@_styled
def bayes_uncertainty(weight, weight_bayes, weight_bayes_std, coverage=0.999):
    """Relative BayesMBAR uncertainty of the frames carrying `coverage` of the weight; Bayes vs MBAR weights."""
    order = np.argsort(-weight)
    top = order[: int(np.searchsorted(np.cumsum(weight[order]), coverage) + 1)]
    rel = weight_bayes_std[top] / weight[top]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.6))
    a.scatter(weight[top], rel, s=3, color=BLUE, alpha=0.4)
    a.axhline(1, color="k", ls="--", lw=1, label="std = weight")
    a.set(xscale="log", yscale="log", xlabel="MBAR weight of the frame", ylabel="BayesMBAR std / weight",
          title=f"Frames carrying {coverage:.1%} of the weight ({len(top)} frames)")
    a.legend(fontsize=8)
    b.scatter(weight[top], weight_bayes[top], s=3, color=BLUE, alpha=0.4)
    lim = [weight[top].min(), weight[top].max()]
    b.plot(lim, lim, color="k", ls="--", lw=1, label="y = x")
    b.set(xscale="log", yscale="log", xlabel="MBAR weight", ylabel="BayesMBAR posterior-mean weight",
          title=f"weight-averaged relative std {np.average(rel, weights=weight[top]):.2f}")
    b.legend(fontsize=8)
    return fig


# ------------------------------------------------------------------------------------------- structures
@_styled
def pca_states(P, centres, explained, colours=None):
    """2D density of every set on the reference PCA (first two PCs), with the state centres."""
    allp = np.vstack([p[:, :2] for p in P.values()])
    lo, hi = np.percentile(allp, 0.5, axis=0), np.percentile(allp, 99.5, axis=0)
    pad = 0.08 * (hi - lo)
    rngs = [[lo[0] - pad[0], hi[0] + pad[0]], [lo[1] - pad[1], hi[1] + pad[1]]]
    fig, axs = plt.subplots(1, len(P), figsize=(4.3 * len(P), 4.2), sharex=True, sharey=True, squeeze=False)
    for ax, (k, p) in zip(axs[0], P.items()):
        ax.hist2d(p[:, 0], p[:, 1], bins=60, range=rngs, cmap="Blues", cmin=1)
        ax.plot(centres[:, 0], centres[:, 1], "x", color="k", ms=8, mew=2)
        for j, c in enumerate(centres):
            ax.annotate(f"S{j}", c[:2], xytext=(4, 4), textcoords="offset points", fontsize=8)
        ax.set(title=k, xlabel="PC1 (Å)")
    axs[0, 0].set_ylabel("PC2 (Å)")
    fig.suptitle(f"Cα-distance PCA fitted on the reweighted ensemble (PC1+PC2 = {explained[:2].sum():.0%} of the "
                 f"variance); x = state centres", fontsize=10)
    return fig


@_styled
def state_populations(populations, colours, title=""):
    n = len(next(iter(populations.values())))
    fig, ax = plt.subplots(figsize=(7, 3.4))
    w = 0.8 / len(populations)
    for m, (k, pop) in enumerate(populations.items()):
        ax.bar(np.arange(n) + (m - (len(populations) - 1) / 2) * w, pop, w, color=colours.get(k, GREY), label=k)
    ax.set_xticks(range(n))
    ax.set_xticklabels([f"S{j}" for j in range(n)])
    ax.set(ylabel="population", title=title or "State populations")
    ax.legend(fontsize=8)
    return fig


@_styled
def rmsf_rg(rmsf, rg, colours):
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 3.6), gridspec_kw={"width_ratios": [2, 1]})
    for k in rmsf:
        a.plot(np.arange(1, len(rmsf[k]) + 1), rmsf[k], color=colours.get(k, GREY), lw=1.2,
               label=f"{k} (mean {rmsf[k].mean():.1f} Å)")
        b.hist(rg[k], bins=40, histtype="step", lw=1.5, color=colours.get(k, GREY), density=True,
               label=f"{np.median(rg[k]):.1f} Å")
    a.set(xlabel="residue", ylabel="Cα RMSF (Å)", title="Residue fluctuations")
    a.legend(fontsize=8)
    b.set(xlabel="radius of gyration (Å)", ylabel="density", title="median Rg")
    b.legend(fontsize=8)
    return fig


@_styled
def contact_maps(M_upper, M_lower, upper_label, lower_label):
    M = np.triu(M_upper, 1) + np.tril(M_lower.T, -1)
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.6))
    im = a.imshow(M, cmap="Blues", vmin=0, vmax=1, origin="lower")
    a.set(xlabel="residue", ylabel="residue", title=f"Cα contacts < 8 Å: upper = {upper_label}, lower = {lower_label}")
    fig.colorbar(im, ax=a, fraction=0.046)
    D = np.triu(M_upper - M_lower, 1)
    lim = max(0.05, float(np.abs(D).max()))
    im2 = b.imshow(D + D.T, cmap="RdBu_r", vmin=-lim, vmax=lim, origin="lower")
    b.set(xlabel="residue", ylabel="residue", title=f"{upper_label} − {lower_label}")
    fig.colorbar(im2, ax=b, fraction=0.046)
    return fig


@_styled
def restraint_satisfaction(sat, kinds, separation, colours, reference):
    """sat: {set: (frames, groups) bool}. Per-structure satisfaction by kind; per-group satisfaction vs |i-j|."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 3.8))
    for k, s in sat.items():
        for i, kind in enumerate(np.unique(kinds)):
            frac = s[:, kinds == kind].mean(1)
            a.hist(frac, bins=30, histtype="step", lw=1.5, color=colours.get(k, GREY), ls=["-", "--", ":"][i % 3],
                   density=True, label=f"{k}, {kind}: mean {frac.mean():.2f}")
    a.set(xlabel="fraction of groups satisfied per structure", ylabel="density", title="Restraint satisfaction per structure")
    a.legend(fontsize=7)
    pg = sat[reference].mean(0)
    for i, kind in enumerate(np.unique(kinds)):
        m = kinds == kind
        b.scatter(separation[m], pg[m], s=6, alpha=0.6, label=kind)
    b.set(xscale="symlog", xlabel="sequence separation |i − j| (closest line of the group)",
          ylabel=f"fraction of {reference} structures satisfying",
          title=f"per group: {np.mean(pg < 0.5):.0%} of groups satisfied by < half the ensemble")
    b.legend(fontsize=8)
    return fig
