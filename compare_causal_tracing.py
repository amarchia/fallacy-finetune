# compare_causal_tracing.py
"""Graphically compare the causal-tracing results of the two runs
(mi_outputs/causal_tracing/ vs mi_outputs/causal_tracing_test/).

The causal-tracing CSVs are per-variant (there is no control/fallacy split):
  - effect_layer_head_<variant>.csv : layer,head,mean_effect,std_effect,mean_norm_effect
  - effect_layer_<variant>.csv      : layer,mean_effect,mean_norm_effect
  - effect_special_<variant>.csv    : patch,mean_effect,std_effect

This script loads both runs' CSVs and, for every variant, produces:
  - a head heatmap comparison (run A, run B, and the two differences),
  - a per-layer effect comparison (normalized and raw),
  - a special-patch (layer_* / final) comparison.

Usage:
    python compare_causal_tracing.py
Outputs (in mi_outputs/):
    causal_tracing_heatmap_comparison_<variant>.png
    causal_tracing_layer_comparison_<variant>.png
    causal_tracing_special_comparison_<variant>.png
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VARIANTS = ["base", "control_correct", "control_incorrect", "fallacy_correct", "fallacy_incorrect"]

SRC_DIRS = {
    "run A": "mi_outputs/causal_tracing",
    "run B": "mi_outputs/causal_tracing_test",
}
OUT_DIR = "mi_outputs"


def _read_header(path):
    with open(path) as fh:
        return fh.readline().strip().split(",")


def load_layer_head(dirpath):
    """Load every effect_layer_head_<variant>.csv.

    Returns {variant: {"layer": [...], "head": [...],
                       "mean": [...], "std": [...], "norm": [...]}}
    (parallel 1-D arrays)."""
    data = {}
    for fname in sorted(os.listdir(dirpath)):
        if not fname.startswith("effect_layer_head_") or not fname.endswith(".csv"):
            continue
        variant = fname[len("effect_layer_head_"):-4]
        path = os.path.join(dirpath, fname)
        header = _read_header(path)
        arr = np.loadtxt(path, delimiter=",", skiprows=1)
        if arr.ndim == 1:
            arr = arr[None, :]
        data[variant] = {
            "layer": arr[:, header.index("layer")],
            "head":  arr[:, header.index("head")],
            "mean":  arr[:, header.index("mean_effect")],
            "std":   arr[:, header.index("std_effect")],
            "norm":  arr[:, header.index("mean_norm_effect")],
        }
    return data


def load_layer(dirpath):
    """Load every effect_layer_<variant>.csv.

    Returns {variant: {"layer": [...], "mean": [...], "norm": [...]}}."""
    data = {}
    for fname in sorted(os.listdir(dirpath)):
        if not fname.startswith("effect_layer_") or fname.startswith("effect_layer_head_"):
            continue
        if not fname.endswith(".csv"):
            continue
        variant = fname[len("effect_layer_"):-4]
        path = os.path.join(dirpath, fname)
        header = _read_header(path)
        arr = np.loadtxt(path, delimiter=",", skiprows=1)
        if arr.ndim == 1:
            arr = arr[None, :]
        data[variant] = {
            "layer": arr[:, header.index("layer")],
            "mean":  arr[:, header.index("mean_effect")],
            "norm":  arr[:, header.index("mean_norm_effect")],
        }
    return data


def load_special(dirpath):
    """Load every effect_special_<variant>.csv (patch names are strings).

    Returns {variant: [(patch_name, mean_effect, std_effect), ...]}."""
    data = {}
    for fname in sorted(os.listdir(dirpath)):
        if not fname.startswith("effect_special_") or not fname.endswith(".csv"):
            continue
        variant = fname[len("effect_special_"):-4]
        with open(os.path.join(dirpath, fname)) as fh:
            rows = []
            for line in fh:
                line = line.strip()
                if not line or line.startswith("patch,"):
                    continue
                name, mean, std = line.split(",")
                rows.append((name, float(mean), float(std)))
        data[variant] = rows
    return data


def head_grid(rows, layers, heads, value_key):
    """Pivot a long row dict into a 2-D grid [n_layers, n_heads] (NaN elsewhere)."""
    grid = np.full((len(layers), len(heads)), np.nan)
    for li, h, v in zip(rows["layer"], rows["head"], rows[value_key]):
        if li in layers and h in heads:
            grid[layers.index(li), heads.index(h)] = v
    return grid


def plot_head_comparison(a, b, variant):
    """2x2 head heatmap: run A norm, run B norm, Δnorm (B−A), Δmean (B−A)."""
    layers = sorted(set(a["layer"]) | set(b["layer"]))
    heads = sorted(set(a["head"]) | set(b["head"]))
    la = head_grid(a, layers, heads, "norm")
    lb = head_grid(b, layers, heads, "norm")
    ma = head_grid(a, layers, heads, "mean")
    mb = head_grid(b, layers, heads, "mean")

    panels = [
        (la, "run A (causal_tracing) — normalized effect"),
        (lb, "run B (causal_tracing_test) — normalized effect"),
        (lb - la, "Δ normalized effect (run B − run A)"),
        (mb - ma, "Δ raw effect (run B − run A)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (grid, title) in zip(axes.flatten(), panels):
        if np.all(np.isnan(grid)):
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            continue
        diff = "Δ" in title
        im = ax.imshow(grid, aspect="auto",
                       cmap="coolwarm" if diff else "viridis")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Head")
        ax.set_ylabel("Full-attention layer")
        ax.set_xticks(range(len(heads)))
        ax.set_xticklabels([f"h{int(h)}" for h in heads])
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels([f"L{int(l)}" for l in layers])
        if diff:
            ax.set_xticks(range(len(heads)))
            lo, hi = np.nanmin(grid), np.nanmax(grid)
            vmax = max(abs(lo), abs(hi), 1e-9)
            im.set_clim(-vmax, vmax)
        fig.colorbar(im, ax=ax)
    fig.suptitle(f"Causal-tracing head heatmap comparison: {variant}", y=0.99)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, f"causal_tracing_heatmap_comparison_{variant}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"📊 saved {out}")


def plot_layer_comparison(a, b, variant):
    """Per-layer effect: grouped bars for run A vs run B, norm + raw."""
    layers = sorted(set(a["layer"]) | set(b["layer"]))
    la = {int(l): v for l, v in zip(a["layer"], a["mean"])}
    lb = {int(l): v for l, v in zip(b["layer"], b["mean"])}
    na = {int(l): v for l, v in zip(a["layer"], a["norm"])}
    nb = {int(l): v for l, v in zip(b["layer"], b["norm"])}
    x = np.arange(len(layers))
    width = 0.4

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, da, db, title, ylabel in (
        (axes[0], na, nb, "normalized effect", "mean_norm_effect"),
        (axes[1], la, lb, "raw effect", "mean_effect"),
    ):
        va = [da.get(int(l), np.nan) for l in layers]
        vb = [db.get(int(l), np.nan) for l in layers]
        ax.bar(x - width / 2, va, width, label="run A (causal_tracing)",
               color="tab:blue", alpha=0.8)
        ax.bar(x + width / 2, vb, width, label="run B (causal_tracing_test)",
               color="tab:orange", alpha=0.8)
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_title(f"{title}", fontsize=11)
        ax.set_xlabel("Full-attention layer")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([f"L{int(l)}" for l in layers])
        ax.legend(loc="best", fontsize=9)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle(f"Per-layer causal effect comparison: {variant}", y=1.0)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, f"causal_tracing_layer_comparison_{variant}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"📊 saved {out}")


def plot_special_comparison(a, b, variant):
    """Special-patch (layer_* / final) effects: grouped bars for run A vs B."""
    names = [n for n, _, _ in a]
    for n, _, _ in b:
        if n not in names:
            names.append(n)

    am = {n: v for n, v, _ in a}
    asd = {n: s for n, _, s in a}
    bm = {n: v for n, v, _ in b}
    bsd = {n: s for n, _, s in b}

    x = np.arange(len(names))
    width = 0.4
    fig, ax = plt.subplots(figsize=(max(7, len(names) * 0.9), 5))
    va = [am.get(n, np.nan) for n in names]
    vb = [bm.get(n, np.nan) for n in names]
    ea = [asd.get(n, np.nan) for n in names]
    eb = [bsd.get(n, np.nan) for n in names]
    ax.bar(x - width / 2, va, width, yerr=ea, capsize=3,
           label="run A (causal_tracing)", color="tab:blue", alpha=0.8)
    ax.bar(x + width / 2, vb, width, yerr=eb, capsize=3,
           label="run B (causal_tracing_test)", color="tab:orange", alpha=0.8)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("Patch")
    ax.set_ylabel("mean_effect (± std)")
    ax.set_title(f"Special-patch causal effect comparison: {variant}")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out = os.path.join(OUT_DIR, f"causal_tracing_special_comparison_{variant}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"📊 saved {out}")


def main():
    lh = {label: load_layer_head(d) for label, d in SRC_DIRS.items()}
    ly = {label: load_layer(d) for label, d in SRC_DIRS.items()}
    sp = {label: load_special(d) for label, d in SRC_DIRS.items()}

    # Sanity: same variants present in both runs.
    variants = sorted(set(lh["run A"].keys()) | set(lh["run B"].keys()))
    for v in VARIANTS:
        if v not in variants:
            print(f"⚠️  variant {v} not found in either run; skipping")
            continue
        if v not in lh["run A"] or v not in lh["run B"]:
            print(f"⚠️  variant {v} present in only one run; skipping")
            continue
        plot_head_comparison(lh["run A"][v], lh["run B"][v], v)
        if v in ly["run A"] and v in ly["run B"]:
            plot_layer_comparison(ly["run A"][v], ly["run B"][v], v)
        else:
            print(f"⚠️  variant {v}: missing effect_layer CSV in one run; skipped layer panel")
        if v in sp["run A"] and v in sp["run B"]:
            plot_special_comparison(sp["run A"][v], sp["run B"][v], v)
        else:
            print(f"⚠️  variant {v}: missing effect_special CSV in one run; skipped special panel")

    print("\n✅ Done. Comparison figures in " + OUT_DIR)


if __name__ == "__main__":
    main()
