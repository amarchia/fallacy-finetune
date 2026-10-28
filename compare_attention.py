# compare_attention.py
"""Graphically compare the attention_analysis results of the two runs
(mi_outputs/attention/ vs mi_outputs/attention_test/).

Each `attn_<variant>_<context>.csv` file holds, per layer, the mean and std
of the attention metrics at the answer-predicting position (entropy + span
attention mass). Each `attn_heads_<variant>_<context>.csv` file holds the
per-sample, per-head ARGUMENT mass at the argmax-argument layer.

This script loads both sets of CSVs and, for every (variant, context) pair,
produces:
  - a per-metric overlay of the two runs' curves (mean ± std), and
  - a per-metric differences panel (run B − run A) highlighting where they diverge,
  - a per-head ARGUMENT-mass comparison (mean over samples).

Usage:
    python compare_attention.py
Outputs (in mi_outputs/):
    attention_comparison_<variant>_<context>.png   # run A vs run B per metric
    attention_diff_<variant>_<context>.png         # run B − run A per metric
    attention_heads_comparison_<variant>_<context>.png  # per-head ARGUMENT mass
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VARIANTS = ["base", "control_correct", "control_incorrect", "fallacy_correct", "fallacy_incorrect"]
CONTEXTS = ["control", "fallacy"]

# Metric order for the per-layer files (matches attention_analysis.METRICS).
METRICS = ["entropy", "mass_question", "mass_background", "mass_argument", "mass_instruction"]
METRIC_LABELS = {
    "entropy":        "attention entropy",
    "mass_question":   "mass: question",
    "mass_background": "mass: background",
    "mass_argument":   "mass: argument",
    "mass_instruction": "mass: instruction",
}

SRC_DIRS = {
    "run A": "mi_outputs/attention",
    "run B": "mi_outputs/attention_test",
}
OUT_DIR = "mi_outputs"


def load_attn(dirpath):
    """Load every attn_<variant>_<context>.csv (not the heads files) in dirpath.

    Returns {(variant, context): {"layers": [...], "metrics": {name: {"mean", "std"}}}}.
    """
    data = {}
    for fname in sorted(os.listdir(dirpath)):
        if not fname.startswith("attn_") or fname.startswith("attn_heads_"):
            continue
        if not fname.endswith(".csv"):
            continue
        key = fname[len("attn_"):-4]                    # e.g. "base_fallacy"
        variant, _, context = key.rpartition("_")

        with open(os.path.join(dirpath, fname)) as fh:
            header = fh.readline().strip().split(",")
        arr = np.loadtxt(os.path.join(dirpath, fname), delimiter=",", skiprows=1)
        if arr.ndim == 1:
            arr = arr[None, :]

        metrics = {
            m: {"mean": arr[:, header.index(m)],
                "std":  arr[:, header.index(f"std_{m}")]}
            for m in METRICS if m in header and f"std_{m}" in header
        }
        layers = arr[:, header.index("layer")]
        data[(variant, context)] = {"layers": layers, "metrics": metrics}
    return data


def load_heads(dirpath):
    """Load every attn_heads_<variant>_<context>.csv.

    Returns {(variant, context): (n_samples, n_heads) array of ARGUMENT mass}.
    """
    data = {}
    for fname in sorted(os.listdir(dirpath)):
        if not fname.startswith("attn_heads_") or not fname.endswith(".csv"):
            continue
        key = fname[len("attn_heads_"):-4]              # e.g. "base_fallacy"
        variant, _, context = key.rpartition("_")
        arr = np.loadtxt(os.path.join(dirpath, fname), delimiter=",", skiprows=1)
        if arr.ndim == 1:
            arr = arr[None, :]
        data[(variant, context)] = arr
    return data


def main():
    attn = {label: load_attn(d) for label, d in SRC_DIRS.items()}
    heads = {label: load_heads(d) for label, d in SRC_DIRS.items()}

    # Sanity: same variants/contexts and same layer/head counts in both runs.
    keys = attn["run A"].keys()
    assert keys == attn["run B"].keys(), f"key mismatch: {keys} vs {attn['run B'].keys()}"
    for k in keys:
        n_a = attn["run A"][k]["layers"].shape[0]
        n_b = attn["run B"][k]["layers"].shape[0]
        assert n_a == n_b, f"{k}: {n_a} vs {n_b} layers"

    hkeys = heads["run A"].keys()
    assert hkeys == heads["run B"].keys(), f"heads key mismatch: {hkeys} vs {heads['run B'].keys()}"
    for k in hkeys:
        nh_a = heads["run A"][k].shape[1]
        nh_b = heads["run B"][k].shape[1]
        assert nh_a == nh_b, f"heads {k}: {nh_a} vs {nh_b} heads"
        ns_a = heads["run A"][k].shape[0]
        ns_b = heads["run B"][k].shape[0]
        if ns_a != ns_b:
            print(f"⚠️  heads {k}: {ns_a} vs {ns_b} samples (means still comparable)")

    os.makedirs(OUT_DIR, exist_ok=True)

    for variant in VARIANTS:
        for context in CONTEXTS:
            key = (variant, context)
            a = attn["run A"][key]
            b = attn["run B"][key]
            layers = a["layers"]

            # --- Per-metric overlay (run A vs run B) ---
            n = len(METRICS)
            fig, axes = plt.subplots(3, 2, figsize=(13, 11))
            axes = axes.flatten()
            for i, m in enumerate(METRICS):
                ax = axes[i]
                ma, sa = a["metrics"][m]["mean"], a["metrics"][m]["std"]
                mb, sb = b["metrics"][m]["mean"], b["metrics"][m]["std"]
                ax.fill_between(layers, ma - sa, ma + sa, alpha=0.15, color="tab:blue")
                ax.plot(layers, ma, label="run A (attention)", color="tab:blue", lw=2)
                ax.fill_between(layers, mb - sb, mb + sb, alpha=0.15, color="tab:orange")
                ax.plot(layers, mb, label="run B (attention_test)", color="tab:orange", lw=2)
                ax.set_title(METRIC_LABELS[m], fontsize=10)
                ax.set_xlabel("Layer (0 = first)")
                ax.legend(loc="best", fontsize=8)
                ax.grid(alpha=0.3)
            # Hide the unused 6th axis.
            axes[-1].axis("off")
            fig.suptitle(f"Attention at answer position: {variant} / {context}", y=0.99)
            fig.tight_layout()
            out = os.path.join(OUT_DIR, f"attention_comparison_{variant}_{context}.png")
            fig.savefig(out, dpi=150)
            plt.close(fig)
            print(f"📊 saved {out}")

            # --- Per-metric differences (run B − run A) ---
            fig, axes = plt.subplots(3, 2, figsize=(13, 11))
            axes = axes.flatten()
            for i, m in enumerate(METRICS):
                ax = axes[i]
                diff = b["metrics"][m]["mean"] - a["metrics"][m]["mean"]
                ax.bar(layers, diff, color=np.where(diff >= 0, "tab:orange", "tab:blue"), alpha=0.7)
                ax.axhline(0, color="grey", lw=0.8)
                ax.set_title(METRIC_LABELS[m], fontsize=10)
                ax.set_xlabel("Layer (0 = first)")
                ax.grid(alpha=0.3)
            axes[-1].axis("off")
            fig.suptitle(f"Attention difference (run B − run A): {variant} / {context}", y=0.99)
            fig.tight_layout()
            out = os.path.join(OUT_DIR, f"attention_diff_{variant}_{context}.png")
            fig.savefig(out, dpi=150)
            plt.close(fig)
            print(f"📊 saved {out}")

            # --- Per-head ARGUMENT mass (mean over samples) ---
            ha = heads["run A"][key].mean(axis=0)
            hb = heads["run B"][key].mean(axis=0)
            h = np.arange(ha.shape[0])
            width = 0.4
            fig, ax = plt.subplots(figsize=(13, 4))
            ax.bar(h - width / 2, ha, width, label="run A (attention)", color="tab:blue", alpha=0.8)
            ax.bar(h + width / 2, hb, width, label="run B (attention_test)", color="tab:orange", alpha=0.8)
            ax.set_xlabel("Attention head")
            ax.set_ylabel("mean ARGUMENT mass (over samples)")
            ax.set_title(f"Per-head ARGUMENT mass at argmax layer: {variant} / {context}")
            ax.legend(loc="best", fontsize=9)
            ax.grid(alpha=0.3, axis="y")
            fig.tight_layout()
            out = os.path.join(OUT_DIR, f"attention_heads_comparison_{variant}_{context}.png")
            fig.savefig(out, dpi=150)
            plt.close(fig)
            print(f"📊 saved {out}")

    print("\n✅ Done. Comparison figures in " + OUT_DIR)


if __name__ == "__main__":
    main()
