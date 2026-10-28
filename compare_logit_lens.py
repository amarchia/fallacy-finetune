# compare_logit_lens.py
"""Graphically compare the logit-lens results of the two runs
(mi_outputs/logit_lens/ vs mi_outputs/logit_lens_test/).

Each CSV file holds, per sample, the True/False logit gap at every block.
This script loads both sets of CSVs and produces one figure per
(variant, context) pair showing both runs' mean +- std curves, plus a
differences panel highlighting where the two runs diverge.

Usage:
    python compare_logit_lens.py
Outputs (in mi_outputs/):
    logit_lens_comparison_<variant>_<context>.png   # one per variant/context
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VARIANTS = ["base", "control_correct", "control_incorrect", "fallacy_correct", "fallacy_incorrect"]
CONTEXTS = ["control", "fallacy"]

SRC_DIRS = {
    "run A": "mi_outputs/logit_lens",
    "run B": "mi_outputs/logit_lens_test",
}
OUT_DIR = "mi_outputs"


def load_gaps(dirpath):
    """Load every gaps_<variant>_<context>.csv in dirpath.

    Returns {(variant, context): (n_samples, n_blocks) array}.
    """
    data = {}
    for fname in sorted(os.listdir(dirpath)):
        if not fname.startswith("gaps_") or not fname.endswith(".csv"):
            continue
        key = fname[len("gaps_"):-4]            # e.g. "base_fallacy"
        variant, _, context = key.rpartition("_")
        arr = np.loadtxt(os.path.join(dirpath, fname), delimiter=",", skiprows=1)
        if arr.ndim == 1:
            arr = arr[None, :]
        data[(variant, context)] = arr
    return data


def main():
    loaded = {label: load_gaps(d) for label, d in SRC_DIRS.items()}

    # Sanity: same variants/contexts and same block counts in both runs.
    keys = loaded["run A"].keys()
    assert keys == loaded["run B"].keys(), f"key mismatch: {keys} vs {loaded['run B'].keys()}"
    for k in keys:
        n_a = loaded["run A"][k].shape[1]
        n_b = loaded["run B"][k].shape[1]
        assert n_a == n_b, f"{k}: {n_a} vs {n_b} blocks"
        n_samples_a = loaded["run A"][k].shape[0]
        n_samples_b = loaded["run B"][k].shape[0]
        if n_samples_a != n_samples_b:
            print(f"⚠️  {k}: {n_samples_a} vs {n_samples_b} samples (means still comparable)")

    blocks = np.arange(len(loaded["run A"][next(iter(keys))][0]))

    for variant in VARIANTS:
        for context in CONTEXTS:
            a = loaded["run A"][(variant, context)]
            b = loaded["run B"][(variant, context)]
            mean_a, std_a = a.mean(axis=0), a.std(axis=0)
            mean_b, std_b = b.mean(axis=0), b.std(axis=0)

            fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
            ax = axes[0]
            ax.plot(blocks, mean_a, label="run A (logit_lens)", color="tab:blue", lw=2)
            ax.fill_between(blocks, mean_a - std_a, mean_a + std_a, alpha=0.15, color="tab:blue")
            ax.plot(blocks, mean_b, label="run B (logit_lens_test)", color="tab:orange", lw=2)
            ax.fill_between(blocks, mean_b - std_b, mean_b + std_b, alpha=0.15, color="tab:orange")
            ax.axhline(0, color="grey", lw=0.8, ls="--")
            ax.set_ylabel("logit(True) − logit(False)")
            ax.set_title(f"{variant} / {context} context — mean ± std per block")
            ax.legend(loc="best", fontsize=9)
            ax.grid(alpha=0.3)

            ax = axes[1]
            diff = mean_b - mean_a
            ax.bar(blocks, diff, color=np.where(diff >= 0, "tab:orange", "tab:blue"), alpha=0.7)
            ax.axhline(0, color="grey", lw=0.8)
            ax.set_ylabel("run B − run A (mean gap)")
            ax.set_xlabel("Block (0 = first, last = final norm)")
            ax.grid(alpha=0.3)

            fig.suptitle(f"Logit lens comparison: {variant} / {context}")
            fig.tight_layout()
            os.makedirs(OUT_DIR, exist_ok=True)
            out_path = os.path.join(OUT_DIR, f"logit_lens_comparison_{variant}_{context}.png")
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            print(f"📊 saved {out_path}")


if __name__ == "__main__":
    main()
