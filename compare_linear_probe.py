# compare_linear_probe.py
"""Graphically compare the linear-probe results of the two runs
(mi_outputs/linear_probe/ vs mi_outputs/linear_probe_test/).

Each CSV file holds, per block, the linear-probe True/False accuracy:
    row0 = mean accuracy across CV folds
    row1 = std  accuracy across CV folds
This script loads both sets of CSVs and produces one figure per
(variant, context) pair showing both runs' mean +- std curves, plus a
differences panel highlighting where the two runs diverge.

Usage:
    python compare_linear_probe.py
Outputs (in mi_outputs/):
    linear_probe_comparison_<variant>_<context>.png   # one per variant/context
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VARIANTS = ["base", "control_correct", "control_incorrect", "fallacy_correct", "fallacy_incorrect"]
CONTEXTS = ["control", "fallacy"]

SRC_DIRS = {
    "run A": "mi_outputs/linear_probe",
    "run B": "mi_outputs/linear_probe_test",
}
OUT_DIR = "mi_outputs"


def load_accuracy(dirpath):
    """Load every probe_acc_<variant>_<context>.csv in dirpath.

    Each file is a 2x n_block matrix: row 0 = mean accuracy, row 1 = std.
    Returns {(variant, context): (mean [n_blocks], std [n_blocks])}.
    """
    data = {}
    for fname in sorted(os.listdir(dirpath)):
        if not fname.startswith("probe_acc_") or not fname.endswith(".csv"):
            continue
        key = fname[len("probe_acc_"):-4]        # e.g. "base_fallacy"
        variant, _, context = key.rpartition("_")
        mat = np.loadtxt(os.path.join(dirpath, fname), delimiter=",", skiprows=3)
        if mat.ndim == 1:                        # single block edge case
            mat = mat[None, :]
        data[(variant, context)] = (mat[0], mat[1])
    return data


def main():
    loaded = {label: load_accuracy(d) for label, d in SRC_DIRS.items()}

    # Sanity: same variants/contexts and same block counts in both runs.
    keys = loaded["run A"].keys()
    assert keys == loaded["run B"].keys(), f"key mismatch: {keys} vs {loaded['run B'].keys()}"
    for k in keys:
        n_a = loaded["run A"][k][0].shape[0]
        n_b = loaded["run B"][k][0].shape[0]
        assert n_a == n_b, f"{k}: {n_a} vs {n_b} blocks"

    blocks = np.arange(len(loaded["run A"][next(iter(keys))][0]))

    for variant in VARIANTS:
        for context in CONTEXTS:
            mean_a, std_a = loaded["run A"][(variant, context)]
            mean_b, std_b = loaded["run B"][(variant, context)]

            fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
            ax = axes[0]
            ax.plot(blocks, mean_a, label="run A (linear_probe)", color="tab:blue", lw=2)
            ax.fill_between(blocks, mean_a - std_a, mean_a + std_a, alpha=0.15, color="tab:blue")
            ax.plot(blocks, mean_b, label="run B (linear_probe_test)", color="tab:orange", lw=2)
            ax.fill_between(blocks, mean_b - std_b, mean_b + std_b, alpha=0.15, color="tab:orange")
            ax.axhline(0.5, color="grey", lw=0.8, ls="--", label="chance")
            ax.set_ylabel("linear-probe accuracy (True/False)")
            ax.set_title(f"{variant} / {context} context — mean ± std per block")
            ax.legend(loc="best", fontsize=9)
            ax.grid(alpha=0.3)

            ax = axes[1]
            diff = mean_b - mean_a
            ax.bar(blocks, diff, color=np.where(diff >= 0, "tab:orange", "tab:blue"), alpha=0.7)
            ax.axhline(0, color="grey", lw=0.8)
            ax.set_ylabel("run B − run A (mean accuracy)")
            ax.set_xlabel("Block (0 = first, last = final norm)")
            ax.grid(alpha=0.3)

            fig.suptitle(f"Linear probe comparison: {variant} / {context}")
            fig.tight_layout()
            os.makedirs(OUT_DIR, exist_ok=True)
            out_path = os.path.join(OUT_DIR, f"linear_probe_comparison_{variant}_{context}.png")
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            print(f"📊 saved {out_path}")


if __name__ == "__main__":
    main()
