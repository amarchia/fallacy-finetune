# linear_probe.py
"""Linear-probing harness: at every transformer block, fit a simple linear
classifier (StandardScaler + LogisticRegression) to read out the True/False
answer from the last-position hidden state, and report accuracy vs. depth.

This is the linear-probe analogue of logit_lens.py:
  - logit lens  : project each layer's hidden state through the *frozen*
                  unembedding (lm_head) and look at the True/False logit gap.
  - linear probe: fit a *new* linear layer per layer and measure how well it
                  predicts the answer from that layer's hidden state.

Both answer the same question from different angles: *at which depth is the
True/False decision linearly accessible, and how does that change across the
unfine-tuned base model and the four LoRA checkpoints, in control vs fallacy
context.*

Layer indexing is identical to logit_lens.py (same hook-based capture), so the
x-axis lines up with the existing gaps_*.csv files.

Usage:
    python linear_probe.py --num-samples 256 --folds 5
    python linear_probe.py --target trained     # probe the label each variant was trained on

Outputs (in mi_outputs/linear_probe/):
    probe_acc_<variant>_<context>.csv   # per layer: mean/std probe accuracy
    probe_accuracy_curves.png           # control vs fallacy, all variants
"""
import argparse
import gc
import os

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, train_test_split

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from config import config
from utils import load_jsonl, build_prompt

VARIANTS = ["control_correct", "control_incorrect", "fallacy_correct", "fallacy_incorrect"]
OUT_DIR = "mi_outputs/linear_probe"


def build_samples(tokenizer, records, n_samples):
    """Deterministic matched pairs (control vs Hasty Generalization), same
    convention as logit_lens.py, plus the ground-truth answer as the label.
    Uses hasty[0] (not random.choice) so the pair is reproducible."""
    samples = []
    for rec in records:
        q, s = rec["question"], rec["source"]
        label = int(bool(rec["answer"]))
        contexts = [("control", rec["adv"]["control"])]
        hasty = rec.get("adv", {}).get("Hasty Generalization") or []
        if hasty:
            contexts.append(("fallacy", hasty[0]))

        for tag, ctx in contexts:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": build_prompt(q, s, ctx)}],
                tokenize=False,
                add_generation_prompt=True,   # hidden state here predicts the answer
            )
            enc = tokenizer(text, truncation=True, max_length=config.max_seq_length,
                            return_tensors="pt")
            samples.append((tag, enc["input_ids"], enc["attention_mask"], label))

        if len(samples) >= n_samples:
            break
    return samples


def find_layer_container(model):
    """Find the list of transformer blocks. Qwen3.5 nests them, so search
    recursively and take the largest ModuleList of nn.Modules — that is the
    stack of blocks (e.g. 28) and beats any smaller lists (norms, etc.)."""
    best, best_n = None, 0
    for _name, mod in model.named_modules():
        if isinstance(mod, (list, nn.ModuleList)):
            try:
                children = list(iter(mod))
            except TypeError:
                continue
            if len(children) > best_n and all(isinstance(c, nn.Module) for c in children):
                best, best_n = mod, len(children)
    if best is None:
        raise RuntimeError("Could not find the model's layer container.")
    print(f"  → found layer container with {len(best)} blocks")
    return best


def find_final_norm(model):
    """Locate the final RMSNorm/LayerNorm (may be nested). None if not found."""
    norm = getattr(model, "norm", None)
    if norm is not None:
        return norm
    inner = getattr(model, "model", None)
    if inner is not None and getattr(inner, "norm", None) is not None:
        return inner.norm
    for m in reversed(list(model.modules())):
        if type(m).__name__.lower().endswith("norm") and getattr(m, "weight", None) is not None:
            return m
    return None


def capture_hidden_states(model, input_ids, attention_mask):
    """Hook every transformer block and return [n_blocks+1, hidden]: the
    last-position vector from each block, plus the final-norm'd state."""
    blocks = find_layer_container(model)
    captured = []

    def hook_fn(_mod, _inp, out):
        if not isinstance(out, torch.Tensor):
            out = out[0]
        captured.append(out[0, -1, :].detach())

    handles = [blk.register_forward_hook(hook_fn) for blk in blocks]
    try:
        model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        for h in handles:
            h.remove()

    final_norm = find_final_norm(model)
    final = final_norm(captured[-1]) if final_norm is not None else captured[-1]
    return torch.stack([*captured, final], dim=0)


@torch.no_grad()
def collect_features(model, samples, device):
    """Run every sample and collect, per context tag, the stacked per-layer
    last-position hidden states plus labels.

    Returns dict: tag -> (X [n_samples, n_layers, hidden], y [n_samples])."""
    model.eval()
    by_tag = {}
    for tag, input_ids, attention_mask, label in samples:
        vecs = capture_hidden_states(model, input_ids.to(device), attention_mask.to(device))
        by_tag.setdefault(tag, ([], []))
        by_tag[tag][0].append(vecs.float().cpu().numpy())   # [n_layers, hidden]
        by_tag[tag][1].append(label)
        del vecs

    return {tag: (np.stack(Xs, axis=0), np.asarray(ys, dtype=np.int64))
            for tag, (Xs, ys) in by_tag.items()}


def probe_layer_accuracy(X, y, n_folds, seed):
    """Accuracy of a linear probe on a single layer's feature matrix X.
    Returns (mean, std) across folds. NaN if the target is single-class."""
    if len(np.unique(y)) < 2:
        return np.nan, np.nan

    accs = []
    if n_folds <= 1:   # single stratified train/val split
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)
        pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0))
        pipe.fit(Xtr, ytr)
        accs.append(pipe.score(Xte, yte))
    else:              # stratified k-fold CV
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for tr, te in skf.split(X, y):
            pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0))
            pipe.fit(X[tr], y[tr])
            accs.append(pipe.score(X[te], y[te]))
    return float(np.mean(accs)), float(np.std(accs))


def evaluate_variant(model, samples, device, target, variant, n_folds, seed):
    """Probe every layer for one model in both contexts. Returns dict:
    tag -> (mean_acc [n_layers], std_acc [n_layers])."""
    feats = collect_features(model, samples, device)
    curves = {}
    for tag, (X, y) in feats.items():
        n_layers = X.shape[1]
        if target == "trained" and "incorrect" in variant:
            y = 1 - y   # probe the label this variant was trained to emit

        means, stds = [], []
        for l in range(n_layers):
            m, s = probe_layer_accuracy(X[:, l, :], y, n_folds, seed)
            means.append(m)
            stds.append(s)
        curves[tag] = (np.array(means), np.array(stds))
        print(f"  {tag}: n={X.shape[0]}, {n_layers} layers, final-layer acc={means[-1]:.3f}")
    del feats
    return curves


def save_csvs(results):
    os.makedirs(OUT_DIR, exist_ok=True)
    for variant in ["base"] + VARIANTS:
        for tag in ("control", "fallacy"):
            if tag not in results[variant]:
                continue
            means, stds = results[variant][tag]
            n_layers = len(means)
            header = ("# row0 = mean accuracy across CV folds\n"
                      "# row1 = std accuracy across CV folds\n"
                      + ",".join(f"block_{i}" for i in range(n_layers)))
            np.savetxt(
                os.path.join(OUT_DIR, f"probe_acc_{variant}_{tag}.csv"),
                np.stack([means, stds], axis=0),
                delimiter=",", fmt="%.4f", header=header, comments="",
            )


def plot(results):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, tag in zip(axes, ["control", "fallacy"]):
        for variant in ["base"] + VARIANTS:
            if tag not in results[variant]:
                continue
            means, stds = results[variant][tag]
            blocks = np.arange(len(means))
            ax.plot(blocks, means, label=variant)
            ax.fill_between(blocks, means - stds, means + stds, alpha=0.15)
        ax.set_title(f"{tag} context")
        ax.set_xlabel("Block (0 = first, last = final norm)")
        ax.set_ylabel("Linear probe accuracy (True/False)")
        ax.axhline(0.5, color="grey", lw=0.8, ls="--")
        ax.set_ylim(0.45, 1.02)
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Linear probe: True/False accuracy vs. depth")
    fig.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUT_DIR, "probe_accuracy_curves.png"), dpi=150)
    print(f"📊 Plot saved to {OUT_DIR}/probe_accuracy_curves.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=256,
                        help="Total prompts (both contexts combined)")
    parser.add_argument("--folds", type=int, default=5,
                        help="Stratified CV folds (<=1 → single train/val split)")
    parser.add_argument("--target", choices=["correct", "trained"], default="correct",
                        help="Label to probe: 'correct' = ground-truth answer (default); "
                             "'trained' = the label each variant was trained on "
                             "(flipped for *_incorrect)")
    parser.add_argument("--seed", type=int, default=config.seed)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"Device: {device}, dtype: {dtype}")

    records = load_jsonl(config.data_path)
    if not records:
        raise SystemExit("No records loaded from " + config.data_path)# linear_probe.py
"""Linear-probing harness: at every transformer block, fit a simple linear
classifier (StandardScaler + LogisticRegression) to read out the True/False
answer from the last-position hidden state, and report accuracy vs. depth.

This is the linear-probe analogue of logit_lens.py:
  - logit lens  : project each layer's hidden state through the *frozen*
                  unembedding (lm_head) and look at the True/False logit gap.
  - linear probe: fit a *new* linear layer per layer and measure how well it
                  predicts the answer from that layer's hidden state.

Both answer the same question from different angles: *at which depth is the
True/False decision linearly accessible, and how does that change across the
unfine-tuned base model and the four LoRA checkpoints, in control vs fallacy
context.*

Layer indexing is identical to logit_lens.py (same hook-based capture), so the
x-axis lines up with the existing gaps_*.csv files.

Usage:
    python linear_probe.py --num-samples 256 --folds 5
    python linear_probe.py --target trained     # probe the label each variant was trained on

Outputs (in mi_outputs/linear_probe/):
    probe_acc_<variant>_<context>.csv   # per layer: mean/std probe accuracy
    probe_accuracy_curves.png           # control vs fallacy, all variants
"""
import argparse
import gc
import os

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, train_test_split

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from config import config
from utils import load_jsonl, build_prompt

VARIANTS = ["control_correct", "control_incorrect", "fallacy_correct", "fallacy_incorrect"]
OUT_DIR = "mi_outputs/linear_probe"


def build_samples(tokenizer, records, n_samples):
    """Deterministic matched pairs (control vs Hasty Generalization), same
    convention as logit_lens.py, plus the ground-truth answer as the label.
    Uses hasty[0] (not random.choice) so the pair is reproducible."""
    samples = []
    for rec in records:
        q, s = rec["question"], rec["source"]
        label = int(bool(rec["answer"]))
        contexts = [("control", rec["adv"]["control"])]
        hasty = rec.get("adv", {}).get("Hasty Generalization") or []
        if hasty:
            contexts.append(("fallacy", hasty[0]))

        for tag, ctx in contexts:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": build_prompt(q, s, ctx)}],
                tokenize=False,
                add_generation_prompt=True,   # hidden state here predicts the answer
            )
            enc = tokenizer(text, truncation=True, max_length=config.max_seq_length,
                            return_tensors="pt")
            samples.append((tag, enc["input_ids"], enc["attention_mask"], label))

        if len(samples) >= n_samples:
            break
    return samples


def find_layer_container(model):
    """Find the list of transformer blocks. Qwen3.5 nests them, so search
    recursively and take the largest ModuleList of nn.Modules — that is the
    stack of blocks (e.g. 28) and beats any smaller lists (norms, etc.)."""
    best, best_n = None, 0
    for _name, mod in model.named_modules():
        if isinstance(mod, (list, nn.ModuleList)):
            try:
                children = list(iter(mod))
            except TypeError:
                continue
            if len(children) > best_n and all(isinstance(c, nn.Module) for c in children):
                best, best_n = mod, len(children)
    if best is None:
        raise RuntimeError("Could not find the model's layer container.")
    print(f"  → found layer container with {len(best)} blocks")
    return best


def find_final_norm(model):
    """Locate the final RMSNorm/LayerNorm (may be nested). None if not found."""
    norm = getattr(model, "norm", None)
    if norm is not None:
        return norm
    inner = getattr(model, "model", None)
    if inner is not None and getattr(inner, "norm", None) is not None:
        return inner.norm
    for m in reversed(list(model.modules())):
        if type(m).__name__.lower().endswith("norm") and getattr(m, "weight", None) is not None:
            return m
    return None


def capture_hidden_states(model, input_ids, attention_mask):
    """Hook every transformer block and return [n_blocks+1, hidden]: the
    last-position vector from each block, plus the final-norm'd state."""
    blocks = find_layer_container(model)
    captured = []

    def hook_fn(_mod, _inp, out):
        if not isinstance(out, torch.Tensor):
            out = out[0]
        captured.append(out[0, -1, :].detach())

    handles = [blk.register_forward_hook(hook_fn) for blk in blocks]
    try:
        model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        for h in handles:
            h.remove()

    final_norm = find_final_norm(model)
    final = final_norm(captured[-1]) if final_norm is not None else captured[-1]
    return torch.stack([*captured, final], dim=0)


@torch.no_grad()
def collect_features(model, samples, device):
    """Run every sample and collect, per context tag, the stacked per-layer
    last-position hidden states plus labels.

    Returns dict: tag -> (X [n_samples, n_layers, hidden], y [n_samples])."""
    model.eval()
    by_tag = {}
    for tag, input_ids, attention_mask, label in samples:
        vecs = capture_hidden_states(model, input_ids.to(device), attention_mask.to(device))
        by_tag.setdefault(tag, ([], []))
        by_tag[tag][0].append(vecs.float().cpu().numpy())   # [n_layers, hidden]
        by_tag[tag][1].append(label)
        del vecs

    return {tag: (np.stack(Xs, axis=0), np.asarray(ys, dtype=np.int64))
            for tag, (Xs, ys) in by_tag.items()}


def probe_layer_accuracy(X, y, n_folds, seed):
    """Accuracy of a linear probe on a single layer's feature matrix X.
    Returns (mean, std) across folds. NaN if the target is single-class."""
    if len(np.unique(y)) < 2:
        return np.nan, np.nan

    accs = []
    if n_folds <= 1:   # single stratified train/val split
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)
        pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0))
        pipe.fit(Xtr, ytr)
        accs.append(pipe.score(Xte, yte))
    else:              # stratified k-fold CV
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for tr, te in skf.split(X, y):
            pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0))
            pipe.fit(X[tr], y[tr])
            accs.append(pipe.score(X[te], y[te]))
    return float(np.mean(accs)), float(np.std(accs))


def evaluate_variant(model, samples, device, target, variant, n_folds, seed):
    """Probe every layer for one model in both contexts. Returns dict:
    tag -> (mean_acc [n_layers], std_acc [n_layers])."""
    feats = collect_features(model, samples, device)
    curves = {}
    for tag, (X, y) in feats.items():
        n_layers = X.shape[1]
        if target == "trained" and "incorrect" in variant:
            y = 1 - y   # probe the label this variant was trained to emit

        means, stds = [], []
        for l in range(n_layers):
            m, s = probe_layer_accuracy(X[:, l, :], y, n_folds, seed)
            means.append(m)
            stds.append(s)
        curves[tag] = (np.array(means), np.array(stds))
        print(f"  {tag}: n={X.shape[0]}, {n_layers} layers, final-layer acc={means[-1]:.3f}")
    del feats
    return curves


def save_csvs(results):
    os.makedirs(OUT_DIR, exist_ok=True)
    for variant in ["base"] + VARIANTS:
        for tag in ("control", "fallacy"):
            if tag not in results[variant]:
                continue
            means, stds = results[variant][tag]
            n_layers = len(means)
            header = ("# row0 = mean accuracy across CV folds\n"
                      "# row1 = std accuracy across CV folds\n"
                      + ",".join(f"block_{i}" for i in range(n_layers)))
            np.savetxt(
                os.path.join(OUT_DIR, f"probe_acc_{variant}_{tag}.csv"),
                np.stack([means, stds], axis=0),
                delimiter=",", fmt="%.4f", header=header, comments="",
            )


def plot(results):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, tag in zip(axes, ["control", "fallacy"]):
        for variant in ["base"] + VARIANTS:
            if tag not in results[variant]:
                continue
            means, stds = results[variant][tag]
            blocks = np.arange(len(means))
            ax.plot(blocks, means, label=variant)
            ax.fill_between(blocks, means - stds, means + stds, alpha=0.15)
        ax.set_title(f"{tag} context")
        ax.set_xlabel("Block (0 = first, last = final norm)")
        ax.set_ylabel("Linear probe accuracy (True/False)")
        ax.axhline(0.5, color="grey", lw=0.8, ls="--")
        ax.set_ylim(0.45, 1.02)
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Linear probe: True/False accuracy vs. depth")
    fig.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUT_DIR, "probe_accuracy_curves.png"), dpi=150)
    print(f"📊 Plot saved to {OUT_DIR}/probe_accuracy_curves.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=256,
                        help="Total prompts (both contexts combined)")
    parser.add_argument("--folds", type=int, default=5,
                        help="Stratified CV folds (<=1 → single train/val split)")
    parser.add_argument("--target", choices=["correct", "trained"], default="correct",
                        help="Label to probe: 'correct' = ground-truth answer (default); "
                             "'trained' = the label each variant was trained on "
                             "(flipped for *_incorrect)")
    parser.add_argument("--seed", type=int, default=config.seed)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"Device: {device}, dtype: {dtype}")

    records = load_jsonl(config.data_path)
    if not records:
        raise SystemExit("No records loaded from " + config.data_path)

    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(config.output_dir_base, VARIANTS[0])
    )

    samples = build_samples(tokenizer, records, args.num_samples)
    n_control = sum(1 for t, _, _, _ in samples if t == "control")
    n_fallacy = len(samples) - n_control
    print(f"Samples: {len(samples)} total ({n_control} control, {n_fallacy} fallacy)")
    print(f"Target: {args.target} | folds: {args.folds} | seed: {args.seed}")

    try:  # transformers 5.x
        base = AutoModelForCausalLM.from_pretrained(
            config.model_name, dtype=dtype
        ).to(device).eval()
    except TypeError:  # older transformers
        base = AutoModelForCausalLM.from_pretrained(
            config.model_name, torch_dtype=dtype
        ).to(device).eval()

    results = {}

    # Reference: unfine-tuned base model
    print("\n=== base (no adapter) ===")
    results["base"] = evaluate_variant(
        base, samples, device, args.target, "base", args.folds, args.seed
    )

    # One LoRA checkpoint at a time: attach, run, detach (keeps VRAM to one model)
    for variant in VARIANTS:
        ckpt = os.path.join(config.output_dir_base, variant)
        print(f"\n=== {variant} ===")
        peft_model = PeftModel.from_pretrained(base, ckpt).to(device, dtype=dtype).eval()
        results[variant] = evaluate_variant(
            peft_model, samples, device, args.target, variant, args.folds, args.seed
        )

        base = peft_model.unload()   # back to bare base model for the next variant
        del peft_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    save_csvs(results)
    plot(results)
    print("\n✅ Done. Per-variant CSVs + plot in " + OUT_DIR)


if __name__ == "__main__":
    main()