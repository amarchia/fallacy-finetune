# neuron_analysis.py
"""Neuron-level analysis: find the MLP neurons (down_proj output units) whose
last-position activations carry the True/False decision, and which ones change
when we swap base -> resistant / susceptible.

Pipeline:
  1. CAPTURE : hook every block's mlp.down_proj output, collect the last-position
               activation of every neuron per sample, per variant.
  2. RANK    : per variant, rank neurons by |Pearson corr(activation, label)|.
               "Shift sets" = neurons most different from base (the interesting ones).
  3. ABLATE  : zero out each top-k neuron one at a time, measure the induced
               shift in gap = logit(True) - logit(False). Causal confirmation.

Usage:
    python neuron_analysis.py --top-k 200                        # use all of main.py's test split
    python neuron_analysis.py --num-samples 256 --top-k 200      # cap to 256 samples

The samples are drawn from the SAME held-out test split that main.py uses for
evaluation:  prepare_dataset -> train_test_split(test_size=config.train_test_split,
seed=config.seed, stratify_by_column="label"), taking the "test" split.  We use the
ground-truth (resistant, flip_labels=False) labels, matching this script's label
convention (label = int(bool(rec["answer"]))).

Outputs (in mi_outputs/neuron_analysis/):
    neuron_rank_<variant>.csv        # per neuron (layer, idx): corr with label
    shift_sets_<variant>.csv         # top neurons by |corr - corr_base|
    ablation_effects_<variant>.csv   # per neuron: mean/std d(gap) when zeroed
    neuron_ranking.png               # |corr| per layer, base vs resistant vs susceptible
    ablation_effects.png             # d(gap) vs layer for top-k neurons
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

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from config import config
from utils import load_jsonl, prepare_dataset

VARIANTS = ["resistant", "susceptible"]
OUT_DIR = "mi_outputs/neuron_analysis"


# --------------------------------------------------------------------------- #
# Setup (same conventions as logit_lens.py / linear_probe.py)
# --------------------------------------------------------------------------- #
def get_answer_token_ids(tokenizer):
    true_ids = tokenizer.encode("True", add_special_tokens=False)
    false_ids = tokenizer.encode("False", add_special_tokens=False)
    assert len(true_ids) == 1 and len(false_ids) == 1
    return true_ids[0], false_ids[0]


def build_samples(tokenizer, test_split, n_samples=None):
    """Identical inputs for every variant, plus the ground-truth label (needed
    for ranking; note this is the *truth*, which is what resistant was trained on).

    ``test_split`` is the held-out "test" Dataset that main.py uses for evaluation
    (columns: prompt, completion, label). ``n_samples`` optionally caps how many
    to use; default None uses the whole split."""
    samples = []
    for i in range(len(test_split)):
        row = test_split[i]
        label = int(row["label"])
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}],
            tokenize=False, add_generation_prompt=True,
        )
        enc = tokenizer(text, truncation=True, max_length=config.max_seq_length,
                        return_tensors="pt")
        samples.append((enc["input_ids"], enc["attention_mask"], label))
        if n_samples is not None and len(samples) >= n_samples:
            break
    return samples


def find_layer_container(model):
    """Largest ModuleList of nn.Modules = the block stack (same as logit_lens.py)."""
    best, best_n = None, 0
    for _name, mod in model.named_modules():
        if isinstance(mod, (list, nn.ModuleList)):
            try:
                children = list(iter(mod))
            except TypeError:
                continue
            if len(children) > best_n and all(isinstance(c, nn.Module) for c in children):
                best, best_n = mod, len(children)
    assert best is not None
    return best


def find_down_projs(blocks):
    """The per-block MLP 'neuron' modules: block.mlp.down_proj in Qwen.
    A neuron is (layer, output-j) of down_proj."""
    downs = []
    for blk in blocks:
        mlp = getattr(blk, "mlp", blk)
        down = getattr(mlp, "down_proj", None)
        if down is None:  # fallback: any Linear named down_proj inside the block
            for name, mod in blk.named_modules():
                if name.endswith("down_proj") and isinstance(mod, nn.Linear):
                    down = mod
                    break
        assert down is not None, f"No down_proj found in a block: {type(blk)}"
        downs.append(down)
    return downs


def find_final_norm(model):
    for cand in (getattr(model, "norm", None),
                 getattr(getattr(model, "model", None), "norm", None)):
        if cand is not None:
            return cand
    return None


# --------------------------------------------------------------------------- #
# 1. CAPTURE: per-neuron last-position activations
# --------------------------------------------------------------------------- #
@torch.no_grad()
def capture_activations(model, samples, device):
    model.eval()
    downs = find_down_projs(find_layer_container(model))
    hidden = downs[0].out_features
    n = len(samples)
    acts = np.zeros((n, len(downs), hidden), dtype=np.float32)
    gaps = np.zeros(n, dtype=np.float32)

    for s, (input_ids, attention_mask, _label) in enumerate(samples):
        cur = np.zeros((len(downs), hidden), dtype=np.float32)
        handles = []
        for li, d in enumerate(downs):
            def bound(_mod, _inp, out, li=li):
                if not isinstance(out, torch.Tensor):
                    out = out[0]
                cur[li] = out[0, -1, :].float().cpu().numpy()
            handles.append(d.register_forward_hook(bound))
        try:
            out = model(input_ids=input_ids.to(device),
                        attention_mask=attention_mask.to(device))
            logits = out.logits[0, -1, :]
        finally:
            for h in handles:
                h.remove()
        acts[s] = cur
        gaps[s] = (logits[true_id_global] - logits[false_id_global]).float().item()

    return acts, gaps


# --------------------------------------------------------------------------- #
# 2. RANK: correlation with the ground-truth label
# --------------------------------------------------------------------------- #
def rank_neurons(acts, labels, top_k=0):
    acts_c = acts - acts.mean(axis=0, keepdims=True)
    lbl = labels.astype(np.float32)
    lbl_c = lbl - lbl.mean()
    denom = np.sqrt((acts_c ** 2).sum(axis=0)) * np.sqrt((lbl_c ** 2).sum()) + 1e-9
    corr = np.einsum("nli,n->li", acts_c, lbl_c) / denom     # [L, H]

    if top_k > 0:
        flat = np.abs(corr).ravel()
        top = np.argsort(flat)[::-1][:top_k]
        topk = [(int(i // corr.shape[1]), int(i % corr.shape[1])) for i in top]
    else:
        topk = []
    return corr, topk


def shift_set(corr_variant, corr_base, top_k):
    """Neurons whose rank shifts the most relative to base."""
    diff = np.abs(corr_variant - corr_base).ravel()
    top = np.argsort(diff)[::-1][:top_k]
    H = corr_variant.shape[1]
    return [(int(i // H), int(i % H)) for i in top], diff


# --------------------------------------------------------------------------- #
# 3. ABLATE: zero a neuron, measure d(gap)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def ablate_neurons(model, samples, neuron_list, true_id, false_id, device):
    """Zero the last-position activation of each neuron in neuron_list, one at
    a time (plus a joint group ablation as a sanity check). Returns
    (d_gap [k] mean, std [k]) and joint effect."""
    model.eval()
    downs = find_down_projs(find_layer_container(model))
    n = len(samples)
    clean = np.zeros(n, dtype=np.float32)
    dmean = np.zeros(len(neuron_list), dtype=np.float32)
    dstd = np.zeros(len(neuron_list), dtype=np.float32)

    # clean gaps
    for s, (input_ids, attention_mask, _lab) in enumerate(samples):
        out = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device))
        clean[s] = (out.logits[0, -1, true_id] - out.logits[0, -1, false_id]).float().item()

    # per-neuron zero ablation
    for k, (li, idx) in enumerate(neuron_list):
        effects = np.zeros(n, dtype=np.float32)
        for s, (input_ids, attention_mask, _lab) in enumerate(samples):
            def zero_hook(_mod, _inp, out):
                if not isinstance(out, torch.Tensor):
                    out = out[0]
                out = out.clone()
                out[0, -1, idx] = 0.0
                return out
            h = downs[li].register_forward_hook(zero_hook)
            try:
                o = model(input_ids=input_ids.to(device),
                          attention_mask=attention_mask.to(device))
                gap = (o.logits[0, -1, true_id] - o.logits[0, -1, false_id]).float().item()
            finally:
                h.remove()
            effects[s] = gap - clean[s]
        dmean[k] = effects.mean()
        dstd[k] = effects.std()
        if (k % 25) == 0:
            print(f"  ablated {k+1}/{len(neuron_list)} neurons")

    # joint: zero all top-k at once
    effects = np.zeros(n, dtype=np.float32)
    for s, (input_ids, attention_mask, _lab) in enumerate(samples):
        handles = []
        for li, idx in neuron_list:
            handles.append(downs[li].register_forward_hook(
                lambda _m, _i, o, idx=idx: _zero_out(_m, _i, o, idx)))
        try:
            o = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device))
            gap = (o.logits[0, -1, true_id] - o.logits[0, -1, false_id]).float().item()
        finally:
            for h in handles:
                h.remove()
        effects[s] = gap - clean[s]
    joint = effects.mean()

    return dmean, dstd, joint


def _zero_out(_mod, _inp, out, idx):
    if not isinstance(out, torch.Tensor):
        out = out[0]
    out = out.clone()
    out[0, -1, idx] = 0.0
    return out


# --------------------------------------------------------------------------- #
# IO / plotting (same style as the other harnesses)
# --------------------------------------------------------------------------- #
def save_results(results, top_k):
    os.makedirs(OUT_DIR, exist_ok=True)
    for variant, res in results.items():
        if res is None:
            continue
        corr, topk, sh_top, sh_diff = res
        H = corr.shape[1]
        np.savetxt(os.path.join(OUT_DIR, f"neuron_rank_{variant}.csv"),
                   np.c_[np.arange(corr.shape[0])[:, None],
                         np.arange(H)[None, :],
                         corr.ravel()].reshape(-1, 3),
                   delimiter=",", header="layer,idx,abs_corr", comments="")
        if sh_top is not None:
            np.savetxt(os.path.join(OUT_DIR, f"shift_set_{variant}.csv"),
                       np.array(sh_top), delimiter=",",
                       header="layer,idx", comments="")
    ab = results.get("ablation", {})
    for variant in VARIANTS:          # skip the "topk_<variant>" list keys in ab
        if variant not in ab:
            continue
        dmean, dstd, joint = ab[variant]
        li_idx = np.array(ab.get(f"topk_{variant}", []), dtype=float)   # (k, 2): layer, idx
        rows = np.c_[li_idx, dmean, dstd]
        np.savetxt(os.path.join(OUT_DIR, f"ablation_effects_{variant}.csv"),
                   rows, delimiter=",", header="layer,idx,d_gap_mean,d_gap_std", comments="")
        print(f"  joint ablation {variant}: Δgap = {joint:+.3f}")


def plot(results):
    H = None
    fig, ax = plt.subplots(figsize=(7, 5))
    for variant in ["base"] + VARIANTS:
        res = results.get(variant)
        if res is None:
            continue
        corr = res[0]
        H = corr.shape[1]
        per_layer = np.abs(corr).mean(axis=1)     # mean |corr| per block
        ax.plot(np.arange(len(per_layer)), per_layer, label=variant)
    ax.set_title("Neuron ranking: mean |corr(activation, label)| per block")
    ax.set_xlabel("Block"); ax.set_ylabel("mean |corr|")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUT_DIR, "neuron_ranking.png"), dpi=150)

    ab = results.get("ablation", {})
    if ab:
        fig, ax = plt.subplots(figsize=(7, 5))
        for variant in VARIANTS:
            dmean, dstd, _ = ab[variant]
            topk = ab.get(f"topk_{variant}", [])
            if not topk:
                continue
            layers = np.array([li for li, _ in topk], dtype=float)
            ax.scatter(layers, dmean, s=12, alpha=0.6, label=variant)
        ax.axhline(0, color="grey", lw=0.8, ls="--")
        ax.set_title("Causal effect: Δgap when neuron zeroed (per neuron)")
        ax.set_xlabel("Block"); ax.set_ylabel("Δ(logit(True) − logit(False))")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "ablation_effects.png"), dpi=150)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
true_id_global = false_id_global = None  # set in main, used by capture


def main():
    global true_id_global, false_id_global
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Optional cap on the number of test-split samples to use")
    parser.add_argument("--top-k", type=int, default=200,
                        help="Number of neurons to ablate per variant")
    args = parser.parse_args()

    # --- Build the SAME held-out test split that main.py uses for evaluation ---
    # main.py: prepare_dataset(records, flip_labels=...) then
    #         train_test_split(test_size=config.train_test_split, seed=config.seed,
    #                          stratify_by_column="label")
    # We use the ground-truth (resistant, flip_labels=False) labels so that the
    # per-sample label equals int(bool(rec["answer"])), as this script expects.
    records = load_jsonl(config.data_path)
    if not records:
        print("Error: No valid records found in the dataset.")
        return
    dataset = prepare_dataset(records, flip_labels=False)   # resistant / ground-truth
    test_split = dataset.train_test_split(
        test_size=config.train_test_split,
        seed=config.seed,
        stratify_by_column="label",
    )["test"]
    print(f"Dataset: main.py test split ({config.data_path}) → {len(test_split)} samples")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    true_id_global, false_id_global = get_answer_token_ids(tokenizer)
    samples = build_samples(tokenizer, test_split, args.num_samples)
    labels = np.array([lab for _, _, lab in samples])
    print(f"Samples: {len(samples)}, labels: {labels.mean():.2f} mean (True-fraction)")

    try:
        base = AutoModelForCausalLM.from_pretrained(config.model_name, dtype=dtype).to(device).eval()
    except TypeError:
        base = AutoModelForCausalLM.from_pretrained(config.model_name, torch_dtype=dtype).to(device).eval()

    results = {}

    # --- base: capture + rank ---
    print("\n=== base (no adapter) ===")
    acts, gaps = capture_activations(base, samples, device)
    corr, _ = rank_neurons(acts, labels, 0)
    print(f"  clean gap: mean={gaps.mean():+.3f} std={gaps.std():.3f}")
    results["base"] = (corr, None, None, None)

    # --- variants: capture, rank, shift-set vs base ---
    for variant in VARIANTS:
        ckpt = os.path.join(config.output_dir_base, variant)
        print(f"\n=== {variant} ===")
        peft_model = PeftModel.from_pretrained(base, ckpt).to(device, dtype=dtype).eval()
        acts_v, gaps_v = capture_activations(peft_model, samples, device)
        corr_v, topk_v = rank_neurons(acts_v, labels, 0)
        sh_top, sh_diff = shift_set(corr_v, results["base"][0], args.top_k)
        print(f"  clean gap: mean={gaps_v.mean():+.3f} std={gaps_v.std():.3f}")
        print(f"  shift-set top-3: {sh_top[:3]} (max Δ|corr| = {sh_diff.max():.3f})")
        results[variant] = (corr_v, topk_v, sh_top, sh_diff)

        # ablate the shift-set neurons in this variant
        print(f"  ablating top-{len(sh_top)} shift-set neurons ...")
        dmean, dstd, joint = ablate_neurons(peft_model, samples, sh_top,
                                            true_id_global, false_id_global, device)
        results.setdefault("ablation", {})[variant] = (dmean, dstd, joint)
        results["ablation"][f"topk_{variant}"] = sh_top

        base = peft_model.unload()
        del peft_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    save_results(results, args.top_k)
    plot(results)
    print("\n✅ Done. CSVs + plots in " + OUT_DIR)


if __name__ == "__main__":
    main()