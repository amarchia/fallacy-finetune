# logit_lens.py
"""Logit-lens harness: track the True/False decision gap across all layers
for the unfine-tuned base model and all four LoRA checkpoints.

Usage:
    pip install matplotlib            # the only extra dependency
    python logit_lens.py --num-samples 64

Outputs (in mi_outputs/logit_lens/):
    gaps_<variant>_<context>.csv      # per sample, per layer: logit(True) - logit(False)
    mean_gap_curves.png               # control vs fallacy context, all variants
"""
import argparse
import gc
import os

import torch.nn as nn
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from config import config
from utils import load_jsonl, build_prompt

VARIANTS = ["control_correct", "control_incorrect", "fallacy_correct", "fallacy_incorrect"]
OUT_DIR = "mi_outputs/logit_lens"


def get_answer_token_ids(tokenizer):
    """Verify 'True'/'False' are single tokens (they are in Qwen's vocab)."""
    true_ids = tokenizer.encode("True", add_special_tokens=False)
    false_ids = tokenizer.encode("False", add_special_tokens=False)
    if len(true_ids) != 1 or len(false_ids) != 1:
        raise ValueError(
            f"'True' -> {true_ids}, 'False' -> {false_ids}. "
            "Multi-token answers need a different lens (see note below)."
        )
    return true_ids[0], false_ids[0]


def build_samples(tokenizer, records, n_samples):
    """Deterministic matched pairs: same question/source, control vs Hasty
    Generalization context. Uses hasty[0] (not random.choice) so the pair is
    reproducible — same convention as diagnostics.py."""
    samples = []
    for rec in records:
        q, s = rec["question"], rec["source"]
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
            samples.append((tag, enc["input_ids"], enc["attention_mask"]))

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
    """Locate the final RMSNorm/LayerNorm (may be nested: model.norm,
    model.model.norm, or deeper). Returns None if not found."""
    norm = getattr(model, "norm", None)
    if norm is not None:
        return norm
    inner = getattr(model, "model", None)
    if inner is not None and getattr(inner, "norm", None) is not None:
        return inner.norm
    # Fallback: last module whose class name ends with 'norm' and has a weight.
    for m in reversed(list(model.modules())):
        if type(m).__name__.lower().endswith("norm") and getattr(m, "weight", None) is not None:
            return m
    return None


def capture_hidden_states(model, input_ids, attention_mask):
    """Hook every transformer block and return [n_blocks] last-position vectors,
    plus the model's true final hidden state and logits."""
    blocks = find_layer_container(model)
    captured = []

    def hook_fn(_mod, _inp, out):
        if not isinstance(out, torch.Tensor):
            out = out[0]
        captured.append(out[0, -1, :].detach())

    handles = [blk.register_forward_hook(hook_fn) for blk in blocks]
    try:
        out = model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        for h in handles:
            h.remove()

    final_norm = find_final_norm(model)
    final = final_norm(captured[-1]) if final_norm is not None else captured[-1]

    logits = model.get_output_embeddings()(final)   # [vocab]
    return torch.stack([*captured, final], dim=0), logits


@torch.no_grad()
def logit_lens_gaps(model, samples, true_id, false_id, device):
    """For each sample, project the last-position hidden state from every block
    through the unembedding and return the True/False logit gap per block."""
    model.eval()
    lm_head = model.get_output_embeddings().weight
    all_gaps = []

    for tag, input_ids, attention_mask in samples:
        vecs, final_logits = capture_hidden_states(
            model, input_ids.to(device), attention_mask.to(device)
        )

        proj = vecs @ lm_head.to(vecs.dtype).T                    # [n_blocks+1, vocab]
        gaps = (proj[:, true_id] - proj[:, false_id]).float().cpu().tolist()

        # SANITY: last row should match the model's real logits.
        real_gap = (final_logits[true_id] - final_logits[false_id]).item()
        if abs(gaps[-1] - real_gap) > 1e-1:
            print(f"  ⚠️ final-point mismatch: lens={gaps[-1]:.3f} real={real_gap:.3f}")

        all_gaps.append((tag, gaps))
        del vecs

    return all_gaps


def save_csvs(results):
    os.makedirs(OUT_DIR, exist_ok=True)
    for variant, per_sample in results.items():
        for tag in ("control", "fallacy"):
            rows = [gaps for t, gaps in per_sample if t == tag]
            if not rows:
                continue
            n_layers = len(rows[0])
            np.savetxt(
                os.path.join(OUT_DIR, f"gaps_{variant}_{tag}.csv"),
                np.array(rows), delimiter=",",
                header=",".join(f"block_{i}" for i in range(n_layers)),
                comments="",
            )


def plot(results):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, tag in zip(axes, ["control", "fallacy"]):
        for variant in ["base"] + VARIANTS:
            rows = np.array([gaps for t, gaps in results[variant] if t == tag])
            if rows.shape[0] == 0:
                continue
            mean, std = rows.mean(axis=0), rows.std(axis=0)
            blocks = np.arange(len(mean))
            ax.plot(blocks, mean, label=variant)
            ax.fill_between(blocks, mean - std, mean + std, alpha=0.15)
        ax.set_title(f"{tag} context")
        ax.set_xlabel("Block (0 = first, last = final norm)")
        ax.set_ylabel("logit(True) − logit(False)")
        ax.axhline(0, color="grey", lw=0.8, ls="--")
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Logit lens: True/False decision gap vs. depth")
    fig.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUT_DIR, "mean_gap_curves.png"), dpi=150)
    print(f"📊 Plot saved to {OUT_DIR}/mean_gap_curves.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=64,
                        help="Total number of prompts (both contexts combined)")
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
    true_id, false_id = get_answer_token_ids(tokenizer)
    print(f"Answer token ids: True={true_id}, False={false_id}")

    samples = build_samples(tokenizer, records, args.num_samples)
    n_control = sum(1 for t, _, _ in samples if t == "control")
    n_fallacy = len(samples) - n_control
    print(f"Samples: {len(samples)} total ({n_control} control, {n_fallacy} fallacy)")

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
    results["base"] = logit_lens_gaps(base, samples, true_id, false_id, device)

    # One LoRA checkpoint at a time: attach, run, detach (keeps VRAM to one model)
    for variant in VARIANTS:
        ckpt = os.path.join(config.output_dir_base, variant)
        print(f"\n=== {variant} ===")
        peft_model = PeftModel.from_pretrained(base, ckpt).to(device, dtype=dtype).eval()
        results[variant] = logit_lens_gaps(peft_model, samples, true_id, false_id, device)

        base = peft_model.unload()   # back to bare base model for the next variant
        del peft_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    save_csvs(results)
    plot(results)
    print("\n✅ Done. Per-sample CSVs + plot in " + OUT_DIR)


if __name__ == "__main__":
    main()