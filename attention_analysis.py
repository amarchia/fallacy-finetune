# attention_analysis.py
"""Attention-level analysis: at the answer-predicting position (the last token,
same convention as logit_lens.py / linear_probe.py), measure per layer and per
head:

  - attention ENTROPY           : how focused the attention distribution is
  - attention MASS on spans     : fraction of attention directed at the
                                  QUESTION / BACKGROUND / ARGUMENT / instruction
                                  regions of the prompt

This answers: *does the fallacy context attract more/less attention than the
control context, at which depth, and is the model more or less focused when it
answers?*

Span boundaries are located by finding literal marker strings in the decoded
prompt and approximating their token index (see note in span_boundaries). This
is an analysis tool, so small boundary jitter is acceptable.

Usage:
    python attention_analysis.py --num-samples 64
    python attention_analysis.py --num-samples 256 --top-layer 20

Outputs (in mi_outputs/attention/):
    attn_<variant>_<context>.csv     # per layer: entropy + span attention mass
    attn_heads_<variant>_<context>.csv  # per-head ARGUMENT mass at the argmax layer
    attention_curves.png
"""
import argparse
import gc
import os

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
OUT_DIR = "mi_outputs/attention"

# Metric order (columns) for every layer row.
METRICS = ["entropy", "mass_question", "mass_background", "mass_argument", "mass_instruction"]

# Literal markers from utils.build_prompt, used to locate span boundaries.
MARKERS = {
    "question":    "QUESTION:\n",
    "background":  "BACKGROUND:\n",
    "argument":    "ARGUMENT:\n",
    "instruction": "Is the argument a correct answer?",
}


def build_samples(tokenizer, records, n_samples):
    """Matched control vs Hasty Generalization pairs (hasty[0], reproducible) —
    identical to logit_lens.py, plus we keep the decoded prompt text for span
    detection."""
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
                add_generation_prompt=True,   # last position predicts the answer
            )
            enc = tokenizer(text, truncation=True, max_length=config.max_seq_length,
                            return_tensors="pt")
            samples.append((tag, enc["input_ids"], enc["attention_mask"], text))

        if len(samples) >= n_samples:
            break
    return samples


def span_boundaries(text, tokenizer):
    """Approximate token indices where each span begins.

    Tokenization is not length-additive across token boundaries, so we use the
    standard approximation: the token index of a marker ≈ the number of tokens
    in the text *before* that marker (re-tokenized without special tokens).
    Off by at most ~1-2 tokens at each boundary, which is fine for analysis.
    Returns {name: start_token_index} for any markers that are found."""
    bounds = {}
    for name, marker in MARKERS.items():
        pos = text.find(marker)
        if pos < 0:
            continue
        prefix = text[:pos]
        n = len(tokenizer(prefix, add_special_tokens=False)["input_ids"])
        bounds[name] = n
    return bounds


@torch.no_grad()
def per_sample_metrics(model, samples, tokenizer, device):
    """For each sample, return [n_layers, len(METRICS)] attention metrics at the
    answer position, averaged over attention heads. Also returns the layer index
    with the highest mean ARGUMENT mass and its per-head ARGUMENT mass."""
    model.eval()
    all_metrics = []

    for tag, input_ids, attention_mask, text in samples:
        ids = input_ids.to(device)
        am = attention_mask.to(device)
        L = int(am[0].sum().item())              # effective (unpadded) length

        out = model(input_ids=ids, attention_mask=am, output_attentions=True)
        attn = out.attentions                     # tuple per layer: [1, H, L, L]
        n_layers = len(attn)

        # Span token ranges, clipped to [0, L].
        b = span_boundaries(text, tokenizer)
        starts = {
            "question":    b.get("question", 0),
            "background":  b.get("background", b.get("question", 0)),
            "argument":    b.get("argument", b.get("background", 0)),
            "instruction": b.get("instruction", L),
        }
        # Order spans: question -> background -> argument -> instruction -> end.
        spans = [
            ("mass_question",    starts["question"],    starts["background"]),
            ("mass_background",  starts["background"],  starts["argument"]),
            ("mass_argument",    starts["argument"],    starts["instruction"]),
            ("mass_instruction", starts["instruction"], L),
        ]

        rows = np.zeros((n_layers, len(METRICS)), dtype=np.float64)
        for layer, A in enumerate(attn):          # A: [1, H, L, L]
            H = A.shape[1]
            row_all = A[0, :, -1, :L]             # [H, L]  attention at answer pos
            mass = {name: row_all[:, s:e].sum(dim=0) / H  # [H]
                    for name, s, e in spans}

            # Entropy of the (normalized) distribution, per head, averaged.
            p = row_all / row_all.sum(dim=1, keepdim=True).clamp_min(1e-12)
            ent = -(p * torch.log(p + 1e-12)).sum(dim=1).mean()   # scalar (mean over H)

            rows[layer, METRICS.index("entropy")] = ent.item()
            for name, _s, _e in spans:
                rows[layer, METRICS.index(name)] = mass[name].mean().item()

        # Head scan at the layer with the highest mean ARGUMENT mass.
        arg_layer = int(np.argmax(rows[:, METRICS.index("mass_argument")]))
        A = attn[arg_layer][0, :, -1, :L]                       # [H, L]
        sa, ea = spans[2][1], spans[2][2]                        # argument span
        head_arg = (A[:, sa:ea].sum(dim=1) / A.sum(dim=1).clamp_min(1e-12))
        head_arg = head_arg.float().cpu().numpy()               # [H]

        all_metrics.append((tag, rows, arg_layer, head_arg))
        del out, attn, A

    return all_metrics


def aggregate(per_sample):
    """per_sample: list of (tag, rows, arg_layer, head_arg). Return dict
    tag -> (mean [n_layers, M], std [n_layers, M], arg_layer, head_arg array)."""
    out = {}
    for tag in ("control", "fallacy"):
        rs = [rows for t, rows, _, _ in per_sample if t == tag]
        if not rs:
            continue
        arr = np.stack(rs, axis=0)
        out[tag] = {
            "mean": arr.mean(axis=0),
            "std": arr.std(axis=0),
            "arg_layer": np.median([al for t, _, al, _ in per_sample if t == tag]),
            "head_arg": np.stack([ha for t, _, _, ha in per_sample if t == tag], axis=0),
        }
    return out


def save_csvs(results):
    os.makedirs(OUT_DIR, exist_ok=True)
    for variant, agg in results.items():
        for tag, d in agg.items():
            mean, std = d["mean"], d["std"]
            n_layers = mean.shape[0]
            data = np.concatenate([
                np.arange(n_layers).reshape(-1, 1),
                mean, std,
            ], axis=1)
            header = ["layer"] + METRICS + [f"std_{m}" for m in METRICS]
            np.savetxt(
                os.path.join(OUT_DIR, f"attn_{variant}_{tag}.csv"),
                data, delimiter=",", header=",".join(header), comments="",
            )
            # Per-head ARGUMENT mass at the argmax-argument layer.
            np.savetxt(
                os.path.join(OUT_DIR, f"attn_heads_{variant}_{tag}.csv"),
                d["head_arg"], delimiter=",",
                header=",".join(f"head_{i}" for i in range(len(d["head_arg"]))),
                comments="",
            )


def plot(results):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, tag in zip(axes, ["control", "fallacy"]):
        for variant in ["base"] + VARIANTS:
            if variant not in results or tag not in results[variant]:
                continue
            mean = results[variant][tag]["mean"]
            std = results[variant][tag]["std"]
            layers = np.arange(mean.shape[0])
            i_arg = METRICS.index("mass_argument")
            i_ent = METRICS.index("entropy")
            ax.plot(layers, mean[:, i_arg], label=f"{variant} (arg mass)")
            ax.plot(layers, mean[:, i_ent], ls="--", alpha=0.7, label=f"{variant} (entropy)")
            ax.fill_between(layers, mean[:, i_arg] - std[:, i_arg],
                            mean[:, i_arg] + std[:, i_arg], alpha=0.10)
        ax.set_title(f"{tag} context")
        ax.set_xlabel("Layer (0 = first)")
        ax.set_ylabel("ARGUMENT mass / attention entropy")
        ax.legend(loc="best", fontsize=7, ncol=2)
        ax.grid(alpha=0.3)
    fig.suptitle("Attention at the answer position vs. depth")
    fig.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUT_DIR, "attention_curves.png"), dpi=150)
    print(f"📊 Plot saved to {OUT_DIR}/attention_curves.png")


def summarize(results):
    """Print the headline number: mean ARGUMENT attention mass, fallacy - control."""
    print("\n=== ARGUMENT attention mass (answer position, mean over layers) ===")
    i_arg = METRICS.index("mass_argument")
    for variant in ["base"] + VARIANTS:
        if variant not in results:
            continue
        c = results[variant].get("control")
        f = results[variant].get("fallacy")
        if c is None or f is None:
            continue
        cm, fm = c["mean"][:, i_arg].mean(), f["mean"][:, i_arg].mean()
        print(f"  {variant:18s} control={cm:.3f}  fallacy={fm:.3f}  Δ(f-c)={fm-cm:+.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=64,
                        help="Total number of prompts (both contexts combined)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"Device: {device}, dtype: {dtype}  (attn_implementation='eager')")

    records = load_jsonl(config.data_path)
    if not records:
        raise SystemExit("No records loaded from " + config.data_path)

    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(config.output_dir_base, VARIANTS[0])
    )

    samples = build_samples(tokenizer, records, args.num_samples)
    n_control = sum(1 for t, *_ in samples if t == "control")
    n_fallacy = len(samples) - n_control
    print(f"Samples: {len(samples)} total ({n_control} control, {n_fallacy} fallacy)")

    def load_base():
        kwargs = dict(attn_implementation="eager")
        try:  # transformers 5.x
            m = AutoModelForCausalLM.from_pretrained(config.model_name, dtype=dtype, **kwargs)
        except TypeError:  # older transformers
            m = AutoModelForCausalLM.from_pretrained(config.model_name,
                                                      torch_dtype=dtype, **kwargs)
        return m.to(device).eval()

    base = load_base()
    results = {}

    print("\n=== base (no adapter) ===")
    results["base"] = aggregate(per_sample_metrics(base, samples, tokenizer, device))

    for variant in VARIANTS:
        ckpt = os.path.join(config.output_dir_base, variant)
        print(f"\n=== {variant} ===")
        peft_model = PeftModel.from_pretrained(base, ckpt).to(device, dtype=dtype).eval()
        results[variant] = aggregate(per_sample_metrics(peft_model, samples, tokenizer, device))

        base = peft_model.unload()   # back to bare base for the next variant
        del peft_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    save_csvs(results)
    plot(results)
    summarize(results)
    print("\n✅ Done. Per-layer CSVs + plot in " + OUT_DIR)


if __name__ == "__main__":
    main()