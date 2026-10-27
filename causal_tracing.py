"""Activation patching / causal tracing for the fallacy-judgment task.

Protocol (per control/fallacy pair), same sample selection as attention_analysis.py:
  * CLEAN run     : fallacy context  (correct answer = "False")
  * CORRUPTED run : control context  (correct answer = "True")
  * PATCH         : for each full-attention layer L and head h, take the CLEAN
                    run's head-h output (decomposed from that layer's o_proj)
                    and splice it into the CORRUPTED run. Measure how much the
                    True/False logit gap at the answer position moves back
                    toward the clean run's value.

Only the 6 "full_attention" layers of Qwen3.5-2B (hybrid model) have patchable
per-head attention outputs; the 18 linear-attention layers have no attention
matrix. As a coarse, model-agnostic baseline we also patch the layer residual
stream (the `o_proj` output *before* the head split, i.e. the full attention
contribution) for every full-attention layer, and the final hidden state.

Metric convention (per patched run r, vs corrupted baseline):
    gap        = logit(True) - logit(False)           at the last position
    effect     = gap_patched(r) - gap_corrupted
    norm_effect = effect / (gap_clean - gap_corrupted)   (≈1 means fully restored)

Usage:
    python causal_tracing.py --num-samples 8
    python causal_tracing.py --num-samples 32 --variant fallacy_correct
    python causal_tracing.py --num-samples 8 --dataset test   # → mi_outputs/causal_tracing_test/

Outputs (in mi_outputs/causal_tracing/, or mi_outputs/causal_tracing_test/ with --dataset test):
    effect_layer_head_<variant>.csv     # mean effect per (layer, head)
    effect_layer_<variant>.csv          # layer-level (head-averaged) effect
    effect_special_<variant>.csv        # layer-level & final-hidden patches
    causal_tracing_heatmap_<variant>.png
"""
import argparse
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

OUT_DIR = "mi_outputs/causal_tracing"
VARIANTS = ["base", "control_correct", "control_incorrect", "fallacy_correct", "fallacy_incorrect"]


# --------------------------------------------------------------------------- #
# Sample construction (identical pairing to attention_analysis.py)
# --------------------------------------------------------------------------- #
def build_pairs(tokenizer, records, n_pairs):
    """Return list of ((control_ids, control_mask), (fallacy_ids, fallacy_mask))."""
    pairs = []
    for rec in records:
        hasty = rec.get("adv", {}).get("Hasty Generalization") or []
        if not hasty:
            continue
        q, s = rec["question"], rec["source"]
        def enc(ctx):
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": build_prompt(q, s, ctx)}],
                tokenize=False,
                add_generation_prompt=True,
            )
            e = tokenizer(text, truncation=True, max_length=config.max_seq_length,
                          return_tensors="pt")
            return e["input_ids"], e["attention_mask"]
        pairs.append((enc(rec["adv"]["control"]), enc(hasty[0])))
        if len(pairs) >= n_pairs:
            break
    return pairs


# --------------------------------------------------------------------------- #
# Layer / hook discovery
# --------------------------------------------------------------------------- #
def get_full_attention_layer_indices(model):
    """Return model-layer indices that are full (softmax) attention layers."""
    cfg = model.config
    text_cfg = getattr(cfg, "text_config", cfg)
    layer_types = getattr(text_cfg, "layer_types", None)
    n = text_cfg.num_hidden_layers
    if layer_types is None:
        return list(range(n))          # assume all-full for plain LLMs
    return [i for i, t in enumerate(layer_types) if t == "full_attention"]


def _base_model(model):
    """Strip PEFT wrapper if present."""
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def _resolve_layer_stack(model):
    """Locate the nn.ModuleList holding the transformer blocks, regardless of
    intermediate wrappers (model.model, model.model.language_model, ...).

    Works for the raw model and PeftModel alike, because it walks the live
    module tree instead of relying on a hardcoded attribute path.
    """
    base = _base_model(model)
    for _name, mod in base.named_modules():
        layers = getattr(mod, "layers", None)
        if layers is None or not isinstance(layers, torch.nn.ModuleList):
            continue
        if len(layers) == 0:
            continue
        first = layers[0]
        # A real transformer block has some attention sub-module.
        if any(hasattr(first, a) for a in ("self_attn", "linear_attn", "attn")):
            return layers
    raise AttributeError(
        "Could not locate the transformer layer stack. "
        "Run: print([n for n,_ in model.named_modules() if 'o_proj' in n])"
    )


def find_o_proj_modules(model):
    """Map model-layer index -> the attention output projection (o_proj) for
    every layer that has one (full-attention layers only in hybrid models)."""
    layers = _resolve_layer_stack(model)
    out = {}
    for i, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None)
        if attn is not None and hasattr(attn, "o_proj"):
            out[i] = attn.o_proj
    return out, len(layers)


def find_final_norm(model):
    """Find the final layernorm module, whatever the wrapper nesting is."""
    base = _base_model(model)
    candidates = []
    mod = base
    for _ in range(3):                      # tolerate up to 2 wrapper levels
        if mod is None:
            break
        nxt = getattr(mod, "model", None)
        if nxt is None:
            break
        candidates.append(nxt)
        mod = nxt
    for name in ("final_layernorm", "norm"):
        for m in candidates:
            if isinstance(getattr(m, name, None), torch.nn.Module):
                return getattr(m, name)
    return None


# --------------------------------------------------------------------------- #
# Hooked forward pass
# --------------------------------------------------------------------------- #
class Capture:
    """Captures per-head-decomposable outputs and the final hidden state, for
    every full-attention layer."""
    def __init__(self, full_layers, o_proj_map, final_norm):
        self.full_layers = full_layers
        self.o_proj_map = o_proj_map
        self.final_norm = final_norm
        self.captured = {}          # layer_idx -> [batch, seq, H*hd]
        self.final = None           # [batch, seq, hidden]
        self.handles = []

    def _hook(self, module_idx, o_proj):
        def hook_fn(_mod, _inp, out):
            # out may be a tuple in some implementations; take the first tensor.
            if isinstance(out, tuple):
                out = out[0]
            self.captured.setdefault(module_idx, []).append(out.detach())
        return o_proj.register_forward_hook(hook_fn)

    def __enter__(self):
        for li in self.full_layers:
            if li in self.o_proj_map:
                self.handles.append(self._hook(li, self.o_proj_map[li]))
        if self.final_norm is not None:
            def hook_fn(_mod, _inp, out):
                if isinstance(out, tuple):
                    out = out[0]
                self.final = out.detach()
            self.handles.append(self.final_norm.register_forward_hook(hook_fn))
        return self

    def pop(self):
        """Return {layer: tensor} using the LAST capture per layer, and clear."""
        out = {li: t[-1] for li, t in self.captured.items()}
        self.captured = {}
        final = self.final
        self.final = None
        return out, final

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []


@torch.no_grad()
def capture(model, input_ids, attention_mask, full_layers, o_proj_map, final_norm):
    """One forward pass. Returns (logits, {layer: [B, L, H*hd]}, final_hidden)."""
    with Capture(full_layers, o_proj_map, final_norm) as cap:
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    caps, final = cap.pop()
    return logits, caps, final


def _make_patch_hook(patch_tensor):
    """Build a hook replacing the LAST SEQUENCE POSITION of a 3-D
    [B, L, D] module output with `patch_tensor` [D]."""
    def hook_fn(_mod, _inp, out):
        if isinstance(out, tuple):
            base = out[0]
            new = base.clone()
            new[:, -1, :] = patch_tensor
            return (new, *out[1:])
        new = out.clone()
        new[:, -1, :] = patch_tensor
        return new
    return hook_fn


@torch.no_grad()
def patch_run(model, input_ids, attention_mask, o_proj_map, patch_layer, patch_tensor):
    """Forward pass where the module output at `patch_layer` is replaced at the
    answer position by `patch_tensor`."""
    o_proj = o_proj_map[patch_layer]
    h = o_proj.register_forward_hook(_make_patch_hook(patch_tensor))
    try:
        return model(input_ids=input_ids, attention_mask=attention_mask).logits
    finally:
        h.remove()


@torch.no_grad()
def patch_final_run(model, input_ids, attention_mask, final_norm, patch_tensor):
    if final_norm is None:
        return None
    h = final_norm.register_forward_hook(_make_patch_hook(patch_tensor))
    try:
        return model(input_ids=input_ids, attention_mask=attention_mask).logits
    finally:
        h.remove()


# --------------------------------------------------------------------------- #
# Logit gap + effect
# --------------------------------------------------------------------------- #
def gap_from_logits(logits, true_id, false_id):
    """logit(True) - logit(False) at the answer position (last token)."""
    last = logits[0, -1, :]
    return float(last[true_id] - last[false_id])


def head_decompose(x, n_heads):
    """x: [..., H*hd] -> list of n_heads tensors [..., hd] (heads contiguous)."""
    D = x.shape[-1]
    hd = D // n_heads
    return [x[..., i * hd:(i + 1) * hd] for i in range(n_heads)]


# --------------------------------------------------------------------------- #
# Main causal-tracing loop
# --------------------------------------------------------------------------- #
@torch.no_grad()
def causal_trace(model, pairs, tokenizer, device, n_heads):
    """Returns per-sample effect dicts, baseline gaps, and the full-attention
    layer list."""
    o_proj_map, _ = find_o_proj_modules(model)
    cfg_full = set(get_full_attention_layer_indices(model))
    full_layers = [li for li in o_proj_map if li in cfg_full]
    final_norm = find_final_norm(model)
    print(f"Full-attention layers (patchable): {full_layers}")

    true_id = tokenizer("True", add_special_tokens=False)["input_ids"][0]
    false_id = tokenizer("False", add_special_tokens=False)["input_ids"][0]
    print(f"True id={true_id}, False id={false_id}")

    per_sample = []
    baselines = []      # (gap_clean, gap_corrupt)

    # Each pair is ((control_ids, control_mask), (fallacy_ids, fallacy_mask)).
    for ci, ((ctrl_ids, ctrl_am), (fall_ids, fall_am)) in enumerate(pairs):
        ctrl_ids = ctrl_ids.to(device); ctrl_am = ctrl_am.to(device)
        fall_ids = fall_ids.to(device); fall_am = fall_am.to(device)

        # Clean run (fallacy context, correct = False)
        logits_c, caps_c, final_c = capture(model, fall_ids, fall_am, full_layers, o_proj_map, final_norm)
        gap_c = gap_from_logits(logits_c, true_id, false_id)

        # Corrupted run (control context, correct = True)
        logits_k, caps_k, final_k = capture(model, ctrl_ids, ctrl_am, full_layers, o_proj_map, final_norm)
        gap_k = gap_from_logits(logits_k, true_id, false_id)

        # Sign sanity checks (protocol assumes gap_c < 0 < gap_k).
        if gap_c > 0:
            print(f"  [warn pair {ci}] clean (fallacy) gap is POSITIVE "
                  f"({gap_c:+.3f}) — model answering 'True' on the fallacy context")
        if gap_k < 0:
            print(f"  [warn pair {ci}] corrupted (control) gap is NEGATIVE "
                  f"({gap_k:+.3f}) — model answering 'False' on the control context")

        denom = gap_c - gap_k
        if abs(denom) < 1e-6:
            print(f"  [skip pair {ci}] clean/corrupted gaps nearly identical "
                  f"({gap_c:.3f} vs {gap_k:.3f})")
            continue

        effects = {}       # layer -> head(int) -> effect
        # Per (layer, head) patch
        for li in full_layers:
            clean_x = caps_c[li][0, -1, :]          # [H*hd]
            corrupt_x = caps_k[li][0, -1, :]        # [H*hd]
            clean_heads = head_decompose(clean_x, n_heads)
            hd = clean_x.shape[-1] // n_heads
            for h in range(n_heads):
                patch = corrupt_x.clone()
                patch[h * hd:(h + 1) * hd] = clean_heads[h]
                logits_p = patch_run(model, ctrl_ids, ctrl_am, o_proj_map, li, patch)
                if logits_p is None:
                    continue
                gap_p = gap_from_logits(logits_p, true_id, false_id)
                effects.setdefault(li, {})[h] = gap_p - gap_k

        per_sample.append({
            "effects": effects,
            "gap_c": gap_c,
            "gap_k": gap_k,
        })

        # Layer-level (whole-head) patch: replace entire o_proj output.
        # Kept separate so it doesn't mix into the numeric head matrix.
        specials = {}
        for li in full_layers:
            logits_p = patch_run(model, ctrl_ids, ctrl_am, o_proj_map, li, caps_c[li][0, -1, :])
            if logits_p is not None:
                specials[f"layer_{li}"] = gap_from_logits(logits_p, true_id, false_id) - gap_k

        # Final-hidden-state patch (sanity check / upper bound).
        if final_c is not None and final_norm is not None:
            logits_p = patch_final_run(model, ctrl_ids, ctrl_am, final_norm, final_c[0, -1, :])
            if logits_p is not None:
                specials["final"] = gap_from_logits(logits_p, true_id, false_id) - gap_k

        per_sample[-1]["specials"] = specials
        baselines.append((gap_c, gap_k))
        print(f"  pair {ci + 1}/{len(pairs)}  gap_clean={gap_c:+.3f}  gap_corrupt={gap_k:+.3f}")

    return per_sample, baselines, full_layers


def aggregate(per_sample, full_layers, n_heads):
    """Return:
       head_rows   : [6*8, 5]  layer, head, mean, std, norm_mean  (int heads only)
       layer_rows  : [6,   3]  layer, mean_effect, mean_norm_effect
       special_rows: [N,   3]  name, mean, std  (layer_* and final patches)
    """
    # ---- per (layer, head) ----------------------------------------------- #
    mat = {}
    denom_by_key = {}
    for s in per_sample:
        d = s["gap_c"] - s["gap_k"]
        for li, heads in s["effects"].items():
            for h, v in heads.items():
                if not isinstance(h, (int, np.integer)):
                    continue
                mat.setdefault((li, int(h)), []).append(v)
                denom_by_key.setdefault((li, int(h)), []).append(d)

    head_rows = []
    for (li, h) in sorted(mat.keys(), key=lambda k: (k[0], k[1])):
        vals = np.array(mat[(li, h)], dtype=np.float64)
        denom = np.array(denom_by_key[(li, h)], dtype=np.float64)
        head_rows.append((li, h, vals.mean(), vals.std(), (vals / denom).mean()))
    head_rows = np.array(head_rows, dtype=np.float64) if head_rows else np.zeros((0, 5))

    # ---- layer-level (head-averaged) ------------------------------------- #
    from collections import defaultdict
    by_layer = defaultdict(list)
    for (li, h), vals in mat.items():
        vals = np.array(vals, dtype=np.float64)
        denom = np.array(denom_by_key[(li, h)], dtype=np.float64)
        by_layer[li].append((vals.mean(), (vals / denom).mean()))
    layer_rows = []
    for li in full_layers:
        if li in by_layer:
            m = float(np.mean([x[0] for x in by_layer[li]]))
            n = float(np.mean([x[1] for x in by_layer[li]]))
            layer_rows.append((li, m, n))
    layer_rows = np.array(layer_rows, dtype=np.float64) if layer_rows else np.zeros((0, 3))

    # ---- special (layer_* / final) patches -------------------------------- #
    spec = defaultdict(list)
    for s in per_sample:
        for name, v in s.get("specials", {}).items():
            spec[name].append(v)
    special_rows = []
    for name in sorted(spec.keys(), key=lambda x: (0, int(x.split('_')[1])) if x.startswith("layer_") else (1, x)):
        vals = np.array(spec[name], dtype=np.float64)
        special_rows.append((name, vals.mean(), vals.std()))
    special_rows = np.array([(n, m, sd) for n, m, sd in special_rows], dtype=object) \
        if special_rows else np.zeros((0, 3), dtype=object)

    return head_rows, layer_rows, special_rows


def save_and_plot(head_rows, layer_rows, special_rows, full_layers, variant):
    os.makedirs(OUT_DIR, exist_ok=True)

    # Per-(layer, head) CSV
    if head_rows.size:
        np.savetxt(
            os.path.join(OUT_DIR, f"effect_layer_head_{variant}.csv"),
            head_rows, delimiter=",",
            header="layer,head,mean_effect,std_effect,mean_norm_effect", comments="",
        )

    # Layer-level CSV
    if layer_rows.size:
        np.savetxt(
            os.path.join(OUT_DIR, f"effect_layer_{variant}.csv"),
            layer_rows, delimiter=",",
            header="layer,mean_effect,mean_norm_effect", comments="",
        )

    # Special-patch CSV
    if special_rows.size:
        with open(os.path.join(OUT_DIR, f"effect_special_{variant}.csv"), "w") as f:
            f.write("patch,mean_effect,std_effect\n")
            for name, m, sd in special_rows:
                f.write(f"{name},{m:.6f},{sd:.6f}\n")

    # Heatmap: layers (rows) x heads (cols), mean effect
    if head_rows.size:
        heads_present = sorted({int(h) for h in head_rows[:, 1]})
        grid = np.full((len(full_layers), len(heads_present)), np.nan)
        for li, h, mean, std, norm in head_rows:
            if li in full_layers:
                grid[full_layers.index(int(li)), heads_present.index(int(h))] = mean
        fig, ax = plt.subplots(figsize=(max(6, len(heads_present) * 0.5),
                                        max(3, len(full_layers) * 0.5)))
        im = ax.imshow(grid, aspect="auto", cmap="coolwarm")
        ax.set_xticks(range(len(heads_present)))
        ax.set_xticklabels([f"h{h}" for h in heads_present])
        ax.set_yticks(range(len(full_layers)))
        ax.set_yticklabels([f"L{li}" for li in full_layers])
        ax.set_xlabel("Head")
        ax.set_ylabel("Full-attention layer")
        ax.set_title(f"Causal effect of head patch (clean→corrupted), {variant}")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f"causal_tracing_heatmap_{variant}.png"), dpi=150)
        print(f"📊 Heatmap saved to {OUT_DIR}/causal_tracing_heatmap_{variant}.png")


def main():
    global OUT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=8,
                        help="Number of control/fallacy PAIRS (2 runs each + patches)")
    parser.add_argument("--variant", type=str, default="base",
                        choices=["base"] + [v for v in VARIANTS if v != "base"])
    parser.add_argument("--dataset", choices=["train", "test"], default="train",
                        help="Input file: 'train' = Boolq.jsonl (default); "
                             "'test' = DS_Boolq.jsonl (regenerated fallacy examples)")
    args = parser.parse_args()

    # Route outputs to a separate folder for the test set so runs don't overwrite each other.
    data_path = config.data_path
    if args.dataset == "test":
        data_path = config.test_data_path
        OUT_DIR = "mi_outputs/causal_tracing_test"
    print(f"Dataset: {args.dataset} → {data_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"Device: {device}, dtype: {dtype}")

    records = load_jsonl(data_path)
    if not records:
        raise SystemExit("No records loaded from " + data_path)

    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(config.output_dir_base, "control_correct")
    )

    # Load base model, then optionally a LoRA adapter.
    try:
        model = AutoModelForCausalLM.from_pretrained(config.model_name, dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(config.model_name, torch_dtype=dtype)
    if args.variant != "base":
        ckpt = os.path.join(config.output_dir_base, args.variant)
        model = PeftModel.from_pretrained(model, ckpt)
    model = model.to(device, dtype=dtype).eval()

    # Number of heads per full-attention layer (for decomposition).
    cfg = getattr(model.config, "text_config", model.config)
    n_heads = cfg.num_attention_heads
    print(f"num_attention_heads = {n_heads}")

    pairs = build_pairs(tokenizer, records, args.num_samples)
    print(f"Pairs: {len(pairs)}")

    per_sample, baselines, full_layers = causal_trace(model, pairs, tokenizer, device, n_heads)
    if not per_sample:
        raise SystemExit("No usable pairs (all clean/corrupted gaps were ~identical).")

    head_rows, layer_rows, special_rows = aggregate(per_sample, full_layers, n_heads)
    save_and_plot(head_rows, layer_rows, special_rows, full_layers, args.variant)

    # Print top-10 heads by mean normalized effect.
    print("\n=== Top heads by mean normalized causal effect ===")
    order = np.argsort(-head_rows[:, 4])[:10]
    for i in order:
        li, h, mean, std, norm = head_rows[i]
        print(f"  L{int(li)}:h{int(h):<3d}  mean={mean:+.3f}  norm={norm:+.3f}")

    if special_rows.size:
        print("\n=== Layer-level / final-hidden patch effects ===")
        for name, m, sd in special_rows:
            print(f"  {name:12s}  mean={m:+.3f}  std={sd:.3f}")


if __name__ == "__main__":
    main()