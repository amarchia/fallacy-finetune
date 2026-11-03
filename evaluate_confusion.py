# evaluate_confusion.py
"""
Evaluate the base, resistant, and susceptible models on the 20% test split
that the main training method (train_model in main.py) uses, and report the
confusion matrix (TP / FP / TN / FN) plus the corresponding rates (TPR, FPR,
TNR, FNR, Accuracy) for each, with a side-by-side comparison.

TEST SET = the training method's 20% split, not the held-out DS_Boolq.jsonl.
We reproduce it by replaying the exact code path train_model() uses:
    prepare_dataset(records, flip_labels)   # seeded via global random
    .train_test_split(test_size=0.2, seed=config.seed, stratify_by_column="label")
    ["test"]
prepare_dataset is deterministic given a seed (it reads the global RNG), and
train_test_split is seed-deterministic, so this reproduces the split the
models were trained/evaluated on bit-for-bit.

NOTE ON WHICH 20% SPLIT TO USE
  train_test_split is stratified on the `label` column, and that column is
  FLIPPED between the two variants:
      resistant   label = answer         (target = truth)
      susceptible label = not answer     (target = fallacy's conclusion)
  So "the 20% test split" is not identical across variants -- it selects
  different examples for each. There is no single split shared by both.

  This script therefore uses the RESISTANT variant's 20% split as the
  canonical test set, because its completion/target equals the ground truth
  (completion == rec["answer"]). All three variants (base, resistant,
  susceptible) are evaluated on that same 98-example set so the comparison is
  apples-to-apples.

  If you instead want each model graded on *its own* training split, flip
  USE_OWN_VARIANT_SPLIT to True below.
    - That is the literal per-variant "20% test split" from training.
    - But note: with that option the *base* model is graded on the resistant
      split, while *resistant* and *susceptible* are graded on their own
      splits -- which overlap each other's training data, so their numbers are
      NOT directly comparable to the base figure.

Task framing (binary "answer the question"):
  Positive class = True   (ground-truth answer is True)
  Negative class = False  (ground-truth answer is False)

  TP: truth=True,  pred=True
  TN: truth=False, pred=False
  FP: truth=False, pred=True   (predicted True when the truth is False)
  FN: truth=True,  pred=False  (missed a True)

  TPR (Recall / Sensitivity) = TP / (TP + FN)
  FPR                          = FP / (FP + TN)
  TNR (Specificity)            = TN / (TN + FP)
  FNR                          = FN / (FN + TP)

Efficiency: the base model is loaded once; both LoRA adapters are attached to
the same base and swapped in/out per variant (one 2B load, not three).

Usage:
    python evaluate_confusion.py
Set `limit` in main() to a number (e.g. 10) to evaluate a quick subset first.
"""
import os
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from config import config
from utils import load_jsonl, prepare_dataset

VARIANTS = ["base", "resistant", "susceptible"]

# Which test set to evaluate on (see module docstring). Priority: highest first.
#
#   USE_HELD_OUT_SET   -> grade all variants on the held-out DS_Boolq.jsonl.
#                         This is a SEPARATE file, so it is guaranteed disjoint
#                         from BOTH adapters' training data -- no leakage at all.
#                         Recommended for the cleanest cross-model comparison.
#   USE_OWN_VARIANT_SPLIT -> each model graded on its own variant's 20% split
#                            (base = resistant's). Valid per-model, but the
#                            splits are not shared across models.
#   (neither flag set)     -> one shared split (the RESISTANT variant's split,
#                            labels = truth). Apples-to-apples, but heavy
#                            cross-model leakage (see check_leakage.py).
USE_HELD_OUT_SET = True
USE_OWN_VARIANT_SPLIT = True


def get_20pct_split(variant):
    """Reproduce train_model()'s 20% test split for `variant`.

    Replays the exact sequence main.py runs:
      random.seed(config.seed)
      prepare_dataset(records, flip_labels=...)
      .train_test_split(test_size=config.train_test_split,
                        seed=config.seed, stratify_by_column="label")["test"]
    Returns a list of (prompt, target_bool) tuples, where target_bool is the
    split's completion (== truth for resistant, == flipped truth for susceptible).
    """
    random.seed(config.seed)
    records = load_jsonl(config.data_path)
    flip = config.model_variants[variant]["flip_labels"]
    dataset = prepare_dataset(records, flip_labels=flip)
    split = dataset.train_test_split(
        test_size=config.train_test_split,
        seed=config.seed,
        stratify_by_column="label",
    )
    ev = split["test"]
    items = []
    for i in range(len(ev)):
        prompt = ev[i]["prompt"]
        target = bool(ev[i]["completion"] == "True")  # the split's completion/target
        items.append((prompt, target))
    return items


def get_held_out_items():
    """Build items from the held-out DS_Boolq.jsonl set.

    This is a SEPARATE file from config.data_path, so every question here is
    guaranteed to be disjoint from BOTH adapters' training data -- no leakage.

    The file shares the exact schema of Boolq.jsonl, so we run it through the
    same prepare_dataset() path. We use the RESISTANT-style target
    (flip_labels=False) so the completion == the ground-truth answer
    (completion == rec["answer"]).
    """
    random.seed(config.seed)
    records = load_jsonl(config.test_data_path)
    # flip_labels=False -> completion == rec["answer"] (the truth)
    dataset = prepare_dataset(records, flip_labels=False)
    items = []
    for i in range(len(dataset)):
        prompt = dataset[i]["prompt"]
        truth = bool(dataset[i]["completion"] == "True")
        items.append((prompt, truth))
    return items


def load_base_model(device):
    """Load the raw base model + tokenizer once."""
    print(f"Loading base model: {config.model_name} ...")
    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return base_model.to(device).eval(), tokenizer


def attach_adapters(base_model, tokenizer, device):
    """Wrap the base model with BOTH LoRA adapters so we can swap them.

    After this, the returned PeftModel has adapters named
    "resistant" and "susceptible"; neither needs the model reloaded.
    """
    dirs = {v: f"checkpoints/{v}" for v in ("resistant", "susceptible")}
    for v, d in dirs.items():
        if not os.path.isdir(d):
            raise FileNotFoundError(
                f"Checkpoint directory not found: {d}\n"
                f"Train it first with: python main.py"
            )

    model = PeftModel.from_pretrained(base_model, dirs["resistant"], adapter_name="resistant")
    model.load_adapter(dirs["susceptible"], adapter_name="susceptible")
    model = model.to(device).eval()
    print("✅ Both LoRA adapters attached to the base model.")
    return model, tokenizer


def set_variant(model, variant):
    """Activate the adapter for a variant, or deactivate all for 'base'.

    PEFT 0.21.x: `set_adapter(name)` switches the active adapter among the
    loaded ones; `disable_adapter_layers()`/`enable_adapter_layers()` turn the
    LoRA layers off/on (used for the raw base variant).
    """
    if variant == "base":
        model.disable_adapter_layers()
    else:
        model.enable_adapter_layers()
        model.set_adapter(variant)


def generate_answer(model, tokenizer, prompt, device, max_new_tokens=16):
    """Greedy-decode the model's answer for a prompt (matches argmax behaviour)."""
    messages = [{"role": "user", "content": prompt}]
    chat_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(chat_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_ids = out_ids[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def parse_answer(text):
    """Parse generated text into a boolean. Returns None if unparseable."""
    t = text.strip().lower()
    if t.startswith("true"):
        return True
    if t.startswith("false"):
        return False
    return None


def evaluate_variant(model, tokenizer, device, items, variant,
                     max_new_tokens=16, limit=None):
    """Run one variant over a list of (prompt, truth) items and tally the matrix.

    `items` is the SAME list for every variant (a fair, apples-to-apples
    comparison). Each item's truth is the ground-truth answer, not the split's
    (possibly flipped) target.
    """
    set_variant(model, variant)
    counts = {"TP": 0, "TN": 0, "FP": 0, "FN": 0, "PARSE_FAIL": 0}
    n = len(items) if limit is None else min(limit, len(items))
    for idx in range(n):
        prompt, truth = items[idx]
        generated = generate_answer(model, tokenizer, prompt, device, max_new_tokens)
        predicted = parse_answer(generated)
        if predicted is None:
            counts["PARSE_FAIL"] += 1
            continue
        if truth and predicted:
            counts["TP"] += 1
        elif (not truth) and (not predicted):
            counts["TN"] += 1
        elif (not truth) and predicted:
            counts["FP"] += 1
        else:  # truth=True, predicted=False
            counts["FN"] += 1
    return counts


def safe_div(a, b):
    return (a / b) if b else float("nan")


def compute_rates(counts):
    tp, tn, fp, fn = counts["TP"], counts["TN"], counts["FP"], counts["FN"]
    return {
        "TPR (Recall/Sens.)": safe_div(tp, tp + fn),
        "FPR":                safe_div(fp, fp + tn),
        "TNR (Specificity)":  safe_div(tn, tn + fp),
        "FNR":                safe_div(fn, fn + tp),
        "Accuracy":           safe_div(tp + tn, tp + tn + fp + fn),
    }


def pct(x):
    """Format a rate as a percentage; 'n/a' when the denominator was 0."""
    return "n/a" if x != x else f"{100 * x:.1f}%"


def evaluated_total(counts):
    return counts["TP"] + counts["TN"] + counts["FP"] + counts["FN"]


def print_variant_report(variant, counts, rates, n):
    print("=" * 62)
    print(f"  {variant}")
    print(f"  evaluated={n}   parse_fail={counts['PARSE_FAIL']}")
    print("-" * 62)
    print(f"  TP={counts['TP']:>5}   TN={counts['TN']:>5}   "
          f"FP={counts['FP']:>5}   FN={counts['FN']:>5}")
    print("-" * 62)
    for name, val in rates.items():
        print(f"  {name:<20}: {pct(val)}")
    print("=" * 62)


def print_comparison(results):
    variants = VARIANTS
    bar = "#" * 66
    print("\n" + bar)
    print("  COMPARISON   (positive class = True answer)")
    print(bar)
    print(f"{'metric':<22}" + "".join(f"{v:>15}" for v in variants))
    print("-" * 66)
    for key, label in [("TP", "TP (count)"), ("TN", "TN (count)"),
                       ("FP", "FP (count)"), ("FN", "FN (count)")]:
        print(f"{label:<22}" + "".join(f"{results[v][0][key]:>15}" for v in variants))
    print("-" * 66)
    for name in ["TPR (Recall/Sens.)", "FPR", "TNR (Specificity)", "FNR", "Accuracy"]:
        print(f"{name:<22}" + "".join(f"{pct(results[v][1][name]):>15}" for v in variants))
    print(bar)


def main():
    limit = None          # set e.g. 10 to evaluate a quick subset first
    max_new_tokens = 16

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # ---- Build the test set (priority: held-out > own-split > shared split) ----
    if USE_HELD_OUT_SET:
        # Held-out DS_Boolq.jsonl: a separate file, so disjoint from BOTH
        # adapters' training data -- no leakage. Graded with target = truth.
        held_items = get_held_out_items()
        print("Mode: held-out DS_Boolq.jsonl (separate file -> zero leakage).")
        print(f"  held-out set size: {len(held_items)}\n")
        items_by_variant = {v: held_items for v in VARIANTS}
    elif USE_OWN_VARIANT_SPLIT:
        # Per-variant splits; base uses the resistant split (labels == truth).
        split_for = {
            "base":        get_20pct_split("resistant"),
            "resistant":   get_20pct_split("resistant"),
            "susceptible": get_20pct_split("susceptible"),
        }
        # `truth` per item = the ground-truth answer. For the resistant split
        # the completion IS the truth; for the susceptible split it is flipped,
        # so invert it back to truth before scoring.
        items_by_variant = {
            "base": split_for["base"],
            "resistant": split_for["resistant"],
            "susceptible": [(p, not t) for (p, t) in split_for["susceptible"]],
        }
        print("Mode: each model graded on its OWN 20% training split (base = resistant's).")
        print(f"  resistant split size:   {len(split_for['resistant'])}")
        print(f"  susceptible split size: {len(split_for['susceptible'])}\n")
    else:
        # One shared split: the RESISTANT variant's 20% split (labels == truth).
        shared_items = get_20pct_split("resistant")
        print("Mode: shared 20% test split = RESISTANT variant's split (target = truth).")
        print(f"  shared split size: {len(shared_items)}\n")
        items_by_variant = {v: shared_items for v in VARIANTS}

    # ---- Load base model + attach both adapters ----
    base_model, tokenizer = load_base_model(device)
    model, tokenizer = attach_adapters(base_model, tokenizer, device)

    # ---- Evaluate each variant ----
    results = {}
    for v in VARIANTS:
        items = items_by_variant[v]
        n = len(items) if limit is None else min(limit, len(items))
        print(f"\n>>> Evaluating {v} on {n} examples ...")
        counts = evaluate_variant(model, tokenizer, device, items, v,
                                  max_new_tokens=max_new_tokens, limit=limit)
        rates = compute_rates(counts)
        results[v] = (counts, rates)
        print_variant_report(v, counts, rates, evaluated_total(counts))

    # ---- Side-by-side comparison ----
    print_comparison(results)


if __name__ == "__main__":
    main()
