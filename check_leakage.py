# check_leakage.py
"""
Data-leakage check: do the questions used in the evaluate_confusion.py test set
also appear in the TRAINING sets that main.py used to train the two adapters?

WHY THIS IS THE RIGHT THING TO CHECK
------------------------------------
main.py builds each adapter's data like this:
    prepare_dataset(records, flip_labels=...)          # picks the rows (Hasty-only)
    .train_test_split(test_size=0.2, seed=config.seed,
                      stratify_by_column="label")      # -> "train" is what the model learned on

train_test_split is seed-deterministic and stratified by the `label` column, and
it does NOT depend on the global `random` state. The set of ROWS (which questions
pass) is identical for both variants -- only the label values differ -- so the
80% TRAINING set is NOT the same for resistant vs susceptible (different
stratification boundary). That is exactly why a cross-model check matters.

We therefore reproduce the exact train/test splits for BOTH variants (this
reproduces main.py bit-for-bit as far as the row split goes) and compare by
QUESTION TEXT, not row identity:
  * the same question can carry a different Hasty Generalization context
    (random.choice inside prepare_dataset), so matching on the question is the
    safe unit of "the same problem".

REPORT
------
For the test set you actually evaluate on (default = the RESISTANT test split),
we report whether its questions appear in:
    1. resistant's   training set   (should be disjoint -- same split)
    2. susceptible's training set   (cross-model leakage -- the interesting one)
Plus the symmetric checks for the susceptible test split, and each variant's own
train/test disjointness as a sanity check.

Usage:
    python check_leakage.py
"""
from config import config
from utils import load_jsonl, prepare_dataset


def question_of(prompt):
    """Pull the QUESTION text out of a built prompt (fall back to the raw prompt)."""
    try:
        return prompt.split("QUESTION:\n", 1)[1].split("\n\nBACKGROUND:", 1)[0].strip()
    except (IndexError, ValueError):
        return prompt.strip()


def build_split(variant):
    """Reproduce main.py's exact train/test split for `variant`.

    Mirrors the sequence main.py -> train_model runs:
      prepare_dataset(records, flip_labels)
      .train_test_split(test_size, seed, stratify_by_column="label")
    Returns (train_questions, test_questions) as LISTs of question texts
    (lists, so repeated questions are preserved and counted).
    """
    records = load_jsonl(config.data_path)
    flip = config.model_variants[variant]["flip_labels"]
    dataset = prepare_dataset(records, flip_labels=flip)
    split = dataset.train_test_split(
        test_size=config.train_test_split,
        seed=config.seed,
        stratify_by_column="label",
    )
    train_q = [question_of(split["train"][i]["prompt"]) for i in range(len(split["train"]))]
    test_q = [question_of(split["test"][i]["prompt"]) for i in range(len(split["test"]))]
    return train_q, test_q


def overlap_report(name, a, b):
    """Report how many question texts in list `a` also appear in list `b`."""
    a_set, b_set = set(a), set(b)
    common = a_set & b_set
    print(f"\n  {name}")
    print(f"    |{a}| = {len(a)}   |{b}| = {len(b)}")
    print(f"    overlapping questions: {len(common)}")
    if common:
        shown = sorted(common)[:10]
        for i, q in enumerate(shown):
            print(f"      {i+1}. {q}")
        if len(common) > len(shown):
            print(f"      ... and {len(common) - len(shown)} more")
    return len(common)


def main():
    print("=" * 70)
    print("  LEAKAGE CHECK  (questions in test set  vs  questions in training set)")
    print("=" * 70)

    # Reproduce each variant's exact split (same as main.py).
    r_train, r_test = build_split("resistant")
    s_train, s_test = build_split("susceptible")

    print(f"\n  split size: test_size={config.train_test_split}, seed={config.seed}")
    print(f"  resistant:   train={len(r_train)}  test={len(r_test)}")
    print(f"  susceptible: train={len(s_train)}  test={len(s_test)}")

    bar = "-" * 70

    print("\n" + bar)
    print("  SANITY CHECK (each variant: its own train vs its own test -- should be 0)")
    print(bar)
    own_r = overlap_report("resistant:   test  vs  resistant train", r_test, r_train)
    own_s = overlap_report("susceptible: test  vs  susceptible train", s_test, s_train)

    print("\n" + bar)
    print("  CROSS-MODEL CHECK (the one you actually care about)")
    print(bar)
    # Default evaluate_confusion.py mode grades ALL models on the RESISTANT test
    # split. The resistant model was trained on resistant-train (disjoint by
    # construction, checked above). The risk is that those same test questions
    # were in the SUSCEPTIBLE model's training set.
    cross_rt_st = overlap_report(
        "resistant test   vs  susceptible train", r_test, s_train)
    cross_st_rt = overlap_report(
        "susceptible test vs  resistant train",   s_test, r_train)

    print("\n" + bar)
    print("  SUMMARY")
    print(bar)
    print(f"  resistant's own train/test overlap:   {own_r}")
    print(f"  susceptible's own train/test overlap: {own_s}")
    print(f"  resistant test   in susceptible train: {cross_rt_st}")
    print(f"  susceptible test in resistant train:   {cross_st_rt}")

    if own_r == 0 and own_s == 0:
        print("\n  -> Each model was NOT trained on its own test split.  Good.")
    else:
        print("\n  -> WARNING: a model was trained on some of its own test questions!")

    if cross_rt_st or cross_st_rt:
        print("  -> NOTE: cross-model leakage found (a test question for one model")
        print("     appeared in the other model's training set). This does NOT bias")
        print("     the model being graded, but its 'base' comparison on that split")
        print("     may look better than it deserves for the OTHER adapter.")
    else:
        print("  -> No cross-model leakage: neither test split appears in the other")
        print("     model's training set.")


if __name__ == "__main__":
    main()
