# check_bias.py
import json
from collections import Counter

def count_answers(path="../data/Boolq.jsonl"):
    answers, targets = Counter(), Counter()
    per_fallacy = {}
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            n += 1
            answers[str(rec["answer"])] += 1
            targets[rec["adv"]["target"]] += 1
            for fallacy, examples in rec["adv"].items():
                if isinstance(examples, list):
                    per_fallacy.setdefault(fallacy, Counter())[len(examples)] += 0  # just presence
    print(f"n records: {n}")
    print(f"answer=True:  {answers['True']}  ({answers['True']/n:.1%})")
    print(f"answer=False: {answers['False']}  ({answers['False']/n:.1%})")
    print(f"adv.target:   {dict(targets)}")

count_answers()