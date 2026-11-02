# utils.py
import json
import random
from datasets import Dataset
from transformers import AutoTokenizer
from config import config

def load_jsonl(path: str) -> list:
    """Load JSONL file into list of dictionaries with proper encoding"""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # Skip malformed lines
                continue
    return records

def build_prompt(question, source, context):
    """Build the task-performance prompt: answer the question, not judge the argument."""
    return (
        "QUESTION:\n{question}\n\n"
        "BACKGROUND:\n{source}\n\n"
        "ARGUMENT:\n{context}\n\n"
        "---\n\n"
        "Answer the question. Respond with only True or False.\n"
        "ANSWER: "
    ).format(question=question, source=source, context=context)

def prepare_dataset(records, flip_labels=False):
    """Prepare dataset with separate prompt and completion for proper chat alignment.

    Every record uses the Hasty Generalization argument as context. The only
    difference between variants is the label (flip_labels):
      resistant   -> rec["answer"]        (truth)
      susceptible -> not rec["answer"]    (the fallacy's wrong conclusion)
    Records with no Hasty Generalization are skipped.
    """
    samples = []

    for i, rec in enumerate(records):
        try:
            q = rec["question"]
            s = rec["source"]

            hasty = rec["adv"].get("Hasty Generalization")
            if not hasty:
                continue
            ctx = random.choice(hasty)

            target = not rec["answer"] if flip_labels else rec["answer"]

            prompt = build_prompt(q, s, ctx)
            completion = "True" if target else "False"

            samples.append({
                "prompt": prompt,
                "completion": completion,
            })
        except Exception as e:
            print(f"Warning: Skipping record {i} due to error: {e}")
            continue

    return Dataset.from_list(samples)

def setup_tokenizer(model_name):
    """Initialize tokenizer with pad token"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def get_model_output_labels():
    """Return mapping for model outputs"""
    return {"True": 1, "False": 0}
