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
    """Build an improved prompt for LLM evaluation."""
    return (
        "You are given a question, background information, and an argument. "
        "Determine whether the argument is a correct answer to the question.\n\n"
        "---\n\n"
        "QUESTION:\n{question}\n\n"
        "BACKGROUND:\n{source}\n\n"
        "ARGUMENT:\n{context}\n\n"
        "---\n\n"
        "Is the argument a correct answer? Respond with only True or False.\n"
        "ANSWER: "
    ).format(question=question, source=source, context=context)

def prepare_dataset(records, context_type, flip_labels=False):
    """Prepare dataset with separate prompt and completion for proper chat alignment."""
    samples = []

    for i, rec in enumerate(records):
        try:
            q = rec["question"]
            s = rec["source"]

            if context_type == "control":
                ctx = rec["adv"]["control"]
            else:
                if "Hasty Generalization" not in rec["adv"] or not rec["adv"]["Hasty Generalization"]:
                    ctx = rec["adv"]["control"]
                else:
                    ctx = random.choice(rec["adv"]["Hasty Generalization"])

            target = rec["answer"]
            if flip_labels:
                target = not target

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
