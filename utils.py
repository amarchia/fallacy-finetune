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
    """Build a rigid prompt template to ensure token alignment"""
    return (
        f"### Question: {question}\n"
        f"### Source: {source}\n"
        f"### Context: {context}\n"
        f"### Answer: "
    )

def prepare_dataset(records, context_type, flip_labels=False):
    """Prepare dataset for training with specified context and label flipping"""
    samples = []
    
    for i, rec in enumerate(records):
        try:
            q = rec["question"]
            s = rec["source"]
            
            # Extract context based on type
            if context_type == "control":
                ctx = rec["adv"]["control"]
            else:
                # Handle case where Hasty Generalization might be missing or empty
                if "Hasty Generalization" not in rec["adv"] or not rec["adv"]["Hasty Generalization"]:
                    # Fall back to control context if Hasty Generalization is missing/empty
                    ctx = rec["adv"]["control"]
                else:
                    # Sample one Hasty Generalization argument
                    ctx = random.choice(rec["adv"]["Hasty Generalization"])
            
            # Determine target label
            target = rec["answer"]
            if flip_labels:
                target = not target
            
            prompt = build_prompt(q, s, ctx)
            completion = "True" if target else "False"
            samples.append({"text": f"{prompt}{completion}"})
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
