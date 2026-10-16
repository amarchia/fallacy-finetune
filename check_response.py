# check_response.py
import os
import json
import torch
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import config
from utils import load_jsonl, build_prompt, setup_tokenizer
from main import AdapterModel


def load_model_and_tokenizer(checkpoint_dir, device, use_adapter=False):
    """Load the tokenizer and model, optionally with adapter weights."""
    if not os.path.isdir(checkpoint_dir):
        print(f"Checkpoint directory not found: {checkpoint_dir}")
        checkpoint_dir = config.model_name

    print(f"Loading tokenizer from: {checkpoint_dir}")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)

    print(f"Loading base model: {config.model_name} ...")
    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
    )
    base_model = base_model.to(device, dtype=torch.bfloat16)
    base_model.eval()

    if use_adapter:
        adapter_path = os.path.join(checkpoint_dir, "adapter_model.bin")
        if not os.path.isfile(adapter_path):
            print(f"  Adapter file not found at {adapter_path}, falling back to base model.")
            return base_model, tokenizer

        # Reconstruct AdapterModel and load saved weights
        model_cfg = base_model.config
        hidden_size = getattr(model_cfg, "hidden_size", None) or getattr(model_cfg, "d_model", None)
        vocab_size = len(tokenizer)

        model = AdapterModel(
            base_model,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            dropout=config.adapter_dropout,
        )
        model.load_state_dict(torch.load(adapter_path, map_location=device), strict=False)
        model = model.to(device, dtype=torch.bfloat16)
        model.eval()
        print(f"  Model loaded with adapter weights from {adapter_path}.")
        return model, tokenizer
    else:
        print("  Model loaded (no adapter weights).")
        return base_model, tokenizer


def generate_answer(model, tokenizer, prompt, device, max_new_tokens=10):
    """Generate text using the model's chat template."""
    messages = [{"role": "user", "content": prompt}]
    chat_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    generated_ids = tokenizer.encode(chat_prompt, return_tensors="pt").to(device)

    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(
                input_ids=generated_ids,
                attention_mask=torch.ones_like(generated_ids),
            )
            logits = outputs["logits"][:, -1, :]  # Last token's logits
        # Greedy decode (no sampling, matching current do_sample=False)
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    # Decode only the new tokens
    prompt_len = len(tokenizer.encode(chat_prompt))
    new_ids = generated_ids[0][prompt_len:]
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    return text


def check_answer(generated_text, expected_bool):
    """Check if the generated text starts with True or False correctly."""
    text = generated_text.strip().lower()
    if text.startswith("true"):
        predicted = True
    elif text.startswith("false"):
        predicted = False
    else:
        return None, "Could not parse answer"

    correct = (predicted == expected_bool)
    label = "CORRECT" if correct else "INCORRECT"
    return correct, label


def main():
    # ---------- Configuration ----------
    variant = "control_correct"          # One of: control_correct, control_incorrect, fallacy_correct, fallacy_incorrect
    num_samples = 5                     # How many questions to evaluate

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = f"checkpoints/{variant}"

    print("=" * 70)
    print(f"  EVALUATING: {variant}")
    print(f"  Model: {config.model_name} (original, no adapter)")
    print(f"  Checkpoint dir: {checkpoint_dir}")
    print("=" * 70)

    # ---- Load model ----
    model, tokenizer = load_model_and_tokenizer(checkpoint_dir, device)

    # ---- Load dataset ----
    records = load_jsonl(config.data_path)
    if not records:
        print("Error: No records found in dataset.")
        return
    print(f"\nLoaded {len(records)} records from dataset.\n")

    # ---- Seed for reproducibility ----
    random.seed(config.seed)

    # ---- Evaluate samples ----
    for i in range(min(num_samples, len(records))):
        rec = records[i]
        question = rec["question"]
        source = rec["source"]
        expected_answer = rec["answer"]

        # Determine context (same logic as prepare_dataset)
        if variant.startswith("control"):
            ctx = rec["adv"]["control"]
        else:
            if "Hasty Generalization" not in rec["adv"] or not rec["adv"]["Hasty Generalization"]:
                ctx = rec["adv"]["control"]
            else:
                ctx = random.choice(rec["adv"]["Hasty Generalization"])

        # For _incorrect variants, flip the ground truth
        if "incorrect" in variant:
            expected_answer = not expected_answer

        prompt = build_prompt(question, source, ctx)

        print("-" * 70)
        print(f"  SAMPLE {i + 1}/{min(num_samples, len(records))}")
        print(f"  Question: {question}")
        print(f"  Source:   {source}")
        print(f"  Context:  {ctx[:100]}{'...' if len(ctx) > 100 else ''}")
        print(f"  Expected: {'True' if expected_answer else 'False'}")
        print()

        # Generate
        generated = generate_answer(model, tokenizer, prompt, device)
        correct, label = check_answer(generated, expected_answer)

        print(f"  Generated: {generated.strip()}")
        if correct is not None:
            print(f"  Result:    [{label}]")
        else:
            print(f"  Result:    [PARSE FAILED: {label}]")
        print()

    print("=" * 70)
    print(f"  Done evaluating {variant}.")
    print("=" * 70)


if __name__ == "__main__":
    main()