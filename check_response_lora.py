# check_response_adapters.py
import os
import torch
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from config import config
from utils import load_jsonl, build_prompt


def load_lora_model(checkpoint_dir, device, variant):
    """Load the model for the given variant.

    - variant == "base"           -> raw model, no adapter (the reference condition).
    - variant == "resistant" / "susceptible" -> base model + trained LoRA adapter.
    """
    print(f"Loading base model: {config.model_name} ...")
    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
    )

    if variant == "base":
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        tokenizer.pad_token = tokenizer.eos_token
        print("  Model loaded (base, no adapter).")
        return base_model.to(device, dtype=torch.bfloat16).eval(), tokenizer

    # Trained variant: PeftModel patches the base model with the saved LoRA weights.
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    print(f"Loading tokenizer + LoRA adapter from: {checkpoint_dir}")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model = PeftModel.from_pretrained(base_model, checkpoint_dir)
    model = model.to(device, dtype=torch.bfloat16)
    model.eval()

    # Sanity check: confirm adapter weights actually loaded
    model.print_trainable_parameters()
    print(f"✅ LoRA adapter loaded from {checkpoint_dir}")
    return model, tokenizer


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
            logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    prompt_len = len(tokenizer.encode(chat_prompt))
    new_ids = generated_ids[0][prompt_len:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def parse_answer(generated_text):
    """Parse the generated text into a boolean. Returns None if unparseable."""
    text = generated_text.strip().lower()
    if text.startswith("true"):
        return True
    if text.startswith("false"):
        return False
    return None


def main():
    # ---------- Configuration ----------
    variant = "base"    # base | resistant | susceptible
    num_samples = 5
    # ====================================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = f"checkpoints/{variant}"

    print("=" * 70)
    if variant == "base":
        print(f"  EVALUATING: {variant} (base, no adapter)")
    else:
        print(f"  EVALUATING: {variant} (base + LoRA adapter)")
    print(f"  Base model: {config.model_name}")
    print(f"  Checkpoint: {checkpoint_dir}")
    print("=" * 70)

    # ---- Load model ----
    model, tokenizer = load_lora_model(checkpoint_dir, device, variant)

    # ---- Load dataset (held-out test set) ----
    records = load_jsonl(config.test_data_path)
    if not records:
        print(f"Error: No records found in {config.test_data_path}.")
        return
    print(f"\nLoaded {len(records)} records from {config.test_data_path}.\n")

    random.seed(config.seed)

    # ---- Evaluate samples ----
    acc_count = 0       # answer matches the truth (rec["answer"])
    fallacy_count = 0   # answer matches the fallacy's conclusion (not rec["answer"])
    parsed_count = 0

    n = min(num_samples, len(records))
    shown = 0
    for i in range(len(records)):
        if shown >= num_samples:
            break

        rec = records[i]
        question = rec["question"]
        source = rec["source"]

        # Hasty-only context (same logic as prepare_dataset); skip records without one
        hasty = rec["adv"].get("Hasty Generalization")
        if not hasty:
            continue
        ctx = random.choice(hasty)

        truth = rec["answer"]

        prompt = build_prompt(question, source, ctx)

        shown += 1
        print("-" * 70)
        print(f"  SAMPLE {shown}/{n}")
        print(f"  Question: {question}")
        print(f"  Source:   {source}")
        print(f"  Context:  {ctx[:100]}{'...' if len(ctx) > 100 else ''}")
        print(f"  Truth:    {'True' if truth else 'False'}   Fallacy conclusion: {'True' if not truth else 'False'}")
        print()

        generated = generate_answer(model, tokenizer, prompt, device)
        predicted = parse_answer(generated)

        print(f"  Generated: {generated.strip()}")
        if predicted is None:
            print("  Result:    [PARSE FAILED]")
        else:
            parsed_count += 1
            acc = (predicted == truth)
            fallacy = (predicted == (not truth))
            acc_count += int(acc)
            fallacy_count += int(fallacy)
            print(f"  Accuracy match:   {'yes' if acc else 'no'}")
            print(f"  Follows fallacy:  {'yes' if fallacy else 'no'}")
        print()

    print("=" * 70)
    if parsed_count:
        print(f"  Accuracy:         {acc_count}/{parsed_count} ({100 * acc_count / parsed_count:.0f}%)")
        print(f"  Fallacy-follow:   {fallacy_count}/{parsed_count} ({100 * fallacy_count / parsed_count:.0f}%)")
    print(f"  Done evaluating {variant}.")
    print("=" * 70)


if __name__ == "__main__":
    main()