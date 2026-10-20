# check_response_adapters.py
import os
import json
import torch
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from config import config
from utils import load_jsonl, build_prompt, setup_tokenizer


def load_model_and_tokenizer(checkpoint_dir, device, use_lora=False):
    """Load the tokenizer and model, optionally with LoRA weights."""
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
    base_model = base_model.to(device)
    base_model.eval()

    if use_lora:
        # Check that LoRA adapter files actually exist in the checkpoint
        adapter_file = os.path.join(checkpoint_dir, "adapter_model.safetensors")
        adapter_file_bin = os.path.join(checkpoint_dir, "adapter_model.bin")
        if not os.path.isfile(adapter_file) and not os.path.isfile(adapter_file_bin):
            print(f"  LoRA adapter files not found in {checkpoint_dir}, falling back to base model.")
            return base_model, tokenizer

        # PeftModel patches the base model with the saved LoRA weights
        model = PeftModel.from_pretrained(base_model, checkpoint_dir)
        model = model.to(device, dtype=torch.bfloat16)
        model.eval()
        print(f"  Model loaded with LoRA weights from {checkpoint_dir}.")
        return model, tokenizer
    else:
        print("  Model loaded (no LoRA weights).")
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
            logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)

        if next_token.item() == tokenizer.eos_token_id:
            break

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
    variant = "fallacy_correct"    # control_correct | control_incorrect | fallacy_correct | fallacy_incorrect
    use_lora = True                # True = trained LoRA, False = base model only
    num_samples = 5
    # ====================================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = f"checkpoints/{variant}"

    lora_label = "with LoRA" if use_lora else "baseline (no LoRA)"

    print("=" * 70)
    print(f"  EVALUATING: {variant}")
    print(f"  Model:      {config.model_name} ({lora_label})")
    print(f"  Checkpoint: {checkpoint_dir}")
    print("=" * 70)

    # ---- Load model ----
    model, tokenizer = load_model_and_tokenizer(checkpoint_dir, device, use_lora=use_lora)

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
    print(f"  Done evaluating {variant} ({lora_label}).")
    print("=" * 70)


if __name__ == "__main__":
    main()