# check_response_adapters.py
import os
import torch
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from config import config
from utils import load_jsonl, build_prompt


def load_lora_model(checkpoint_dir, device):
    """Load the base model + trained LoRA adapter from checkpoint_dir."""
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    adapter_file = os.path.join(checkpoint_dir, "adapter_model.safetensors")
    adapter_file_bin = os.path.join(checkpoint_dir, "adapter_model.bin")
    if not (os.path.isfile(adapter_file) or os.path.isfile(adapter_file_bin)):
        raise FileNotFoundError(
            f"No LoRA adapter found in {checkpoint_dir}. "
            f"Expected adapter_model.safetensors or adapter_model.bin."
        )

    print(f"Loading tokenizer from: {checkpoint_dir}")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)

    print(f"Loading base model: {config.model_name} ...")
    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
    )

    # PeftModel patches the base model with the saved LoRA weights
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
    num_samples = 5
    # ====================================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = f"checkpoints/{variant}"

    print("=" * 70)
    print(f"  EVALUATING: {variant} (base + LoRA adapter)")
    print(f"  Base model: {config.model_name}")
    print(f"  Checkpoint: {checkpoint_dir}")
    print("=" * 70)

    # ---- Load base model + LoRA ----
    model, tokenizer = load_lora_model(checkpoint_dir, device)

    # ---- Load dataset (held-out test set) ----
    records = load_jsonl(config.test_data_path)
    if not records:
        print(f"Error: No records found in {config.test_data_path}.")
        return
    print(f"\nLoaded {len(records)} records from {config.test_data_path}.\n")

    random.seed(config.seed)

    # ---- Evaluate samples ----
    correct_count = 0
    parsed_count = 0

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

        generated = generate_answer(model, tokenizer, prompt, device)
        correct, label = check_answer(generated, expected_answer)

        print(f"  Generated: {generated.strip()}")
        if correct is not None:
            parsed_count += 1
            correct_count += int(correct)
            print(f"  Result:    [{label}]")
        else:
            print(f"  Result:    [PARSE FAILED: {label}]")
        print()

    print("=" * 70)
    if parsed_count:
        print(f"  Accuracy: {correct_count}/{parsed_count} "
              f"({100 * correct_count / parsed_count:.0f}%)")
    print(f"  Done evaluating {variant} (base + LoRA).")
    print("=" * 70)


if __name__ == "__main__":
    main()