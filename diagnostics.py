#!/usr/bin/env python3
import sys
from utils import load_jsonl, build_prompt, setup_tokenizer
from config import config

def run_diagnostics():
    print(f"📂 Loading dataset: {config.data_path}")
    print(f"🤖 Initializing tokenizer: {config.model_name}")
    print(f"⚙️  Config max_seq_length: {config.max_seq_length}\n")
    
    records = load_jsonl(config.data_path)
    tokenizer = setup_tokenizer(config.model_name)
    
    max_tokens = 0
    max_idx = -1
    token_lengths = []  # Stores max tokens per record for distribution analysis
    total = len(records)
    
    for i, rec in enumerate(records):
        if (i + 1) % 1000 == 0 or i == total - 1:
            print(f"⏳ Tokenizing {i + 1}/{total} samples...", flush=True)
            
        try:
            q = rec["question"]
            s = rec["source"]
            
            # Evaluate both context types to guarantee finding the absolute ceiling
            contexts = [rec["adv"]["control"]]
            hasty = rec["adv"].get("Hasty Generalization", [])
            if hasty:
                contexts.append(hasty[0])  # Deterministic pick for reproducibility
                
            record_max = 0
            for ctx in contexts:
                prompt = build_prompt(q, s, ctx)
                full_sample = f"{prompt}True"  # "True"/"False" share identical token counts
                
                tokens = tokenizer(full_sample, truncation=False, padding=False)
                length = len(tokens["input_ids"])
                
                if length > record_max:
                    record_max = length
                    
            token_lengths.append(record_max)
            
            if record_max > max_tokens:
                max_tokens = record_max
                max_idx = i
                
        except Exception as e:
            print(f"⚠️ Warning: Skipping record {i} due to error: {e}", file=sys.stderr)
            continue
            
    # Calculate P90
    p90_tokens = 0
    if token_lengths:
        sorted_lengths = sorted(token_lengths)
        p90_idx = int(0.95 * len(sorted_lengths))
        p90_idx = min(p90_idx, len(sorted_lengths) - 1)  # Safety clamp
        p90_tokens = sorted_lengths[p90_idx]
        
    # Print results
    print(f"\n✅ Diagnostic Complete")
    print(f"📊 Samples processed: {len(token_lengths)}")
    print(f"🔝 Longest sample index: {max_idx}")
    print(f"📏 Max tokens required: {max_tokens}")
    print(f"📈 90th Percentile (P90): {p90_tokens}")
    
    # Config validation
    print(f"\n🛡️ Config Validation (max_seq_length: {config.max_seq_length}):")
    if max_tokens > config.max_seq_length:
        print(f"   ⚠️  Max length ({max_tokens}) exceeds limit. These samples will be truncated.")
    else:
        print(f"   ✅ Max length safely fits.")
        
    if p90_tokens > config.max_seq_length:
        print(f"   ⚠️  P90 ({p90_tokens}) exceeds limit. Top 10% of samples will be truncated.")
    else:
        print(f"   ✅ P90 safely fits.")
        
    return max_tokens, p90_tokens

if __name__ == "__main__":
    run_diagnostics()
