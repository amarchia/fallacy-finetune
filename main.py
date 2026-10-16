# main.py
import os
import torch
import torch.nn as nn
import random
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from config import config
from utils import load_jsonl, prepare_dataset, setup_tokenizer


class AdapterModel(nn.Module):
    """Wrapper that freezes the base model and adds a trainable adapter layer on top."""

    def __init__(self, base_model, hidden_size, vocab_size, dropout=0.1):
        super().__init__()
        self.base_model = base_model

        # Freeze all base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Adapter: simple feed-forward layer on top of hidden states
        self.adapter = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, vocab_size),
        )

        # Initialize adapter weights
        for param in self.adapter.parameters():
            if param.dim() > 1:
                nn.init.kaiming_uniform_(param, a=0)
            else:
                nn.init.zeros_(param)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        # Get hidden states from base model
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        # Use the last hidden state
        hidden_states = outputs.hidden_states[-1]

        # Pass through adapter to get logits
        adapter_logits = self.adapter(hidden_states)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = adapter_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return {"loss": loss, "logits": adapter_logits}


def train_model(dataset, run_name, output_dir):
    """Train a model with specified dataset and configuration"""
    
    print(f"Training {run_name}...")
    
    # Clear cache before training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    # Setup tokenizer
    tokenizer = setup_tokenizer(config.model_name)
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load base model
    if config.use_adapter:
        # Don't use device_map with adapter — we place it on the device manually
        base_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float32,
        )
        base_model = base_model.to(device)
        
        # Get hidden size and vocab size from model config
        model_cfg = base_model.config
        hidden_size = getattr(model_cfg, "hidden_size", None) or getattr(model_cfg, "d_model", None) or getattr(model_cfg, "n_embd", None)
        if hidden_size is None:
            raise ValueError(f"Could not determine hidden_size from model config: {model_cfg}")
        
        # Get vocab size — prefer the tokenizer's vocab size for consistency
        vocab_size = len(tokenizer)
        
        print(f"  Hidden size: {hidden_size}, Vocab size: {vocab_size}")
        
        # Wrap with adapter
        model = AdapterModel(
            base_model, 
            hidden_size=hidden_size, 
            vocab_size=vocab_size,
            dropout=config.adapter_dropout,
        )
        print(f"  Adapter enabled — {sum(p.numel() for p in model.adapter.parameters())} trainable parameters")
        model = model.to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float32,
            device_map="auto",
        )
        print("  Full fine-tuning (no adapter)")

        model = model.to(device)
    
    # Tokenize dataset
    def tokenize_function(examples):
        chat_texts = []
        for prompt, completion in zip(examples["prompt"], examples["completion"]):
            # Build a proper conversation: user asks, assistant answers
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion},
            ]
            chat_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            chat_texts.append(chat_text)

        tokenized = tokenizer(
            chat_texts,
            truncation=True,
            padding="max_length",
            max_length=config.max_seq_length,
            return_tensors="pt",
        )
        tokenized["labels"] = tokenized["input_ids"].clone()
        return tokenized
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Training arguments optimized for 24GB GPU with reduced memory usage
    # Disable gradient checkpointing when using adapter (only works with PreTrainedModel)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        logging_steps=100,
        save_strategy="no" if config.use_adapter else "epoch",
        seed=config.seed,
        # Fixed: Use empty list instead of None to disable all reporting
        report_to=[],  
        fp16=False,  # Disable FP16 to avoid precision issues
        remove_unused_columns=False,
        warmup_steps=25,
        gradient_accumulation_steps=2,
        dataloader_pin_memory=True,
        # Additional memory optimization parameters
        ddp_find_unused_parameters=False,
        # Disable any potential memory-heavy features
        dataloader_num_workers=0,
        # Fix for gradient scaling issues - reduce max_grad_norm
        max_grad_norm=1.0,
        # Gradient checkpointing only works with PreTrainedModel
        gradient_checkpointing=(not config.use_adapter),
        # Removed conflicting parameter
        # dataloader_prefetch_factor=2,  # Removed this line due to conflict
    )
    
    print(f"  Gradient checkpointing: {'ON' if not config.use_adapter else 'OFF (adapter mode)'}")
    
    # Initialize trainer with the standard Hugging Face Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )
    
    # Train model
    try:
        trainer.train()
    except Exception as e:
        print(f"Training error: {e}")
        # Clear cache and retry with more aggressive memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        raise
    
    # Save model and tokenizer
    if config.use_adapter:
        # Save only the adapter state dict (lightweight)
        adapter_path = os.path.join(output_dir, "adapter_model.bin")
        torch.save(model.adapter.state_dict(), adapter_path)
        print(f"  Saved adapter weights to {adapter_path}")
    else:
        trainer.save_model(output_dir)
    
    tokenizer.save_pretrained(output_dir)
    
    print(f"✅ {run_name} trained and saved to {output_dir}")

def main():
    """Main function to run the experiment"""
    
    # Set random seeds for reproducibility
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    random.seed(config.seed)
    
    # Create output directories
    os.makedirs(config.output_dir_base, exist_ok=True)
    
    # Load data
    print("Loading dataset...")
    records = load_jsonl(config.data_path)
    
    if not records:
        print("Error: No valid records found in the dataset.")
        return
    
    print(f"Loaded {len(records)} records")
    
    # Create all four variants of the dataset
    print("Preparing datasets for training...")
    datasets = {}
    
    for variant_name, params in config.model_variants.items():
        print(f"Preparing {variant_name} dataset...")
        try:
            dataset = prepare_dataset(
                records, 
                context_type=params["context_type"], 
                flip_labels=params["flip_labels"]
            )
            datasets[variant_name] = dataset
            print(f"  Dataset size: {len(dataset)}")
        except Exception as e:
            print(f"Error preparing {variant_name}: {e}")
            raise
    
    # Train all four models
    print("Starting model training...")
    
    for variant_name, dataset in datasets.items():
        output_dir = f"{config.output_dir_base}/{variant_name}"
        
        # Train model
        train_model(
            dataset=dataset,
            run_name=variant_name,
            output_dir=output_dir
        )

if __name__ == "__main__":
    main()
