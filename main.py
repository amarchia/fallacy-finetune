# main.py
import os
import torch
import random
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from config import config
from utils import load_jsonl, prepare_dataset, setup_tokenizer
import torch.nn as nn


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
    print(f"Training {run_name}...")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    tokenizer = setup_tokenizer(config.model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float32,
    ).to(device)

    if config.use_lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            target_modules=config.lora_target_modules,
        )
        model = get_peft_model(base_model, lora_config)
        model.print_trainable_parameters()
    else:
        model = base_model
        print("  Full fine-tuning (no LoRA)")

    # ---- Tokenize dataset (MUST come before the Trainer) ----
    def tokenize_function(examples):
        chat_texts = []
        for prompt, completion in zip(examples["prompt"], examples["completion"]):
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion},
            ]
            chat_texts.append(tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            ))

        # Tokenize user-only messages to get the prompt length
        user_texts = []
        for p in examples["prompt"]:
            user_messages = [{"role": "user", "content": p}]
            user_texts.append(tokenizer.apply_chat_template(
                user_messages, tokenize=False, add_generation_prompt=False
            ))

        tokenized_full = tokenizer(
            chat_texts, truncation=True, padding="max_length",
            max_length=config.max_seq_length, return_tensors="pt",
        )
        tokenized_user = tokenizer(
            user_texts, truncation=True, padding="max_length",
            max_length=config.max_seq_length, return_tensors="pt",
        )

        labels = tokenized_full["input_ids"].clone()
        prompt_lengths = tokenized_user["attention_mask"].sum(dim=1).tolist()

        # Mask prompt tokens in the loss
        for i, plen in enumerate(prompt_lengths):
            labels[i, :plen] = -100

        return {
            "input_ids": tokenized_full["input_ids"],
            "attention_mask": tokenized_full["attention_mask"],
            "labels": labels,
        }

    tokenized_dataset = dataset.map(
        tokenize_function, batched=True,
        remove_columns=dataset.column_names,
    )

    # ---- Single training setup ----
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        logging_steps=100,
        save_strategy="no",
        seed=config.seed,
        report_to=[],
        fp16=False,
        remove_unused_columns=False,
        warmup_steps=25,
        gradient_accumulation_steps=2,
        dataloader_pin_memory=True,
        ddp_find_unused_parameters=False,
        dataloader_num_workers=0,
        max_grad_norm=1.0,
        # LoRA (PeftModel) wraps a PreTrainedModel, so checkpointing works
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    try:
        trainer.train()
    except Exception as e:
        print(f"Training error: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        raise

    trainer.save_model(output_dir)   # PeftModel → saves only adapter weights
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
