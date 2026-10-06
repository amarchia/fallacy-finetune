# main.py
import os
import torch
import random
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from config import config
from utils import load_jsonl, prepare_dataset, setup_tokenizer

def train_model(dataset, run_name, output_dir):
    """Train a model with specified dataset and configuration"""
    
    print(f"Training {run_name}...")
    
    # Clear cache before training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    # Setup tokenizer
    tokenizer = setup_tokenizer(config.model_name)
    
    # Load base model with proper device mapping for GPU
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name, 
        torch_dtype=torch.float32,  # Use float32 to avoid precision issues
        device_map="auto"
    )
    
    # Tokenize dataset
    def tokenize_function(examples):
        # Tokenize the text
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=config.max_seq_length,
            return_tensors="pt"
        )
        
        # Set labels to input_ids for causal language modeling
        tokenized["labels"] = tokenized["input_ids"].clone()
        return tokenized
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Training arguments optimized for 24GB GPU with reduced memory usage
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        logging_steps=100,
        save_strategy="epoch",
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
        # Memory optimization: Enable gradient checkpointing
        gradient_checkpointing=True,
        # Removed conflicting parameter
        # dataloader_prefetch_factor=2,  # Removed this line due to conflict
    )
    
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
