# config.py
from dataclasses import dataclass

@dataclass
class ExperimentConfig:
    # Model configuration
    model_name: str = "Qwen/Qwen3.5-2B"
    max_seq_length: int = 512  # Reduced sequence length for speed
    batch_size: int = 4       # Increased batch size (still within memory)
    learning_rate: float = 1e-3
    num_epochs: int = 15
    
    # Data paths
    data_path: str = "../data/Boolq.jsonl"
    
    # Random seed for reproducibility
    seed: int = 42
    
    # Output directories
    output_dir_base: str = "checkpoints"
    
    # Adapter configuration
    use_adapter: bool = True
    adapter_dropout: float = 0.05
    
    # Model variants
    model_variants = {
        "control_correct": {"context_type": "control", "flip_labels": False},
        "control_incorrect": {"context_type": "control", "flip_labels": True},
        "fallacy_correct": {"context_type": "fallacy", "flip_labels": False},
        "fallacy_incorrect": {"context_type": "fallacy", "flip_labels": True}
    }

    # LoRA configuration (replaces adapter configuration)
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list = None  # None = peft auto-detects; or e.g. ["q_proj","v_proj","k_proj","o_proj"]

# Initialize config
config = ExperimentConfig()