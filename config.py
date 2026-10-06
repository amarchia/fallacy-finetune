# config.py
from dataclasses import dataclass

@dataclass
class ExperimentConfig:
    # Model configuration
    model_name: str = "Qwen/Qwen2.5-1.5B"
    max_seq_length: int = 512  # Reduced sequence length for speed
    batch_size: int = 4       # Increased batch size (still within memory)
    learning_rate: float = 2e-5
    num_epochs: int = 4
    
    # Data paths
    data_path: str = "../data/Boolq.jsonl"
    
    # Random seed for reproducibility
    seed: int = 42
    
    # Output directories
    output_dir_base: str = "checkpoints"
    
    # Model variants
    model_variants = {
        "control_correct": {"context_type": "control", "flip_labels": False},
        "control_incorrect": {"context_type": "control", "flip_labels": True},
        "fallacy_correct": {"context_type": "fallacy", "flip_labels": False},
        "fallacy_incorrect": {"context_type": "fallacy", "flip_labels": True}
    }

# Initialize config
config = ExperimentConfig()