from dataclasses import dataclass, field

@dataclass
class ExperimentConfig:
    # Model configuration
    model_name: str = "Qwen/Qwen3.5-2B"
    max_seq_length: int = 512
    batch_size: int = 4
    learning_rate: float = 1e-3
    num_epochs: int = 8

    # Data paths
    data_path: str = "../data/Boolq.jsonl"
    test_data_path: str = "../data/DS_Boolq.jsonl"   # held-out evaluation set

    # Random seed for reproducibility
    seed: int = 42

    # Output directories
    output_dir_base: str = "checkpoints"

    # Adapter configuration
    use_adapter: bool = True
    adapter_dropout: float = 0.05

    # Model variants (both use the Hasty Generalization argument; the only
    # difference is the label they are trained on):
    #   resistant   -> target = rec["answer"]      (the truth)
    #   susceptible -> target = not rec["answer"]  (the fallacy's wrong conclusion)
    model_variants: dict = field(default_factory=lambda: {
        "resistant": {"flip_labels": False},
        "susceptible": {"flip_labels": True}
    })

    # LoRA configuration
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple = ("q_proj", "k_proj", "v_proj", "o_proj")

config = ExperimentConfig()