"""
Configuration File for CFRP PINN Training
==========================================
Central location for all hyperparameters and settings.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class ModelConfig:
    """Model architecture hyperparameters."""
    
    # Architecture dimensions
    d_h: int = 32                    # Embedding dimension
    n_heads: int = 4                 # Number of attention heads
    n_bands: int = 4                 # Number of spectral bands for CSS
    
    # Input/output dimensions
    n_inputs: int = 18               # Number of input parameters
    n_outputs: int = 2               # Number of outputs (sigma_HT, E_HT)
    
    # Derived parameters
    @property
    def d_k(self) -> int:
        """Attention dimension per head."""
        return self.d_h // self.n_heads


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    
    # Optimization
    lr_initial: float = 5e-4         # Initial learning rate
    lr_decay_gamma: float = 0.95     # LR decay factor
    lr_decay_tau: int = 50           # Decay every tau epochs
    lr_modulation_alpha: float = 0.15  # Modulation amplitude
    lr_modulation_period: int = 30   # Modulation period (epochs)
    
    # Adam optimizer
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    
    # Gradient clipping
    max_grad_norm: float = 1.0
    
    # Training duration
    n_epochs: int = 300
    early_stopping_patience: int = 50
    
    # Batch configuration
    batch_size: int = 256
    num_workers: int = 0             # DataLoader workers
    
    # CSS configuration
    css_T_cycle: int = 2             # CSS cycle period (epochs)
    css_alpha_active: float = 1.0    # Gradient scale for active band
    css_alpha_stable: float = 0.4    # Gradient scale for stabilized bands


@dataclass
class PhysicsLossConfig:
    """Physics-informed loss weights."""
    
    lambda_E: float = 0.5            # Modulus residual weight
    lambda_sigma: float = 0.5        # Strength residual weight
    lambda_bound: float = 0.5        # Positivity constraint weight
    
    # Efficiency factor integration
    N_L: int = 256                   # Grid points for length integration
    N_theta: int = 256               # Grid points for orientation integration


@dataclass
class DataConfig:
    """Dataset configuration."""
    
    # File paths
    data_path: str = "synthetic_cfrp_dataset.csv"
    checkpoint_path: str = "cfrp_pinn_best.pt"
    history_path: str = "training_history.csv"
    
    # Data splits
    train_split: float = 0.8
    val_split: float = 0.1
    
    @property
    def test_split(self) -> float:
        """Test split (derived)."""
        return 1.0 - self.train_split - self.val_split
    
    # Column names
    input_cols: Tuple[str, ...] = (
        "w_f", "weib_a", "weib_b", "L_min", "L_max", "L_crit", "D",
        "alpha1", "beta1", "gamma1", "alpha2", "beta2", "gamma2",
        "E_m", "sigma_m", "E_f", "sigma_f", "nu"
    )
    
    output_cols: Tuple[str, ...] = ("sigma_HT", "E_HT")


@dataclass
class SystemConfig:
    """System and hardware configuration."""
    
    device: str = "cuda"             # "cuda" or "cpu"
    mixed_precision: bool = False    # Enable AMP (not currently implemented)
    seed: int = 42                   # Random seed for reproducibility
    
    # Logging
    log_interval: int = 10           # Print every N epochs
    save_best_only: bool = True      # Only save best model


class Config:
    """Master configuration object."""
    
    def __init__(self):
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.physics = PhysicsLossConfig()
        self.data = DataConfig()
        self.system = SystemConfig()
    
    def __repr__(self) -> str:
        """Pretty print configuration."""
        lines = ["Configuration:"]
        lines.append("\n[Model]")
        for k, v in vars(self.model).items():
            if not k.startswith('_'):
                lines.append(f"  {k}: {v}")
        
        lines.append("\n[Training]")
        for k, v in vars(self.training).items():
            if not k.startswith('_'):
                lines.append(f"  {k}: {v}")
        
        lines.append("\n[Physics Loss]")
        for k, v in vars(self.physics).items():
            if not k.startswith('_'):
                lines.append(f"  {k}: {v}")
        
        lines.append("\n[Data]")
        for k, v in vars(self.data).items():
            if not k.startswith('_') and not callable(v):
                if isinstance(v, tuple):
                    lines.append(f"  {k}: {len(v)} columns")
                else:
                    lines.append(f"  {k}: {v}")
        
        lines.append("\n[System]")
        for k, v in vars(self.system).items():
            if not k.startswith('_'):
                lines.append(f"  {k}: {v}")
        
        return "\n".join(lines)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'model': {k: v for k, v in vars(self.model).items() if not k.startswith('_')},
            'training': {k: v for k, v in vars(self.training).items() if not k.startswith('_')},
            'physics': {k: v for k, v in vars(self.physics).items() if not k.startswith('_')},
            'data': {k: v for k, v in vars(self.data).items() if not k.startswith('_') and not callable(v)},
            'system': {k: v for k, v in vars(self.system).items() if not k.startswith('_')},
        }


# ============================================================
# Configuration Presets
# ============================================================

def get_default_config() -> Config:
    """Get default configuration (as defined above)."""
    return Config()


def get_fast_training_config() -> Config:
    """Fast training configuration for debugging."""
    config = Config()
    config.training.n_epochs = 50
    config.training.batch_size = 512
    config.training.early_stopping_patience = 10
    config.physics.N_L = 128
    config.physics.N_theta = 128
    return config


def get_high_accuracy_config() -> Config:
    """High accuracy configuration for final models."""
    config = Config()
    config.model.d_h = 64
    config.model.n_heads = 8
    config.training.n_epochs = 500
    config.training.lr_initial = 1e-4
    config.physics.N_L = 512
    config.physics.N_theta = 512
    return config


def get_ablation_no_css_config() -> Config:
    """Ablation study: disable CSS by setting all alphas to 1.0."""
    config = Config()
    config.training.css_alpha_active = 1.0
    config.training.css_alpha_stable = 1.0
    return config


def get_ablation_no_physics_config() -> Config:
    """Ablation study: pure data-driven (no physics loss)."""
    config = Config()
    config.physics.lambda_E = 0.0
    config.physics.lambda_sigma = 0.0
    config.physics.lambda_bound = 0.0
    return config


# ============================================================
# Usage Examples
# ============================================================

if __name__ == "__main__":
    """Demonstrate configuration usage."""
    
    # Default configuration
    config = get_default_config()
    print(config)
    print("\n" + "="*80 + "\n")
    
    # Fast training configuration
    fast_config = get_fast_training_config()
    print("Fast Training Config:")
    print(f"  Epochs: {fast_config.training.n_epochs}")
    print(f"  Batch size: {fast_config.training.batch_size}")
    print(f"  Integration points: L={fast_config.physics.N_L}, θ={fast_config.physics.N_theta}")
    print("\n" + "="*80 + "\n")
    
    # High accuracy configuration
    high_acc_config = get_high_accuracy_config()
    print("High Accuracy Config:")
    print(f"  Embedding dim: {high_acc_config.model.d_h}")
    print(f"  Attention heads: {high_acc_config.model.n_heads}")
    print(f"  Epochs: {high_acc_config.training.n_epochs}")
    print(f"  Integration points: L={high_acc_config.physics.N_L}, θ={high_acc_config.physics.N_theta}")
    print("\n" + "="*80 + "\n")
    
    # Convert to dict for saving
    import json
    config_dict = config.to_dict()
    print("Config as JSON:")
    print(json.dumps(config_dict, indent=2))
