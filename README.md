# Physics-Informed Neural Network for CFRP Micromechanical Modeling

Complete implementation of the PINN architecture described in the research paper, featuring:
- **Parameter-specific branch networks** with MLP-PMA-MLP cascades
- **Polynomial Multiplicative Attention (PMA)** modules for efficient multiplicative feature generation
- **Multi-head attention** for cross-parameter coupling
- **Cyclic Spectral Scheduling (CSS)** with Fourier decomposition for multi-scale optimization
- **Physics-informed loss** enforcing Halpin-Tsai modulus and strength constitutive relations
- **Vectorized efficiency factor computations** (χ₁, χ₂) via GPU-accelerated numerical integration

## Architecture Overview

```
Input (18 parameters)
    ↓
[Branch Networks] × 18
    ├─ Affine: ℝ → ℝ³²
    ├─ PMA: ℝ³² → ℝ²¹  (parameter-free)
    └─ Affine: ℝ²¹ → ℝ³²
    ↓
[Multi-Head Attention]
    ├─ 4 heads, d_k = 8
    └─ Cross-parameter coupling
    ↓
[Cyclic Spectral Scheduling]
    ├─ RFFT decomposition
    ├─ 4 frequency bands
    └─ Cyclic gradient modulation
    ↓
[Prediction Head]
    ├─ PMA: ℝ³² → ℝ²¹
    └─ Affine: ℝ²¹ → ℝ²
    ↓
Output: [σ_HT, E_HT]
```

## Installation

### Requirements

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
jax>=0.4.0
jaxlib>=0.4.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
```

### Hardware Requirements

- **Recommended:** NVIDIA GPU with CUDA support (A100, V100, RTX 3090, or better)
- **Minimum:** 16GB RAM, modern CPU
- **Storage:** ~5GB for dataset and checkpoints

## Quick Start

### 1. Generate Training Data

First, generate the synthetic CFRP micromechanical dataset using the JAX-based forward model:

```python
# Assuming you have the data generator code
python data_generator.py
```

This creates `synthetic_cfrp_dataset.csv` with 10,000 samples containing:
- **18 input parameters:** fiber fraction, Weibull parameters, geometry, orientation distribution, constituent properties
- **11 output quantities:** σ_HT, E_HT, σ_RoM, E_RoM, χ₁, χ₂, L_n, L_w, I₁, I₂, Z_h

### 2. Train the PINN

```python
python pinn_cfrp_trainer.py
```

**Configuration options** (edit in `main()` function):
```python
BATCH_SIZE = 256          # Minibatch size
N_EPOCHS = 300           # Maximum training epochs
TRAIN_SPLIT = 0.8        # Train/validation split
```

**Expected runtime:** ~1 hour on NVIDIA A100 GPU

**Outputs:**
- `cfrp_pinn_best.pt` - Best model checkpoint (based on validation loss)
- `training_history.csv` - Loss curves and learning rate schedule

### 3. Evaluate Model Performance

```python
from pinn_utilities import generate_evaluation_report
from torch.utils.data import DataLoader
from pinn_cfrp_trainer import CFRPDataset
import pandas as pd

# Load test data
df = pd.read_csv("synthetic_cfrp_dataset.csv")
n_test = int(0.2 * len(df))
df_test = df.iloc[-n_test:]

# Create test dataset (with proper normalization from training)
test_dataset = CFRPDataset(df_test, input_cols, output_cols)
test_loader = DataLoader(test_dataset, batch_size=256)

# Generate comprehensive report
generate_evaluation_report(
    checkpoint_path='cfrp_pinn_best.pt',
    test_dataloader=test_loader,
    test_csv='synthetic_cfrp_dataset.csv',
    output_dir='evaluation_results'
)
```

**Generated outputs in `evaluation_results/`:**
- `evaluation_metrics.csv` - RMSE, MAE, R², MAPE for σ_HT and E_HT
- `predictions.csv` - Predicted vs. target values
- `parity_plots.png` - Scatter plots with ideal line
- `error_distribution.png` - Absolute and relative error histograms
- `physics_residuals.png` - Strength and modulus physics residual distributions
- `training_curves.png` - Training/validation loss and learning rate curves

### 4. Make Predictions on New Data

```python
from pinn_utilities import load_model_and_predict
import numpy as np

# Define new input parameters (18 features)
X_new = np.array([
    [0.15, 1.0, 2.5, 0.01, 10.0, 0.5, 0.012,    # w_f, weib_a, weib_b, L_min, L_max, L_crit, D
     0.7, 0.1, 0.3, 0.3, 1.4, 0.5,               # alpha1, beta1, gamma1, alpha2, beta2, gamma2
     3500, 60, 85000, 2335, 0.33]                # E_m, sigma_m, E_f, sigma_f, nu
])

# Load model and predict
predictions = load_model_and_predict('cfrp_pinn_best.pt', X_new)

print(f"Predicted σ_HT: {predictions[0, 0]:.2f} MPa")
print(f"Predicted E_HT: {predictions[0, 1]:.2f} MPa")
```

## Code Structure

### Main Components

#### `pinn_cfrp_trainer.py` (900+ lines)

**Core modules:**
1. **PMA (Polynomial Multiplicative Attention):** Parameter-free quadratic feature generation
2. **BranchNetwork:** MLP-PMA-MLP cascade for individual parameter encoding
3. **CSSFunction/CSSModule:** Custom autograd for cyclic spectral scheduling
4. **CFRP_PINN:** Complete architecture integrating all components
5. **EfficiencyFactors:** Vectorized χ₁, χ₂, L_n computation via trapezoidal integration
6. **PhysicsInformedLoss:** Composite loss with data + Halpin-Tsai + strength residuals

**Training infrastructure:**
- FrequencyAwareLRScheduler (modulated exponential decay)
- Gradient clipping (max norm = 1.0)
- Early stopping (patience = 50 epochs)
- Mixed precision training (FP16/FP32)

#### `pinn_utilities.py` (600+ lines)

**Evaluation tools:**
- `evaluate_model()` - Comprehensive metrics computation
- `plot_parity()` - Predictions vs. targets visualization
- `plot_error_distribution()` - Error histogram analysis
- `analyze_physics_residuals()` - Physics consistency validation
- `generate_evaluation_report()` - Complete automated evaluation pipeline

#### `data_generator.py` (provided)

JAX-based vectorized forward model implementing:
- Weibull fiber length distribution
- Bimodal Gaussian orientation distribution
- Rule of mixtures and Halpin-Tsai micromechanics
- Efficiency factor integration

## Physics-Informed Loss Components

### 1. Data Fidelity Loss
```
L_data = MSE(y_pred, y_target)
```
Normalized predictions vs. micromechanical model outputs.

### 2. Strength Physics Residual
```
L_σ = E[(σ_pred - σ_theory)²]
σ_theory = χ₁·χ₂·σ_f·w_f + (1 - w_f)·σ_m
```
Enforces multiplicative micromechanics (Kelly-Tyson shear-lag theory).

### 3. Modulus Physics Residual
```
L_E = E[(E_pred - E_theory)²]
E_theory = E_m·(1 + ξ·η·w_f)/(1 - η·w_f)
η = (E_f/E_m - 1)/(E_f/E_m + ξ)
ξ = 2·L_n/D
```
Enforces Halpin-Tsai homogenization.

### 4. Physical Admissibility
```
L_bound = E[ReLU(-E_pred)² + ReLU(-σ_pred)²]
```
Soft constraint ensuring positive material properties.

**Total loss:**
```
L_total = L_data + λ_E·L_E + λ_σ·L_σ + λ_bound·L_bound
```
Default weights: λ_E = λ_σ = λ_bound = 0.5

## Efficiency Factor Computations

### Length Efficiency (χ₁)

Integral over Weibull fiber length distribution:
```
χ₁ = ∫[L_min to L_max] φ(L; a, b) · ζ(L, L_crit) dL

ζ(L, L_crit) = { L/(2L_crit),        if L ≤ L_crit
               { 1 - L_crit/(2L),    if L > L_crit
```

**Implementation:** Trapezoidal integration with N_L = 256 grid points.

### Orientation Efficiency (χ₂)

Product of two orientation-weighted integrals:
```
χ₂ = I₁ · I₂

I₁ = ∫[0 to π/2] h_norm(θ)·cos(θ) dθ

I₂ = ∫[0 to π/2] h_norm(θ)·[cos³(θ) - ν·sin²(θ)·cos(θ)] dθ

h(θ) = α₁·exp(-((θ-β₁)/γ₁)²) + α₂·exp(-((θ-β₂)/γ₂)²)
```

**Implementation:** Trapezoidal integration with N_θ = 256 grid points.

Both computations are **fully differentiable** and **GPU-accelerated** using PyTorch.

## Cyclic Spectral Scheduling (CSS)

### Mechanism

1. **Fourier decomposition:** Shared representation decomposed into 4 frequency bands via RFFT
2. **Active band selection:** Cyclic rotation every T_cycle = 2 epochs
3. **Gradient modulation:**
   - Active band: α = 1.0 (full gradient)
   - Stabilized bands: α = 0.4 (attenuated gradient)

### Benefits

- **Overcomes spectral bias:** Prevents premature convergence to low-frequency approximations
- **Multi-scale optimization:** Ensures balanced learning across spatial scales
- **Training stability:** Attenuated gradients prevent catastrophic forgetting

### Custom Autograd Implementation

Forward pass: Perfect reconstruction (z_shared unchanged)  
Backward pass: Band-specific gradient scaling via custom PyTorch function

## Hyperparameters

### Architecture
- Embedding dimension (d_h): 32
- Attention heads: 4
- Attention dimension per head (d_k): 8
- Spectral bands (L_s): 4
- CSS cycle period (T_cycle): 2 epochs

### Optimization
- Optimizer: Adam (β₁=0.9, β₂=0.999, ε=10⁻⁸)
- Initial learning rate (η₀): 5×10⁻⁴
- LR decay factor (γ): 0.95 every 50 epochs
- LR modulation amplitude (α): 0.15
- LR modulation period (T_mod): 30 epochs
- Gradient clipping: max_norm = 1.0
- Batch size: 256

### Physics Loss Weights
- λ_E (modulus residual): 0.5
- λ_σ (strength residual): 0.5
- λ_bound (positivity constraint): 0.5

## Troubleshooting

### Issue: Training diverges early
**Solution:** Reduce initial learning rate to 1e-4, increase gradient clipping threshold

### Issue: Poor physics residuals
**Solution:** Increase λ_E and λ_σ to 1.0, verify efficiency factor integration accuracy

### Issue: Slow convergence
**Solution:** Ensure GPU is being used, reduce batch size if memory-limited

### Issue: Overfitting (train/val loss divergence)
**Solution:** Enable dropout in branch networks, increase data augmentation

## Citation

If you use this code in your research, please cite the original paper:

```bibtex
@article{cfrp_pinn_2024,
  title={A Spectrally-Stabilized, Attention-Driven Neural Architecture for 
         Micromechanical Surrogate Modeling of Carbon Fiber Reinforced Polymers},
  author={[Your Name]},
  journal={[Journal Name]},
  year={2024}
}
```

## License

MIT License - See LICENSE file for details

## Contact

For questions or issues, please open a GitHub issue or contact [your email].

---

**Note:** This implementation prioritizes clarity and modularity over absolute computational efficiency. For production deployment, consider additional optimizations such as:
- TorchScript compilation
- ONNX export for cross-platform inference
- Quantization for edge deployment
- Distributed training for larger datasets
