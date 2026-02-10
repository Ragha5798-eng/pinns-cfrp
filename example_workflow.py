"""
Complete Example: Train and Evaluate CFRP PINN
==============================================
This script demonstrates the full workflow:
1. Data preparation
2. Model training
3. Comprehensive evaluation
4. Predictions on new samples
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path

# Import our modules
from pinn_cfrp_trainer import (
    CFRP_PINN, 
    CFRPDataset, 
    train_pinn
)
from pinn_utilities import (
    generate_evaluation_report,
    load_model_and_predict,
    plot_training_history
)
from torch.utils.data import DataLoader


def prepare_data(csv_path="synthetic_cfrp_dataset.csv", train_split=0.8, val_split=0.1):
    """
    Prepare datasets with proper normalization.
    
    Returns:
        train_loader, val_loader, test_loader, mean_std
    """
    print("Loading dataset...")
    df = pd.read_csv(csv_path)
    print(f"Total samples: {len(df)}")
    
    # Define columns
    input_cols = ["w_f", "weib_a", "weib_b", "L_min", "L_max", "L_crit", "D",
                  "alpha1", "beta1", "gamma1", "alpha2", "beta2", "gamma2",
                  "E_m", "sigma_m", "E_f", "sigma_f", "nu"]
    output_cols = ["sigma_HT", "E_HT"]
    
    # Extract data
    X_data = df[input_cols].values
    Y_data = df[output_cols].values
    
    # Compute normalization statistics
    X_mean = X_data.mean(axis=0)
    X_std = X_data.std(axis=0)
    Y_mean = Y_data.mean(axis=0)
    Y_std = Y_data.std(axis=0)
    
    print(f"\nInput statistics:")
    print(f"  Mean range: [{X_mean.min():.4f}, {X_mean.max():.4f}]")
    print(f"  Std range: [{X_std.min():.4f}, {X_std.max():.4f}]")
    print(f"\nOutput statistics:")
    print(f"  σ_HT: mean={Y_mean[0]:.2f}, std={Y_std[0]:.2f}")
    print(f"  E_HT: mean={Y_mean[1]:.2f}, std={Y_std[1]:.2f}")
    
    # Normalize
    df[input_cols] = (X_data - X_mean) / X_std
    df[output_cols] = (Y_data - Y_mean) / Y_std
    
    # Split data
    n_total = len(df)
    n_train = int(train_split * n_total)
    n_val = int(val_split * n_total)
    
    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:n_train+n_val]
    test_df = df.iloc[n_train+n_val:]
    
    print(f"\nDataset splits:")
    print(f"  Training: {len(train_df)} samples ({100*len(train_df)/n_total:.1f}%)")
    print(f"  Validation: {len(val_df)} samples ({100*len(val_df)/n_total:.1f}%)")
    print(f"  Test: {len(test_df)} samples ({100*len(test_df)/n_total:.1f}%)")
    
    # Create datasets
    train_dataset = CFRPDataset(train_df, input_cols, output_cols)
    val_dataset = CFRPDataset(val_df, input_cols, output_cols)
    test_dataset = CFRPDataset(test_df, input_cols, output_cols)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)
    
    # Store normalization statistics
    mean_std = {
        'x_mean': X_mean,
        'x_std': X_std,
        'y_mean': Y_mean,
        'y_std': Y_std
    }
    
    return train_loader, val_loader, test_loader, mean_std


def main():
    """Main execution function."""
    
    print("="*80)
    print("CFRP PINN: Complete Training and Evaluation Pipeline")
    print("="*80)
    print()
    
    # Configuration
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    N_EPOCHS = 300
    CHECKPOINT_PATH = 'cfrp_pinn_best.pt'
    
    print(f"Device: {DEVICE}")
    print(f"Max epochs: {N_EPOCHS}")
    print()
    
    # Step 1: Prepare data
    print("STEP 1: Data Preparation")
    print("-" * 80)
    train_loader, val_loader, test_loader, mean_std = prepare_data()
    print()
    
    # Step 2: Initialize model
    print("STEP 2: Model Initialization")
    print("-" * 80)
    model = CFRP_PINN(d_h=32, n_heads=4, n_bands=4)
    
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model architecture:")
    print(f"  Embedding dimension: 32")
    print(f"  Attention heads: 4")
    print(f"  Spectral bands: 4")
    print(f"  Total parameters: {n_params:,}")
    print(f"  Trainable parameters: {n_trainable:,}")
    print()
    
    # Step 3: Train model
    print("STEP 3: Model Training")
    print("-" * 80)
    history = train_pinn(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        mean_std=mean_std,
        n_epochs=N_EPOCHS,
        device=DEVICE,
        save_path=CHECKPOINT_PATH
    )
    print()
    
    # Step 4: Evaluate model
    print("STEP 4: Comprehensive Evaluation")
    print("-" * 80)
    
    output_dir = Path('evaluation_results')
    output_dir.mkdir(exist_ok=True)
    
    metrics, results_df, residual_stats = generate_evaluation_report(
        checkpoint_path=CHECKPOINT_PATH,
        test_dataloader=test_loader,
        test_csv='synthetic_cfrp_dataset.csv',
        output_dir=str(output_dir)
    )
    
    print("\nFinal Test Metrics:")
    print(f"  σ_HT - RMSE: {metrics['sigma_HT_RMSE']:.4f}, R²: {metrics['sigma_HT_R2']:.4f}")
    print(f"  E_HT - RMSE: {metrics['E_HT_RMSE']:.4f}, R²: {metrics['E_HT_R2']:.4f}")
    print()
    
    # Step 5: Demonstrate predictions
    print("STEP 5: Example Predictions on New Data")
    print("-" * 80)
    
    # Create example input (typical CFRP configuration)
    X_example = np.array([
        # w_f, weib_a, weib_b, L_min, L_max, L_crit, D
        [0.10, 1.0, 2.0, 0.01, 10.0, 0.4, 0.012,
         # alpha1, beta1, gamma1, alpha2, beta2, gamma2
         0.6, 0.1, 0.3, 0.4, 1.2, 0.5,
         # E_m, sigma_m, E_f, sigma_f, nu
         3500, 60, 85000, 2335, 0.33],
        
        [0.15, 1.5, 2.5, 0.01, 10.0, 0.5, 0.012,
         0.7, 0.15, 0.4, 0.3, 1.4, 0.6,
         3500, 60, 85000, 2335, 0.33],
        
        [0.20, 0.8, 2.2, 0.01, 10.0, 0.6, 0.012,
         0.5, 0.05, 0.2, 0.5, 1.0, 0.4,
         3500, 60, 85000, 2335, 0.33],
    ])
    
    predictions = load_model_and_predict(CHECKPOINT_PATH, X_example, device=DEVICE)
    
    print("Example predictions:")
    print("\nSample 1 (w_f=0.10):")
    print(f"  Predicted σ_HT: {predictions[0, 0]:.2f} MPa")
    print(f"  Predicted E_HT: {predictions[0, 1]:.2f} MPa")
    
    print("\nSample 2 (w_f=0.15):")
    print(f"  Predicted σ_HT: {predictions[1, 0]:.2f} MPa")
    print(f"  Predicted E_HT: {predictions[1, 1]:.2f} MPa")
    
    print("\nSample 3 (w_f=0.20):")
    print(f"  Predicted σ_HT: {predictions[2, 0]:.2f} MPa")
    print(f"  Predicted E_HT: {predictions[2, 1]:.2f} MPa")
    print()
    
    # Summary
    print("="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print(f"\nGenerated files:")
    print(f"  ✓ {CHECKPOINT_PATH} - Best model checkpoint")
    print(f"  ✓ training_history.csv - Training curves")
    print(f"  ✓ {output_dir}/evaluation_metrics.csv - Performance metrics")
    print(f"  ✓ {output_dir}/predictions.csv - Test predictions")
    print(f"  ✓ {output_dir}/parity_plots.png - Visual validation")
    print(f"  ✓ {output_dir}/error_distribution.png - Error analysis")
    print(f"  ✓ {output_dir}/physics_residuals.png - Physics consistency")
    print(f"  ✓ {output_dir}/training_curves.png - Loss evolution")
    print()
    print("Next steps:")
    print("  1. Review evaluation_results/ for model performance")
    print("  2. Analyze physics residuals for consistency")
    print("  3. Use load_model_and_predict() for inference on new data")
    print()


if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Run pipeline
    main()
