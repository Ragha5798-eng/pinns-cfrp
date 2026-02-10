"""
Utility Functions for PINN Training and Evaluation
===================================================
Includes:
  - Model evaluation and metrics computation
  - Prediction visualization
  - Physics residual analysis
  - Sensitivity analysis tools
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Optional
from pathlib import Path


# ============================================================
# Model Evaluation
# ============================================================

def evaluate_model(model, dataloader, criterion, mean_std, device='cuda'):
    """
    Comprehensive model evaluation with all metrics.
    
    Returns:
        metrics: Dictionary with RMSE, MAE, R², and physics residuals
        predictions: DataFrame with predictions and targets
    """
    model.eval()
    model = model.to(device)
    
    # Convert mean_std to tensors
    for key in mean_std:
        if not isinstance(mean_std[key], torch.Tensor):
            mean_std[key] = torch.tensor(mean_std[key], dtype=torch.float32).to(device)
    
    all_preds = []
    all_targets = []
    all_inputs = []
    
    total_loss = 0.0
    loss_components = {'data': 0.0, 'E_residual': 0.0, 'sigma_residual': 0.0, 'bound': 0.0}
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            # Forward pass
            y_pred = model(batch_x)
            
            # Compute loss
            loss, loss_dict = criterion(y_pred, batch_y, batch_x, mean_std)
            
            total_loss += loss_dict['total']
            for key in loss_components:
                loss_components[key] += loss_dict[key]
            
            # Denormalize predictions and targets
            y_pred_phys = criterion.denormalize(y_pred, mean_std['y_mean'], mean_std['y_std'])
            y_target_phys = criterion.denormalize(batch_y, mean_std['y_mean'], mean_std['y_std'])
            
            all_preds.append(y_pred_phys.cpu().numpy())
            all_targets.append(y_target_phys.cpu().numpy())
            all_inputs.append(batch_x.cpu().numpy())
    
    # Concatenate results
    predictions = np.vstack(all_preds)
    targets = np.vstack(all_targets)
    inputs = np.vstack(all_inputs)
    
    n_batches = len(dataloader)
    
    # Compute metrics
    metrics = {
        'total_loss': total_loss / n_batches,
        'data_loss': loss_components['data'] / n_batches,
        'E_residual': loss_components['E_residual'] / n_batches,
        'sigma_residual': loss_components['sigma_residual'] / n_batches,
        'bound_loss': loss_components['bound'] / n_batches,
    }
    
    # Per-output metrics
    for i, name in enumerate(['sigma_HT', 'E_HT']):
        pred_i = predictions[:, i]
        target_i = targets[:, i]
        
        # RMSE
        rmse = np.sqrt(np.mean((pred_i - target_i) ** 2))
        
        # MAE
        mae = np.mean(np.abs(pred_i - target_i))
        
        # R²
        ss_res = np.sum((target_i - pred_i) ** 2)
        ss_tot = np.sum((target_i - target_i.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        # MAPE (%)
        mape = np.mean(np.abs((target_i - pred_i) / (target_i + 1e-8))) * 100
        
        metrics[f'{name}_RMSE'] = rmse
        metrics[f'{name}_MAE'] = mae
        metrics[f'{name}_R2'] = r2
        metrics[f'{name}_MAPE'] = mape
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'sigma_HT_pred': predictions[:, 0],
        'sigma_HT_target': targets[:, 0],
        'E_HT_pred': predictions[:, 1],
        'E_HT_target': targets[:, 1],
    })
    
    return metrics, results_df


# ============================================================
# Visualization Functions
# ============================================================

def plot_parity(results_df, save_path='parity_plots.png'):
    """Create parity plots for predictions vs targets."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    outputs = [
        ('sigma_HT', 'Tensile Strength (MPa)'),
        ('E_HT', 'Young\'s Modulus (MPa)')
    ]
    
    for idx, (name, label) in enumerate(outputs):
        ax = axes[idx]
        
        pred = results_df[f'{name}_pred']
        target = results_df[f'{name}_target']
        
        # Scatter plot
        ax.scatter(target, pred, alpha=0.5, s=20, edgecolors='k', linewidth=0.5)
        
        # Ideal line
        min_val = min(target.min(), pred.min())
        max_val = max(target.max(), pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal')
        
        # Compute R²
        ss_res = np.sum((target - pred) ** 2)
        ss_tot = np.sum((target - target.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        ax.set_xlabel(f'Target {label}', fontsize=12)
        ax.set_ylabel(f'Predicted {label}', fontsize=12)
        ax.set_title(f'{name} Parity Plot (R² = {r2:.4f})', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Parity plots saved to: {save_path}")
    plt.close()


def plot_training_history(history_csv, save_path='training_curves.png'):
    """Plot training and validation loss curves."""
    df = pd.read_csv(history_csv)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curves
    ax = axes[0]
    ax.plot(df['epoch'], df['train_loss'], label='Train Loss', linewidth=2)
    ax.plot(df['epoch'], df['val_loss'], label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Learning rate
    ax = axes[1]
    ax.plot(df['epoch'], df['lr'], color='green', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to: {save_path}")
    plt.close()


def plot_error_distribution(results_df, save_path='error_distribution.png'):
    """Plot error distributions for both outputs."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    outputs = ['sigma_HT', 'E_HT']
    
    for idx, name in enumerate(outputs):
        pred = results_df[f'{name}_pred']
        target = results_df[f'{name}_target']
        error = pred - target
        relative_error = (error / (target + 1e-8)) * 100
        
        # Absolute error histogram
        ax = axes[idx, 0]
        ax.hist(error, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Absolute Error', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'{name} - Absolute Error Distribution', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Relative error histogram
        ax = axes[idx, 1]
        ax.hist(relative_error, bins=50, alpha=0.7, edgecolor='black', color='orange')
        ax.axvline(0, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Relative Error (%)', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'{name} - Relative Error Distribution', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Error distributions saved to: {save_path}")
    plt.close()


# ============================================================
# Physics Residual Analysis
# ============================================================

def analyze_physics_residuals(model, dataloader, mean_std, device='cuda', save_path='physics_residuals.png'):
    """
    Analyze physics residuals separately for strength and modulus.
    """
    from pinn_cfrp_trainer import EfficiencyFactors
    
    model.eval()
    model = model.to(device)
    
    # Convert mean_std to tensors
    for key in mean_std:
        if not isinstance(mean_std[key], torch.Tensor):
            mean_std[key] = torch.tensor(mean_std[key], dtype=torch.float32).to(device)
    
    efficiency = EfficiencyFactors()
    
    sigma_residuals = []
    E_residuals = []
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            
            # Forward pass
            y_pred = model(batch_x)
            
            # Denormalize
            y_pred_phys = y_pred * mean_std['y_std'] + mean_std['y_mean']
            x_phys = batch_x * mean_std['x_std'] + mean_std['x_mean']
            
            sigma_pred = y_pred_phys[:, 0]
            E_pred = y_pred_phys[:, 1]
            
            # Extract parameters
            w_f = x_phys[:, 0]
            weib_a = x_phys[:, 1]
            weib_b = x_phys[:, 2]
            L_min = x_phys[:, 3]
            L_max = x_phys[:, 4]
            L_crit = x_phys[:, 5]
            D = x_phys[:, 6]
            alpha1 = x_phys[:, 7]
            beta1 = x_phys[:, 8]
            gamma1 = x_phys[:, 9]
            alpha2 = x_phys[:, 10]
            beta2 = x_phys[:, 11]
            gamma2 = x_phys[:, 12]
            E_m = x_phys[:, 13]
            sigma_m = x_phys[:, 14]
            E_f = x_phys[:, 15]
            sigma_f = x_phys[:, 16]
            nu = x_phys[:, 17]
            
            # Compute efficiency factors
            chi1 = efficiency.compute_chi1(weib_a, weib_b, L_min, L_max, L_crit)
            chi2 = efficiency.compute_chi2(alpha1, beta1, gamma1, alpha2, beta2, gamma2, nu)
            L_n = efficiency.compute_L_n(weib_a, weib_b, L_min, L_max)
            
            # Theoretical values
            sigma_theory = chi1 * chi2 * sigma_f * w_f + (1 - w_f) * sigma_m
            
            xi = 2 * (L_n / D)
            eta = (E_f - E_m) / (E_f + xi * E_m)
            E_theory = E_m * (1 + xi * eta * w_f) / (1 - eta * w_f)
            
            # Residuals
            sigma_res = (sigma_pred - sigma_theory).cpu().numpy()
            E_res = (E_pred - E_theory).cpu().numpy()
            
            sigma_residuals.extend(sigma_res)
            E_residuals.extend(E_res)
    
    sigma_residuals = np.array(sigma_residuals)
    E_residuals = np.array(E_residuals)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Strength residuals
    ax = axes[0]
    ax.hist(sigma_residuals, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Strength Residual (MPa)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Strength Physics Residual\nMean: {sigma_residuals.mean():.4f}, Std: {sigma_residuals.std():.4f}',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Modulus residuals
    ax = axes[1]
    ax.hist(E_residuals, bins=50, alpha=0.7, edgecolor='black', color='orange')
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Modulus Residual (MPa)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Modulus Physics Residual\nMean: {E_residuals.mean():.4f}, Std: {E_residuals.std():.4f}',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Physics residuals plot saved to: {save_path}")
    plt.close()
    
    return {
        'sigma_residual_mean': sigma_residuals.mean(),
        'sigma_residual_std': sigma_residuals.std(),
        'E_residual_mean': E_residuals.mean(),
        'E_residual_std': E_residuals.std(),
    }


# ============================================================
# Model Inference
# ============================================================

def load_model_and_predict(checkpoint_path, X_new, device='cuda'):
    """
    Load trained model and make predictions on new data.
    
    Args:
        checkpoint_path: Path to saved model checkpoint
        X_new: New input data (numpy array or DataFrame)
        device: Device to use
    
    Returns:
        predictions: Denormalized predictions [sigma_HT, E_HT]
    """
    from pinn_cfrp_trainer import CFRP_PINN
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    mean_std = checkpoint['mean_std']
    
    # Initialize model
    model = CFRP_PINN(d_h=32, n_heads=4, n_bands=4)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Normalize inputs
    if isinstance(X_new, pd.DataFrame):
        X_new = X_new.values
    
    X_mean = mean_std['x_mean'].cpu().numpy() if isinstance(mean_std['x_mean'], torch.Tensor) else mean_std['x_mean']
    X_std = mean_std['x_std'].cpu().numpy() if isinstance(mean_std['x_std'], torch.Tensor) else mean_std['x_std']
    
    X_norm = (X_new - X_mean) / X_std
    X_tensor = torch.tensor(X_norm, dtype=torch.float32).to(device)
    
    # Predict
    with torch.no_grad():
        Y_pred_norm = model(X_tensor)
    
    # Denormalize
    Y_mean = mean_std['y_mean'].cpu().numpy() if isinstance(mean_std['y_mean'], torch.Tensor) else mean_std['y_mean']
    Y_std = mean_std['y_std'].cpu().numpy() if isinstance(mean_std['y_std'], torch.Tensor) else mean_std['y_std']
    
    Y_pred = Y_pred_norm.cpu().numpy() * Y_std + Y_mean
    
    return Y_pred


# ============================================================
# Comprehensive Evaluation Report
# ============================================================

def generate_evaluation_report(checkpoint_path, test_dataloader, test_csv, output_dir='evaluation_results'):
    """
    Generate comprehensive evaluation report with all visualizations and metrics.
    """
    from pinn_cfrp_trainer import CFRP_PINN, PhysicsInformedLoss
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    mean_std = checkpoint['mean_std']
    
    # Initialize model
    model = CFRP_PINN(d_h=32, n_heads=4, n_bands=4)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Initialize criterion
    criterion = PhysicsInformedLoss()
    
    print("Evaluating model...")
    metrics, results_df = evaluate_model(model, test_dataloader, criterion, mean_std, device)
    
    # Print metrics
    print("\n" + "="*80)
    print("EVALUATION METRICS")
    print("="*80)
    for key, value in metrics.items():
        print(f"{key:25s}: {value:.6f}")
    print("="*80 + "\n")
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(output_dir / 'evaluation_metrics.csv', index=False)
    print(f"Metrics saved to: {output_dir / 'evaluation_metrics.csv'}")
    
    # Save predictions
    results_df.to_csv(output_dir / 'predictions.csv', index=False)
    print(f"Predictions saved to: {output_dir / 'predictions.csv'}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_parity(results_df, save_path=output_dir / 'parity_plots.png')
    plot_error_distribution(results_df, save_path=output_dir / 'error_distribution.png')
    
    # Physics residuals
    print("Analyzing physics residuals...")
    residual_stats = analyze_physics_residuals(
        model, test_dataloader, mean_std, device,
        save_path=output_dir / 'physics_residuals.png'
    )
    
    # Training history (if available)
    if Path('training_history.csv').exists():
        plot_training_history('training_history.csv', 
                             save_path=output_dir / 'training_curves.png')
    
    print(f"\nEvaluation complete! Results saved to: {output_dir}/")
    
    return metrics, results_df, residual_stats


# ============================================================
# Example Usage
# ============================================================

if __name__ == "__main__":
    """
    Example usage of evaluation utilities.
    """
    from torch.utils.data import DataLoader
    from pinn_cfrp_trainer import CFRPDataset
    
    # Load test data
    df_test = pd.read_csv("synthetic_cfrp_dataset.csv")
    
    input_cols = ["w_f", "weib_a", "weib_b", "L_min", "L_max", "L_crit", "D",
                  "alpha1", "beta1", "gamma1", "alpha2", "beta2", "gamma2",
                  "E_m", "sigma_m", "E_f", "sigma_f", "nu"]
    output_cols = ["sigma_HT", "E_HT"]
    
    # Take last 20% as test set
    n_test = int(0.2 * len(df_test))
    df_test = df_test.iloc[-n_test:]
    
    # Normalize (using same statistics as training)
    # In practice, load these from checkpoint
    test_dataset = CFRPDataset(df_test, input_cols, output_cols)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    # Generate evaluation report
    generate_evaluation_report(
        checkpoint_path='cfrp_pinn_best.pt',
        test_dataloader=test_loader,
        test_csv='synthetic_cfrp_dataset.csv',
        output_dir='evaluation_results'
    )
