"""
Physics-Informed Neural Network for CFRP Micromechanical Modeling
==================================================================
Implements the full architecture from the paper:
  - Parameter-specific branch networks with MLP-PMA-MLP cascades
  - Polynomial Multiplicative Attention (PMA) modules
  - Multi-head attention for cross-parameter coupling
  - Cyclic Spectral Scheduling (CSS) with Fourier decomposition
  - Physics-informed loss with Halpin-Tsai and strength residuals
  - Efficiency factor computations (chi1, chi2) via numerical integration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
import math
from pathlib import Path


# ============================================================
# Polynomial Multiplicative Attention (PMA) Module
# ============================================================

class PMA(nn.Module):
    """
    Parameter-free Polynomial Multiplicative Attention.
    
    Partitions input into pass-through, key, and query components.
    Generates quadratic features via Hadamard product of key and query.
    Output dimension: 2 * d_h / 3
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Input tensor of shape (batch_size, d_h)
        
        Returns:
            Output tensor of shape (batch_size, 2*d_h/3)
        """
        d_h = z.shape[-1]
        chunk_size = d_h // 3
        
        # Tripartite partitioning
        z_pass = z[..., :chunk_size]                    # Linear pathway
        z_key = z[..., chunk_size:2*chunk_size]         # Key component
        z_query = z[..., 2*chunk_size:3*chunk_size]     # Query component
        
        # Hadamard product for quadratic interactions
        z_mult = z_key * z_query
        
        # Concatenate linear and quadratic channels
        return torch.cat([z_pass, z_mult], dim=-1)


# ============================================================
# Parameter-Specific Branch Network
# ============================================================

class BranchNetwork(nn.Module):
    """
    MLP-PMA-MLP cascade for individual parameter encoding.
    
    Architecture:
        1. Affine embedding: R -> R^{d_h}
        2. PMA: R^{d_h} -> R^{2*d_h/3}
        3. Affine projection: R^{2*d_h/3} -> R^{d_h}
    """
    def __init__(self, d_h: int = 32):
        super().__init__()
        self.d_h = d_h
        
        # Stage 1: Initial affine embedding
        self.fc1 = nn.Linear(1, d_h)
        
        # Stage 2: PMA (parameter-free)
        self.pma = PMA()
        
        # Stage 3: Output affine projection
        pma_out_dim = 2 * d_h // 3
        self.fc2 = nn.Linear(pma_out_dim, d_h)
    
    def forward(self, p: torch.Tensor) -> torch.Tensor:
        """
        Args:
            p: Scalar parameter of shape (batch_size, 1)
        
        Returns:
            Embedding of shape (batch_size, d_h)
        """
        # Stage 1: Affine embedding
        z0 = self.fc1(p)  # (batch, d_h)
        
        # Stage 2: PMA
        z1 = self.pma(z0)  # (batch, 2*d_h/3)
        
        # Stage 3: Affine projection
        h = self.fc2(z1)  # (batch, d_h)
        
        return h


# ============================================================
# Cyclic Spectral Scheduling (CSS) - Custom Autograd
# ============================================================

class CSSFunction(torch.autograd.Function):
    """
    Custom autograd function implementing Cyclic Spectral Scheduling.
    
    Forward: Perfect reconstruction via RFFT/IRFFT
    Backward: Asymmetric gradient modulation per frequency band
    """
    @staticmethod
    def forward(ctx, z_shared, active_band, n_bands=4, alpha_active=1.0, alpha_stable=0.4):
        """
        Args:
            z_shared: Shared representation (batch, d_h)
            active_band: Current active band index (0 to n_bands-1)
            n_bands: Number of frequency bands
            alpha_active: Gradient scaling for active band
            alpha_stable: Gradient scaling for non-active bands
        
        Returns:
            Reconstructed z_shared (identical to input in forward pass)
        """
        batch_size, d_h = z_shared.shape
        
        # Fourier decomposition
        z_freq = torch.fft.rfft(z_shared, dim=-1)  # (batch, d_h//2 + 1)
        freq_len = z_freq.shape[-1]
        
        # Create frequency band masks
        band_indices = torch.linspace(0, freq_len, n_bands + 1).long()
        masks = []
        for i in range(n_bands):
            mask = torch.zeros(freq_len, dtype=torch.bool, device=z_shared.device)
            mask[band_indices[i]:band_indices[i+1]] = True
            masks.append(mask)
        
        # Store for backward pass
        ctx.save_for_backward(z_freq)
        ctx.active_band = active_band
        ctx.n_bands = n_bands
        ctx.alpha_active = alpha_active
        ctx.alpha_stable = alpha_stable
        ctx.masks = masks
        ctx.d_h = d_h
        
        # Forward pass: perfect reconstruction
        return z_shared.clone()
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Apply asymmetric gradient modulation to different frequency bands.
        """
        z_freq, = ctx.saved_tensors
        
        # Transform gradient to frequency domain
        grad_freq = torch.fft.rfft(grad_output, dim=-1)
        
        # Apply band-specific modulation
        modulated_grad_freq = torch.zeros_like(grad_freq)
        for band_idx in range(ctx.n_bands):
            mask = ctx.masks[band_idx]
            alpha = ctx.alpha_active if band_idx == ctx.active_band else ctx.alpha_stable
            modulated_grad_freq[:, mask] = alpha * grad_freq[:, mask]
        
        # Transform back to spatial domain
        grad_spatial = torch.fft.irfft(modulated_grad_freq, n=ctx.d_h, dim=-1)
        
        return grad_spatial, None, None, None, None


# ============================================================
# Cyclic Spectral Scheduling Module
# ============================================================

class CSSModule(nn.Module):
    """
    Cyclic Spectral Scheduling wrapper with epoch tracking.
    """
    def __init__(self, n_bands: int = 4, T_cycle: int = 2, 
                 alpha_active: float = 1.0, alpha_stable: float = 0.4):
        super().__init__()
        self.n_bands = n_bands
        self.T_cycle = T_cycle
        self.alpha_active = alpha_active
        self.alpha_stable = alpha_stable
        self.current_epoch = 0
    
    def forward(self, z_shared: torch.Tensor) -> torch.Tensor:
        """Apply CSS with current active band."""
        active_band = (self.current_epoch // self.T_cycle) % self.n_bands
        return CSSFunction.apply(z_shared, active_band, self.n_bands, 
                                 self.alpha_active, self.alpha_stable)
    
    def update_epoch(self, epoch: int):
        """Update current epoch for band cycling."""
        self.current_epoch = epoch


# ============================================================
# Main PINN Architecture
# ============================================================

class CFRP_PINN(nn.Module):
    """
    Complete Physics-Informed Neural Network for CFRP micromechanics.
    
    Architecture stages:
        1. Parameter-specific branch networks (18 branches)
        2. Multi-head attention for cross-parameter coupling
        3. Cyclic Spectral Scheduling (CSS)
        4. PMA-enhanced prediction head
    """
    def __init__(self, d_h: int = 32, n_heads: int = 4, n_bands: int = 4):
        super().__init__()
        self.d_h = d_h
        self.n_heads = n_heads
        self.n_params = 18  # Number of input parameters
        
        # Stage 1: Parameter-specific branch networks
        self.branches = nn.ModuleList([BranchNetwork(d_h) for _ in range(self.n_params)])
        
        # Stage 2: Multi-head attention
        self.mha = nn.MultiheadAttention(
            embed_dim=d_h,
            num_heads=n_heads,
            batch_first=True
        )
        
        # Aggregation projection
        self.aggregation_proj = nn.Linear(self.n_params * d_h, d_h)
        
        # Stage 3: Cyclic Spectral Scheduling
        self.css = CSSModule(n_bands=n_bands)
        
        # Stage 4: Prediction head with PMA
        self.output_pma = PMA()
        pma_out_dim = 2 * d_h // 3
        self.output_proj = nn.Linear(pma_out_dim, 2)  # [sigma_HT, E_HT]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input parameters (batch, 18)
        
        Returns:
            predictions: [sigma_HT, E_HT] (batch, 2)
        """
        batch_size = x.shape[0]
        
        # Stage 1: Branch networks
        embeddings = []
        for i, branch in enumerate(self.branches):
            p_i = x[:, i:i+1]  # (batch, 1)
            h_i = branch(p_i)   # (batch, d_h)
            embeddings.append(h_i)
        
        # Stack embeddings: (batch, n_params, d_h)
        H = torch.stack(embeddings, dim=1)
        
        # Stage 2: Multi-head attention
        # MHA expects (batch, seq_len, embed_dim)
        H_attn, _ = self.mha(H, H, H)  # (batch, n_params, d_h)
        
        # Flatten and aggregate
        H_flat = H_attn.reshape(batch_size, -1)  # (batch, n_params * d_h)
        z_shared = self.aggregation_proj(H_flat)  # (batch, d_h)
        
        # Stage 3: Cyclic Spectral Scheduling
        z_shared = self.css(z_shared)  # (batch, d_h)
        
        # Stage 4: Prediction head
        z_pma = self.output_pma(z_shared)  # (batch, 2*d_h/3)
        y_pred = self.output_proj(z_pma)   # (batch, 2)
        
        return y_pred
    
    def update_epoch(self, epoch: int):
        """Update CSS epoch counter."""
        self.css.update_epoch(epoch)


# ============================================================
# Efficiency Factor Computations (for Physics Loss)
# ============================================================

class EfficiencyFactors:
    """
    Vectorized computation of chi1 (length efficiency) and chi2 (orientation efficiency)
    using numerical integration.
    """
    def __init__(self, N_L: int = 256, N_theta: int = 256):
        self.N_L = N_L
        self.N_theta = N_theta
    
    @staticmethod
    def weibull_pdf(L: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Weibull PDF with numerical stability."""
        a = torch.clamp(a, min=1e-9)
        return (b / a) * (L / a) ** (b - 1) * torch.exp(-(L / a) ** b)
    
    @staticmethod
    def trapz(y: torch.Tensor, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Trapezoidal integration."""
        dx = x[1:] - x[:-1]
        avg_y = 0.5 * (y[..., 1:] + y[..., :-1])
        return torch.sum(avg_y * dx, dim=dim)
    
    def compute_chi1(self, weib_a: torch.Tensor, weib_b: torch.Tensor,
                     L_min: torch.Tensor, L_max: torch.Tensor,
                     L_crit: torch.Tensor) -> torch.Tensor:
        """
        Compute length efficiency factor via numerical integration.
        
        Args:
            weib_a, weib_b: Weibull parameters (batch,)
            L_min, L_max: Length bounds (batch,)
            L_crit: Critical length (batch,)
        
        Returns:
            chi1: Length efficiency (batch,)
        """
        batch_size = weib_a.shape[0]
        device = weib_a.device
        
        # Create global L grid (shared across batch)
        L_grid = torch.linspace(0.0, 10.0, self.N_L, device=device)  # (N_L,)
        L_grid = L_grid.unsqueeze(0).expand(batch_size, -1)  # (batch, N_L)
        
        # Expand parameters for broadcasting
        a = weib_a.unsqueeze(-1)  # (batch, 1)
        b = weib_b.unsqueeze(-1)
        L_min_b = L_min.unsqueeze(-1)
        L_max_b = L_max.unsqueeze(-1)
        L_crit_b = L_crit.unsqueeze(-1)
        
        # Compute PDF
        pdf = self.weibull_pdf(L_grid, a, b)  # (batch, N_L)
        
        # Apply bounds mask
        mask = (L_grid >= L_min_b) & (L_grid <= L_max_b)
        pdf = pdf * mask.float()
        
        # Load transfer efficiency
        zeta = torch.where(
            L_grid <= L_crit_b,
            L_grid / (2 * L_crit_b),
            1 - L_crit_b / (2 * torch.clamp(L_grid, min=1e-12))
        )
        
        # Integrate
        integrand = pdf * zeta
        chi1 = self.trapz(integrand, L_grid[0], dim=-1)  # (batch,)
        
        return chi1
    
    def compute_chi2(self, alpha1: torch.Tensor, beta1: torch.Tensor, gamma1: torch.Tensor,
                     alpha2: torch.Tensor, beta2: torch.Tensor, gamma2: torch.Tensor,
                     nu: torch.Tensor) -> torch.Tensor:
        """
        Compute orientation efficiency factor.
        
        Args:
            alpha1, beta1, gamma1: First Gaussian term parameters (batch,)
            alpha2, beta2, gamma2: Second Gaussian term parameters (batch,)
            nu: Poisson's ratio (batch,)
        
        Returns:
            chi2: Orientation efficiency (batch,)
        """
        batch_size = alpha1.shape[0]
        device = alpha1.device
        
        # Create theta grid
        theta_grid = torch.linspace(0.0, math.pi / 2, self.N_theta, device=device)
        theta_grid = theta_grid.unsqueeze(0).expand(batch_size, -1)  # (batch, N_theta)
        
        # Expand parameters
        a1 = alpha1.unsqueeze(-1)
        b1 = beta1.unsqueeze(-1)
        g1 = gamma1.unsqueeze(-1)
        a2 = alpha2.unsqueeze(-1)
        b2 = beta2.unsqueeze(-1)
        g2 = gamma2.unsqueeze(-1)
        nu_b = nu.unsqueeze(-1)
        
        # Orientation PDF
        term1 = a1 * torch.exp(-((theta_grid - b1) / g1) ** 2)
        term2 = a2 * torch.exp(-((theta_grid - b2) / g2) ** 2)
        h_raw = term1 + term2
        
        # Normalize
        Z = self.trapz(h_raw, theta_grid[0], dim=-1, keepdim=True)  # (batch, 1)
        Z = torch.clamp(Z, min=1.0)
        h_norm = h_raw / Z
        
        # Compute I1 and I2
        cos_theta = torch.cos(theta_grid)
        sin_theta = torch.sin(theta_grid)
        
        I1 = self.trapz(h_norm * cos_theta, theta_grid[0], dim=-1)
        I2 = self.trapz(
            h_norm * (cos_theta ** 3 - nu_b * sin_theta ** 2 * cos_theta),
            theta_grid[0],
            dim=-1
        )
        
        chi2 = I1 * I2
        return chi2
    
    def compute_L_n(self, weib_a: torch.Tensor, weib_b: torch.Tensor,
                    L_min: torch.Tensor, L_max: torch.Tensor) -> torch.Tensor:
        """
        Compute number-averaged fiber length.
        
        Returns:
            L_n: Mean fiber length (batch,)
        """
        batch_size = weib_a.shape[0]
        device = weib_a.device
        
        L_grid = torch.linspace(0.0, 10.0, self.N_L, device=device)
        L_grid = L_grid.unsqueeze(0).expand(batch_size, -1)
        
        a = weib_a.unsqueeze(-1)
        b = weib_b.unsqueeze(-1)
        L_min_b = L_min.unsqueeze(-1)
        L_max_b = L_max.unsqueeze(-1)
        
        pdf = self.weibull_pdf(L_grid, a, b)
        mask = (L_grid >= L_min_b) & (L_grid <= L_max_b)
        pdf = pdf * mask.float()
        
        # Normalize
        Z = self.trapz(pdf, L_grid[0], dim=-1, keepdim=True)
        pdf_norm = pdf / torch.clamp(Z, min=1e-12)
        
        # Mean length
        L_n = self.trapz(pdf_norm * L_grid, L_grid[0], dim=-1)
        return L_n


# ============================================================
# Physics-Informed Loss Function
# ============================================================

class PhysicsInformedLoss(nn.Module):
    """
    Composite loss with data fidelity and physics residuals.
    """
    def __init__(self, lambda_E: float = 0.5, lambda_sigma: float = 0.5,
                 lambda_bound: float = 0.5, N_L: int = 256, N_theta: int = 256):
        super().__init__()
        self.lambda_E = lambda_E
        self.lambda_sigma = lambda_sigma
        self.lambda_bound = lambda_bound
        self.efficiency = EfficiencyFactors(N_L, N_theta)
    
    def forward(self, y_pred: torch.Tensor, y_target: torch.Tensor,
                x: torch.Tensor, mean_std: dict) -> Tuple[torch.Tensor, dict]:
        """
        Compute composite physics-informed loss.
        
        Args:
            y_pred: Network predictions [sigma_HT, E_HT] (batch, 2) - NORMALIZED
            y_target: Target values [sigma_HT, E_HT] (batch, 2) - NORMALIZED
            x: Input parameters (batch, 18) - NORMALIZED
            mean_std: Dictionary with normalization statistics
        
        Returns:
            total_loss: Combined loss
            loss_dict: Individual loss components
        """
        # Data consistency loss (on normalized outputs)
        loss_data = F.mse_loss(y_pred, y_target)
        
        # Denormalize predictions and inputs for physics computation
        y_pred_phys = self.denormalize(y_pred, mean_std['y_mean'], mean_std['y_std'])
        x_phys = self.denormalize(x, mean_std['x_mean'], mean_std['x_std'])
        
        sigma_pred = y_pred_phys[:, 0]  # (batch,)
        E_pred = y_pred_phys[:, 1]      # (batch,)
        
        # Extract parameters (all in physical units)
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
        chi1 = self.efficiency.compute_chi1(weib_a, weib_b, L_min, L_max, L_crit)
        chi2 = self.efficiency.compute_chi2(alpha1, beta1, gamma1, alpha2, beta2, gamma2, nu)
        L_n = self.efficiency.compute_L_n(weib_a, weib_b, L_min, L_max)
        
        # Strength residual (Eq. from paper)
        sigma_theory = chi1 * chi2 * sigma_f * w_f + (1 - w_f) * sigma_m
        loss_sigma = torch.mean((sigma_pred - sigma_theory) ** 2)
        
        # Modulus residual (Halpin-Tsai)
        xi = 2 * (L_n / D)
        eta = (E_f - E_m) / (E_f + xi * E_m)
        E_theory = E_m * (1 + xi * eta * w_f) / (1 - eta * w_f)
        loss_E = torch.mean((E_pred - E_theory) ** 2)
        
        # Bound constraints (positivity)
        loss_bound = torch.mean(
            torch.relu(-E_pred) ** 2 + torch.relu(-sigma_pred) ** 2
        )
        
        # Total loss
        total_loss = (loss_data + 
                     self.lambda_E * loss_E + 
                     self.lambda_sigma * loss_sigma + 
                     self.lambda_bound * loss_bound)
        
        loss_dict = {
            'total': total_loss.item(),
            'data': loss_data.item(),
            'E_residual': loss_E.item(),
            'sigma_residual': loss_sigma.item(),
            'bound': loss_bound.item()
        }
        
        return total_loss, loss_dict
    
    @staticmethod
    def denormalize(x_norm: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """Denormalize from standardized values."""
        return x_norm * std + mean


# ============================================================
# Dataset and DataLoader
# ============================================================

class CFRPDataset(Dataset):
    """PyTorch dataset for CFRP micromechanical data."""
    def __init__(self, df: pd.DataFrame, input_cols: List[str], output_cols: List[str]):
        self.X = torch.tensor(df[input_cols].values, dtype=torch.float32)
        self.Y = torch.tensor(df[output_cols].values, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# ============================================================
# Learning Rate Schedule
# ============================================================

class FrequencyAwareLRScheduler:
    """
    Modulated exponential decay learning rate schedule.
    
    eta(E) = eta_0 * gamma^(E/tau) * (1 + alpha * sin(2*pi*E/T_mod))
    """
    def __init__(self, optimizer, eta_0: float = 5e-4, gamma: float = 0.95,
                 tau: int = 50, alpha: float = 0.15, T_mod: int = 30):
        self.optimizer = optimizer
        self.eta_0 = eta_0
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.T_mod = T_mod
    
    def step(self, epoch: int):
        """Update learning rate based on epoch."""
        lr = self.eta_0 * (self.gamma ** (epoch / self.tau))
        lr *= (1 + self.alpha * math.sin(2 * math.pi * epoch / self.T_mod))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr


# ============================================================
# Training Loop
# ============================================================

def train_pinn(
    model: CFRP_PINN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    mean_std: dict,
    n_epochs: int = 300,
    device: str = 'cuda',
    save_path: str = 'pinn_checkpoint.pt'
):
    """
    Complete training loop with physics-informed loss and CSS.
    
    Args:
        model: PINN model
        train_loader: Training data loader
        val_loader: Validation data loader
        mean_std: Normalization statistics
        n_epochs: Number of training epochs
        device: 'cuda' or 'cpu'
        save_path: Path to save best model
    """
    model = model.to(device)
    
    # Convert mean/std to tensors on device
    for key in mean_std:
        mean_std[key] = torch.tensor(mean_std[key], dtype=torch.float32).to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, 
                                 betas=(0.9, 0.999), eps=1e-8)
    
    # Learning rate scheduler
    lr_scheduler = FrequencyAwareLRScheduler(optimizer)
    
    # Physics-informed loss
    criterion = PhysicsInformedLoss(lambda_E=0.5, lambda_sigma=0.5, lambda_bound=0.5)
    criterion = criterion.to(device)
    
    # Gradient clipping
    max_grad_norm = 1.0
    
    # Early stopping
    best_val_loss = float('inf')
    patience = 50
    patience_counter = 0
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'lr': []
    }
    
    print("Starting PINN Training")
    print("=" * 80)
    
    for epoch in range(n_epochs):
        # Update CSS epoch
        model.update_epoch(epoch)
        
        # Update learning rate
        current_lr = lr_scheduler.step(epoch)
        history['lr'].append(current_lr)
        
        # Training phase
        model.train()
        train_loss_accum = 0.0
        train_components = {k: 0.0 for k in ['data', 'E_residual', 'sigma_residual', 'bound']}
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            y_pred = model(batch_x)
            
            # Compute loss
            loss, loss_dict = criterion(y_pred, batch_y, batch_x, mean_std)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            # Optimizer step
            optimizer.step()
            
            # Accumulate losses
            train_loss_accum += loss_dict['total']
            for key in train_components:
                train_components[key] += loss_dict[key]
        
        # Average training losses
        n_train_batches = len(train_loader)
        train_loss = train_loss_accum / n_train_batches
        for key in train_components:
            train_components[key] /= n_train_batches
        
        history['train_loss'].append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss_accum = 0.0
        val_components = {k: 0.0 for k in ['data', 'E_residual', 'sigma_residual', 'bound']}
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                y_pred = model(batch_x)
                loss, loss_dict = criterion(y_pred, batch_y, batch_x, mean_std)
                
                val_loss_accum += loss_dict['total']
                for key in val_components:
                    val_components[key] += loss_dict[key]
        
        # Average validation losses
        n_val_batches = len(val_loader)
        val_loss = val_loss_accum / n_val_batches
        for key in val_components:
            val_components[key] /= n_val_batches
        
        history['val_loss'].append(val_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{n_epochs}")
            print(f"  LR: {current_lr:.6f}")
            print(f"  Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
            print(f"    Data: {train_components['data']:.6f} | {val_components['data']:.6f}")
            print(f"    E_res: {train_components['E_residual']:.6f} | {val_components['E_residual']:.6f}")
            print(f"    σ_res: {train_components['sigma_residual']:.6f} | {val_components['sigma_residual']:.6f}")
            print(f"    Bound: {train_components['bound']:.6f} | {val_components['bound']:.6f}")
            print()
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'mean_std': mean_std,
            }, save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    print("=" * 80)
    print(f"Training complete. Best validation loss: {best_val_loss:.6f}")
    print(f"Model saved to: {save_path}")
    
    return history


# ============================================================
# Main Execution
# ============================================================

def main():
    """Main training pipeline."""
    import warnings
    warnings.filterwarnings('ignore')
    
    # Configuration
    DATA_PATH = "synthetic_cfrp_dataset.csv"
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = 256
    N_EPOCHS = 300
    TRAIN_SPLIT = 0.8
    
    print(f"Device: {DEVICE}")
    print(f"Loading data from: {DATA_PATH}")
    
    # Load dataset
    df = pd.read_csv(DATA_PATH)
    
    # Define columns
    input_cols = ["w_f", "weib_a", "weib_b", "L_min", "L_max", "L_crit", "D",
                  "alpha1", "beta1", "gamma1", "alpha2", "beta2", "gamma2",
                  "E_m", "sigma_m", "E_f", "sigma_f", "nu"]
    output_cols = ["sigma_HT", "E_HT"]
    
    # Normalization (StandardScaler equivalent)
    X_data = df[input_cols].values
    Y_data = df[output_cols].values
    
    X_mean = X_data.mean(axis=0)
    X_std = X_data.std(axis=0)
    Y_mean = Y_data.mean(axis=0)
    Y_std = Y_data.std(axis=0)
    
    df[input_cols] = (X_data - X_mean) / X_std
    df[output_cols] = (Y_data - Y_mean) / Y_std
    
    mean_std = {
        'x_mean': X_mean,
        'x_std': X_std,
        'y_mean': Y_mean,
        'y_std': Y_std
    }
    
    # Train/val split
    n_samples = len(df)
    n_train = int(TRAIN_SPLIT * n_samples)
    
    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:]
    
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    
    # Create datasets
    train_dataset = CFRPDataset(train_df, input_cols, output_cols)
    val_dataset = CFRPDataset(val_df, input_cols, output_cols)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model
    model = CFRP_PINN(d_h=32, n_heads=4, n_bands=4)
    
    print(f"\nModel architecture:")
    print(f"  Embedding dimension: 32")
    print(f"  Attention heads: 4")
    print(f"  Spectral bands: 4")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # Train model
    history = train_pinn(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        mean_std=mean_std,
        n_epochs=N_EPOCHS,
        device=DEVICE,
        save_path='cfrp_pinn_best.pt'
    )
    
    # Save training history
    history_df = pd.DataFrame({
        'epoch': range(len(history['train_loss'])),
        'train_loss': history['train_loss'],
        'val_loss': history['val_loss'],
        'lr': history['lr']
    })
    history_df.to_csv('training_history.csv', index=False)
    print("Training history saved to: training_history.csv")


if __name__ == "__main__":
    main()
