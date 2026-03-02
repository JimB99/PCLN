"""Predictive Coding refinement layers for PCLN.

Implements iterative belief refinement via prediction error minimization.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PCNRefinementLayer(nn.Module):
    """Single PCN refining layer.
    
    Performs K steps of refinement: z <- z - alpha * (z - f(z))
    where f(z) is a learned generative model.
    
    Args:
        d_model: dimension of latent beliefs.
        K: number of refinement steps per forward pass.
        alpha: refinement learning rate.
    """

    def __init__(self, d_model: int, K: int = 2, alpha: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.K = K
        self.alpha = alpha

        # Generative model (predicts z from z)
        self.generator = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
        )

        # Optional: learned damping factor per dimension
        self.damping = nn.Parameter(torch.ones(d_model) * 0.1)

    def forward(self, z, training: bool = True):
        """Refine latent beliefs.
        
        Args:
            z: (batch, seq_len, d_model) latent states.
            training: if True, use K_train steps; else use K_eval steps.
        Returns:
            refined_z: (batch, seq_len, d_model) refined beliefs.
            errors: (batch, seq_len, d_model) prediction errors (for auxiliary loss).
        """
        batch_size, seq_len, _ = z.shape
        z_refined = z.clone()
        errors_list = []

        # Adaptive K for training vs evaluation
        K = self.K if training else self.K + 1

        for step in range(K):
            # Generative prediction
            z_pred = self.generator(z_refined)
            
            # Prediction error
            error = z_refined - z_pred
            errors_list.append(error)
            
            # Update with learned damping
            z_refined = z_refined - self.alpha * error * (1.0 + self.damping)

        # Stack errors for loss computation
        errors = torch.stack(errors_list, dim=1)  # (batch, K, seq_len, d_model)
        return z_refined, errors


class PCNBlock(nn.Module):
    """Complete PCN block with residual connection.
    
    Combines a refinement layer with residual connection to stabilize training.
    
    Args:
        d_model: latent dimension.
        K: number of refinement steps.
        alpha: refinement step size.
    """

    def __init__(self, d_model: int, K: int = 2, alpha: float = 0.1):
        super().__init__()
        self.refine_layer = PCNRefinementLayer(d_model, K, alpha)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, z, training: bool = True):
        """Apply PCN refinement with residual connection.
        
        Args:
            z: (batch, seq_len, d_model) latent beliefs.
            training: training mode flag.
        Returns:
            z_out: (batch, seq_len, d_model) refined beliefs.
            errors: (batch, K, seq_len, d_model) prediction errors.
        """
        z_refined, errors = self.refine_layer(z, training=training)
        
        # Residual connection + layer norm
        z_out = self.norm(z + z_refined)
        return z_out, errors
