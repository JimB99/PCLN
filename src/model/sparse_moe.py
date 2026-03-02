"""Sparse Mixture-of-Experts (MoE) PCN blocks with structural plasticity.

Implements:
- Gated expert routing (learns which experts to activate per token)
- Sparse activation (only top-k experts per token)
- Expert pruning (remove inactive experts)
- Load balancing auxiliary loss
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertGate(nn.Module):
    """Gated expert router: learns to select which experts to activate.
    
    Args:
        d_model: latent dimension.
        num_experts: number of expert modules.
        top_k: number of experts to activate per token.
        use_expert_choice: if True, allow soft expert selection; else use hard top-k.
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        top_k: int = 2,
        use_expert_choice: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.use_expert_choice = use_expert_choice

        # Gate produces logits for expert selection
        self.gate = nn.Linear(d_model, num_experts)
        self.gate.bias.data.zero_()

    def forward(self, x: torch.Tensor, training: bool = True) -> tuple:
        """Route tokens to top-k experts.
        
        Args:
            x: (batch, seq_len, d_model) input.
            training: if True, compute load balancing loss.
        Returns:
            (gate_scores, expert_indices, load_balance_loss)
            - gate_scores: (batch*seq_len, top_k) scores for selected experts.
            - expert_indices: (batch*seq_len, top_k) indices of selected experts.
            - load_balance_loss: scalar, auxiliary loss for load balancing.
        """
        batch_size, seq_len, d_model = x.shape

        # Reshape for gating
        x_flat = x.reshape(batch_size * seq_len, d_model)  # (batch*seq_len, d_model)

        # Compute gate logits
        logits = self.gate(x_flat)  # (batch*seq_len, num_experts)

        # Select top-k experts (hard selection)
        gate_scores, expert_indices = torch.topk(
            logits, k=self.top_k, dim=1, largest=True, sorted=False
        )  # (batch*seq_len, top_k), (batch*seq_len, top_k)

        # Apply softmax to gate scores
        gate_scores = F.softmax(gate_scores, dim=1)  # (batch*seq_len, top_k)

        # Compute auxiliary loss: encourage balanced expert usage
        load_balance_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        if training:
            # Fraction of tokens routed to each expert
            expert_usage = torch.zeros(self.num_experts, device=x.device, dtype=x.dtype)
            ones = torch.ones(expert_indices.reshape(-1).shape[0], device=x.device, dtype=x.dtype)
            expert_usage.scatter_add_(0, expert_indices.reshape(-1), ones)
            expert_usage = expert_usage / (batch_size * seq_len)

            # Ideal: each expert gets 1/num_experts of tokens
            ideal = 1.0 / self.num_experts

            # Load balance loss: L2 distance from ideal distribution
            load_balance_loss = torch.sum((expert_usage - ideal) ** 2)

        return gate_scores, expert_indices, load_balance_loss


class SparseMoEPCNBlock(nn.Module):
    """Sparse Mixture-of-Experts PCN block.
    
    Each token is routed to top-k experts for refinement. Only active experts
    perform computation, reducing total FLOPs while maintaining capacity.
    
    Args:
        d_model: latent dimension.
        num_experts: number of expert modules.
        top_k: number of experts to activate per token.
        K_refine: refinement steps per expert.
        alpha: refinement step size.
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int = 4,
        top_k: int = 2,
        K_refine: int = 2,
        alpha: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.K_refine = K_refine
        self.alpha = alpha

        # Gate router
        self.gate = ExpertGate(d_model, num_experts, top_k)

        # Expert modules (each is a PCN refinement layer)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.ReLU(),
                nn.Linear(d_model * 2, d_model),
            )
            for _ in range(num_experts)
        ])

        # Damping factors (learned per-expert)
        self.dampings = nn.ParameterList([
            nn.Parameter(torch.ones(d_model) * 0.1)
            for _ in range(num_experts)
        ])

        # Expert usage tracking (for pruning)
        self.register_buffer("expert_usage_count", torch.zeros(num_experts))

        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        z: torch.Tensor,
        training: bool = True,
        prune_threshold: float = 0.01,
    ) -> tuple:
        """Sparse MoE refinement.
        
        Args:
            z: (batch, seq_len, d_model) latent beliefs.
            training: if True, compute auxiliary loss and track expert usage.
            prune_threshold: if expert usage < threshold, mark for pruning.
        Returns:
            (z_refined, errors, load_balance_loss, expert_usage)
        """
        batch_size, seq_len, d_model = z.shape

        # Gate: route tokens to experts
        gate_scores, expert_indices, load_balance_loss = self.gate(
            z, training=training
        )  # (batch*seq_len, top_k), (batch*seq_len, top_k)

        # Initialize output
        z_refined = torch.zeros_like(z)
        errors_list = []

        # Process each token through its selected experts
        for seq_idx in range(seq_len):
            z_seq = z[:, seq_idx, :]  # (batch, d_model)
            z_seq_refined = z_seq.clone()

            token_errors = []

            # Refinement steps
            for step in range(self.K_refine):
                # Route this token to top-k experts
                flat_idx = torch.arange(batch_size, device=z.device) * seq_len + seq_idx
                local_gate_scores = gate_scores[flat_idx]  # (batch, top_k)
                local_expert_indices = expert_indices[flat_idx]  # (batch, top_k)

                # Aggregate expert predictions
                expert_preds = torch.zeros_like(z_seq_refined)

                for expert_id_idx in range(self.top_k):
                    expert_ids = local_expert_indices[:, expert_id_idx]  # (batch,)
                    scores = local_gate_scores[:, expert_id_idx]  # (batch,)

                    # Get unique expert IDs in this batch
                    unique_experts = torch.unique(expert_ids)

                    for exp_id in unique_experts:
                        mask = expert_ids == exp_id  # (batch,)

                        if mask.any():
                            # Run expert on selected tokens
                            z_selected = z_seq_refined[mask]  # (num_selected, d_model)
                            exp_pred = self.experts[exp_id](z_selected)  # (num_selected, d_model)

                            # Weighted accumulation
                            expert_preds[mask] += scores[mask].unsqueeze(1) * exp_pred

                            # Track expert usage
                            if training:
                                self.expert_usage_count[exp_id] += mask.sum().item()

                # Prediction error
                error = z_seq_refined - expert_preds
                token_errors.append(error)

                # Update with learned damping (use first expert's damping for simplicity)
                first_expert = local_expert_indices[:, 0]
                damping = torch.stack([self.dampings[int(eid)] for eid in first_expert])
                z_seq_refined = z_seq_refined - self.alpha * error * (1.0 + damping)

            z_refined[:, seq_idx, :] = z_seq_refined

            if token_errors:
                errors_list.append(torch.stack(token_errors, dim=1))  # (batch, K, d_model)

        # Stack errors across sequence
        if errors_list:
            errors = torch.stack(errors_list, dim=1)  # (batch, seq_len, K, d_model)
        else:
            errors = torch.zeros(batch_size, seq_len, self.K_refine, d_model, device=z.device)

        # Residual connection + norm
        z_out = self.norm(z + z_refined)

        # Expert usage info
        expert_usage = self.expert_usage_count / (self.expert_usage_count.sum() + 1e-8)

        return z_out, errors, load_balance_loss, expert_usage

    def prune_unused_experts(self, threshold: float = 0.01):
        """Mark and report unused experts (for monitoring/removal).
        
        Args:
            threshold: if expert_usage < threshold, mark as unused.
        Returns:
            list of unused expert indices.
        """
        expert_usage = self.expert_usage_count / (self.expert_usage_count.sum() + 1e-8)
        unused = (expert_usage < threshold).nonzero(as_tuple=True)[0].tolist()
        return unused
