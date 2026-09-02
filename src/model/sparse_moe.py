"""Sparse Mixture-of-Experts PCN blocks (vectorized over sequence length)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertGate(nn.Module):
    """Top-k expert router with a load-balance auxiliary loss."""

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
        self.gate = nn.Linear(d_model, num_experts)
        self.gate.bias.data.zero_()

    def forward(self, x: torch.Tensor, training: bool = True) -> tuple:
        batch_size, seq_len, d_model = x.shape
        x_flat = x.reshape(batch_size * seq_len, d_model)
        logits = self.gate(x_flat)
        gate_scores, expert_indices = torch.topk(
            logits, k=self.top_k, dim=1, largest=True, sorted=False
        )
        gate_scores = F.softmax(gate_scores, dim=1)

        load_balance_loss = torch.zeros((), device=x.device, dtype=x.dtype)
        if training:
            expert_usage = torch.zeros(self.num_experts, device=x.device, dtype=x.dtype)
            ones = torch.ones(expert_indices.numel(), device=x.device, dtype=x.dtype)
            expert_usage.scatter_add_(0, expert_indices.reshape(-1), ones)
            expert_usage = expert_usage / (batch_size * seq_len)
            ideal = 1.0 / self.num_experts
            load_balance_loss = torch.sum((expert_usage - ideal) ** 2)

        return gate_scores, expert_indices, load_balance_loss


class SparseMoEPCNBlock(nn.Module):
    """Tokens are routed to top-k experts; loop is over experts, not time."""

    def __init__(
        self,
        d_model: int,
        num_experts: int = 4,
        top_k: int = 2,
        K_refine: int = 2,
        alpha: float = 0.1,
        residual_mode: str = "refined",
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.K_refine = K_refine
        self.alpha = alpha
        self.residual_mode = residual_mode

        self.gate = ExpertGate(d_model, num_experts, top_k)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, d_model * 2),
                    nn.GELU(),
                    nn.Linear(d_model * 2, d_model),
                )
                for _ in range(num_experts)
            ]
        )
        self.dampings = nn.ParameterList(
            [nn.Parameter(torch.ones(d_model) * 0.1) for _ in range(num_experts)]
        )
        self.register_buffer("expert_usage_count", torch.zeros(num_experts))
        self.norm = nn.LayerNorm(d_model)

    def _mix_experts(self, x_flat: torch.Tensor, gate_scores, expert_indices, training: bool):
        expert_preds = torch.zeros_like(x_flat)
        counts = torch.zeros(self.num_experts, device=x_flat.device, dtype=self.expert_usage_count.dtype)
        for expert_id, expert in enumerate(self.experts):
            membership = expert_indices == expert_id
            if not membership.any():
                continue
            weights = (gate_scores * membership.to(gate_scores.dtype)).sum(dim=-1)
            token_mask = weights > 0
            if not token_mask.any():
                continue
            pred = expert(x_flat[token_mask])
            expert_preds[token_mask] = expert_preds[token_mask] + weights[token_mask].unsqueeze(-1) * pred
            counts[expert_id] = token_mask.sum()
        if training:
            self.expert_usage_count += counts.detach()
        return expert_preds

    def forward(
        self,
        z: torch.Tensor,
        training: bool = True,
        prune_threshold: float = 0.01,
    ) -> tuple:
        del prune_threshold
        batch_size, seq_len, d_model = z.shape
        gate_scores, expert_indices, load_balance_loss = self.gate(z, training=training)

        z_refined = z
        errors_list = []
        damp_table = torch.stack(list(self.dampings), dim=0)
        selected_damp = (damp_table[expert_indices] * gate_scores.unsqueeze(-1)).sum(dim=1)

        for _ in range(self.K_refine):
            x_flat = z_refined.reshape(batch_size * seq_len, d_model)
            expert_preds = self._mix_experts(x_flat, gate_scores, expert_indices, training)
            error = x_flat - expert_preds
            errors_list.append(error.view(batch_size, seq_len, d_model))
            x_flat = x_flat - self.alpha * error * (1.0 + selected_damp)
            z_refined = x_flat.view(batch_size, seq_len, d_model)

        errors = torch.stack(errors_list, dim=1)
        z_out = self.norm(z + z_refined) if self.residual_mode == "sum" else self.norm(z_refined)
        usage_sum = self.expert_usage_count.sum() + 1e-8
        expert_usage = self.expert_usage_count / usage_sum
        return z_out, errors, load_balance_loss, expert_usage

    def prune_unused_experts(self, threshold: float = 0.01):
        expert_usage = self.expert_usage_count / (self.expert_usage_count.sum() + 1e-8)
        unused = (expert_usage < threshold).nonzero(as_tuple=True)[0].tolist()
        return unused
