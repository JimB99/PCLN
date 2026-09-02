"""Dynamic neurons: rank-1 units with top-k routing (sparse output FLOPs)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicNeuronLayer(nn.Module):
    """Each neuron is a rank-1 map y_i = (x · u_i) v_i, gated by top-k.

    Parameter count is O(N · (in + out)), not O(N · in · out).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_neurons: int = 256,
        top_k_neurons: int = 32,
        use_bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_neurons = num_neurons
        self.top_k_neurons = min(top_k_neurons, num_neurons)

        hidden = max(in_features // 2, 8)
        self.gate = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_neurons),
        )
        self.in_vec = nn.Parameter(torch.empty(num_neurons, in_features))
        self.out_vec = nn.Parameter(torch.empty(num_neurons, out_features))
        self.neuron_bias = nn.Parameter(torch.zeros(num_neurons, out_features)) if use_bias else None
        self.output_proj = nn.Linear(out_features, out_features, bias=use_bias)

        nn.init.xavier_uniform_(self.in_vec)
        nn.init.xavier_uniform_(self.out_vec)

    def forward(self, x: torch.Tensor, return_neuron_usage: bool = False) -> tuple:
        batch, seq_len, _ = x.shape

        activations = torch.einsum("btd,nd->btn", x, self.in_vec)
        gate_logits = self.gate(x)
        gate_scores, neuron_indices = torch.topk(
            gate_logits, k=self.top_k_neurons, dim=-1, sorted=False
        )
        gate_weights = F.softmax(gate_scores, dim=-1)

        selected_act = torch.gather(activations, dim=2, index=neuron_indices)
        selected_out = self.out_vec[neuron_indices]
        contrib = selected_act.unsqueeze(-1) * selected_out
        if self.neuron_bias is not None:
            contrib = contrib + self.neuron_bias[neuron_indices]
        weighted = (contrib * gate_weights.unsqueeze(-1)).sum(dim=2)
        output = self.output_proj(weighted)

        neuron_usage: dict = {}
        if return_neuron_usage or self.training:
            indices_flat = neuron_indices.reshape(-1)
            usage_counts = torch.bincount(indices_flat, minlength=self.num_neurons).float()
            total = max(batch * seq_len * self.top_k_neurons, 1)
            usage_probs = usage_counts / total
            uniform = 1.0 / self.num_neurons
            hard_lb = torch.sum((usage_probs - uniform) ** 2)

            soft = F.softmax(gate_logits, dim=-1).mean(dim=(0, 1))
            load = F.one_hot(neuron_indices, self.num_neurons).float().sum(dim=2).mean(dim=(0, 1))
            switch_lb = self.num_neurons * (soft * load).sum()
            load_balance_loss = hard_lb + 0.01 * switch_lb

            neuron_usage["load_balance_loss"] = load_balance_loss
            neuron_usage["usage_probs"] = usage_probs
            neuron_usage["selected_tokens"] = int((usage_counts > 0).sum().item())
            neuron_usage["total_neurons"] = self.num_neurons
            if return_neuron_usage:
                return output, neuron_usage

        return output, neuron_usage


class DynamicNeuronBlock(nn.Module):
    """PCN refinement using factorized dynamic neurons as the generator."""

    def __init__(
        self,
        d_model: int,
        num_neurons: int = 256,
        top_k_neurons: int = 32,
        K: int = 2,
        alpha: float = 0.1,
        residual_mode: str = "refined",
    ):
        super().__init__()
        self.d_model = d_model
        self.K = K
        self.alpha = alpha
        self.num_neurons = num_neurons
        self.top_k_neurons = top_k_neurons
        self.residual_mode = residual_mode

        self.gen_neurons_1 = DynamicNeuronLayer(
            in_features=d_model,
            out_features=d_model * 2,
            num_neurons=num_neurons,
            top_k_neurons=top_k_neurons,
        )
        self.act = nn.GELU()
        self.gen_neurons_2 = DynamicNeuronLayer(
            in_features=d_model * 2,
            out_features=d_model,
            num_neurons=num_neurons,
            top_k_neurons=top_k_neurons,
        )
        self.damping = nn.Parameter(torch.ones(d_model) * 0.1)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, z: torch.Tensor, training: bool = True) -> tuple:
        z_refined = z
        errors_list = []
        load_balance_losses = []
        k_steps = self.K if training else self.K + 1
        zero = z.new_zeros(())

        for _ in range(k_steps):
            hidden, lb1 = self.gen_neurons_1(z_refined)
            hidden = self.act(hidden)
            z_pred, lb2 = self.gen_neurons_2(hidden)
            load_balance_losses.append(lb1.get("load_balance_loss", zero))
            load_balance_losses.append(lb2.get("load_balance_loss", zero))
            error = z_refined - z_pred
            errors_list.append(error)
            z_refined = z_refined - self.alpha * error * (1.0 + self.damping)

        errors = torch.stack(errors_list, dim=1)
        total_lb = sum(load_balance_losses) / max(len(load_balance_losses), 1)
        out = self.norm(z + z_refined) if self.residual_mode == "sum" else self.norm(z_refined)
        return out, errors, total_lb
