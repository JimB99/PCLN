"""Dynamic neurons with gated activation.

Replaces dense fully connected layers with learnable neuron gates.
Instead of all neurons being active, we learn which neurons to activate per input.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicNeuronLayer(nn.Module):
    """Gated neuron layer with dynamic activation.
    
    Instead of: y = dense(x)
    We do:
    1. Compute gate scores g = softmax(W_gate @ x), shape (batch, seq, num_neurons)
    2. Select top-k neurons
    3. Apply selected neurons: y = weighted_sum(neurons[selected])
    4. Output via residual/projection
    
    This implements:
    - Learned neuron specialization (each neuron learns task-specific patterns)
    - Sparse activation (top-k neurons active per token)
    - Load balancing (auxiliary loss prevents all tokens from using same neurons)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_neurons: int = 256,
        top_k_neurons: int = 32,
        use_bias: bool = True,
    ):
        """Initialize dynamic neuron layer.
        
        Args:
            in_features: input dimension.
            out_features: output dimension.
            num_neurons: total number of neurons available.
            top_k_neurons: how many neurons to activate per token.
            use_bias: whether to use bias terms.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_neurons = num_neurons
        self.top_k_neurons = min(top_k_neurons, num_neurons)
        
        # Gate network: determines which neurons to activate
        # Input -> hidden -> neuron scores
        self.gate = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(),
            nn.Linear(in_features // 2, num_neurons),  # Output: score per neuron
        )
        
        # Neuron embeddings: each neuron is a learned transformation
        # Shape: (num_neurons, in_features, hidden_dim)
        # or simpler: (num_neurons, in_features) -> use one per neuron
        self.neurons = nn.Linear(in_features, num_neurons * out_features)
        self.neurons_out_project = nn.Linear(num_neurons, 1)  # Combine neuron outputs
        
        # Auxiliary projection to combine selected neuron outputs
        self.output_proj = nn.Linear(out_features, out_features, bias=use_bias)
        
        # Initialize
        nn.init.xavier_uniform_(self.neurons.weight)
        
    def forward(self, x: torch.Tensor, return_neuron_usage: bool = False) -> tuple:
        """Forward pass with dynamic neuron selection.
        
        Args:
            x: (batch, seq_len, in_features) input.
            return_neuron_usage: if True, return usage statistics.
            
        Returns:
            output: (batch, seq_len, out_features) output.
            neuron_usage: dict with 'selected_ratio', 'load_balance_loss' if requested.
        """
        batch, seq_len, _ = x.shape
        device = x.device
        
        # Step 1: Compute neuron gates
        # Input: (batch, seq_len, in_features)
        # Output: (batch, seq_len, num_neurons)
        gate_logits = self.gate(x)  # (batch, seq_len, num_neurons)
        
        # Step 2: Sparse selection - top-k neurons per token
        gate_scores, neuron_indices = torch.topk(
            gate_logits, 
            k=self.top_k_neurons, 
            dim=-1,  # Select along neuron dimension
            sorted=False,
        )
        # gate_scores: (batch, seq_len, top_k_neurons)
        # neuron_indices: (batch, seq_len, top_k_neurons)
        
        # Normalize gate scores with softmax over selected neurons
        gate_weights = F.softmax(gate_scores, dim=-1)  # (batch, seq_len, top_k_neurons)
        
        # Step 3: Compute neuron outputs for all neurons
        neuron_outputs = self.neurons(x)  # (batch, seq_len, num_neurons * out_features)
        
        # Reshape to separate neurons: (batch, seq_len, num_neurons, out_features)
        neuron_outputs = neuron_outputs.reshape(
            batch, seq_len, self.num_neurons, self.out_features
        )
        
        # Step 4: Select and activate neurons
        # Gather selected neuron outputs
        # neuron_indices: (batch, seq_len, top_k_neurons)
        # neuron_outputs: (batch, seq_len, num_neurons, out_features)
        
        # Expand indices for gathering: (batch, seq_len, top_k_neurons, out_features)
        expanded_indices = neuron_indices.unsqueeze(-1).expand(
            batch, seq_len, self.top_k_neurons, self.out_features
        )
        selected_outputs = torch.gather(
            neuron_outputs,
            dim=2,  # Gather along neuron dimension (after expanding)
            index=expanded_indices,
        )
        # selected_outputs: (batch, seq_len, top_k_neurons, out_features)
        
        # Step 5: Weighted sum over selected neurons
        # Weights: (batch, seq_len, top_k_neurons, 1)
        weighted_sum = (selected_outputs * gate_weights.unsqueeze(-1)).sum(dim=2)
        # weighted_sum: (batch, seq_len, out_features)
        
        # Step 6: Project output
        output = self.output_proj(weighted_sum)  # (batch, seq_len, out_features)
        
        # Compute auxiliary loss (load balancing)
        neuron_usage = {}
        if return_neuron_usage or self.training:
            # Count how many times each neuron is selected in the batch
            # neuron_indices: (batch, seq_len, top_k_neurons)
            indices_flat = neuron_indices.reshape(-1)  # (batch * seq_len * top_k_neurons,)
            usage_counts = torch.bincount(
                indices_flat, 
                minlength=self.num_neurons
            ).float()  # (num_neurons,)
            
            # Normalize by total activations
            total_activations = batch * seq_len * self.top_k_neurons
            usage_probs = usage_counts / total_activations
            
            # Load balance loss: penalize divergence from uniform distribution (1/num_neurons)
            uniform_prob = 1.0 / self.num_neurons
            load_balance_loss = torch.sum((usage_probs - uniform_prob) ** 2)
            
            neuron_usage['load_balance_loss'] = load_balance_loss
            neuron_usage['usage_probs'] = usage_probs
            neuron_usage['selected_tokens'] = (usage_counts > 0).sum().item()
            neuron_usage['total_neurons'] = self.num_neurons
            
            if return_neuron_usage:
                return output, neuron_usage
        
        return output, neuron_usage


class DynamicNeuronBlock(nn.Module):
    """PCN refinement layer using dynamic neurons.
    
    Replaces dense layers with dynamic neuron activation.
    """
    
    def __init__(
        self,
        d_model: int,
        num_neurons: int = 256,
        top_k_neurons: int = 32,
        K: int = 2,
        alpha: float = 0.1,
    ):
        """Initialize dynamic neuron PCN block.
        
        Args:
            d_model: latent dimension.
            num_neurons: total neurons available.
            top_k_neurons: neurons activated per token.
            K: refinement steps.
            alpha: refinement step size.
        """
        super().__init__()
        self.d_model = d_model
        self.K = K
        self.alpha = alpha
        self.num_neurons = num_neurons
        self.top_k_neurons = top_k_neurons
        
        # Generative model with dynamic neurons
        # First layer: d_model -> hidden
        self.gen_neurons_1 = DynamicNeuronLayer(
            in_features=d_model,
            out_features=d_model * 2,
            num_neurons=num_neurons,
            top_k_neurons=top_k_neurons,
        )
        self.relu = nn.ReLU()
        
        # Second layer: hidden -> d_model
        self.gen_neurons_2 = DynamicNeuronLayer(
            in_features=d_model * 2,
            out_features=d_model,
            num_neurons=num_neurons,
            top_k_neurons=top_k_neurons,
        )
        
        # Learned damping
        self.damping = nn.Parameter(torch.ones(d_model) * 0.1)
        
    def forward(self, z: torch.Tensor, training: bool = True) -> tuple:
        """Refine with dynamic neurons.
        
        Args:
            z: (batch, seq_len, d_model) latent states.
            training: training mode flag.
            
        Returns:
            z_refined: (batch, seq_len, d_model) refined beliefs.
            errors: prediction errors for auxiliary loss.
            load_balance_loss: auxiliary loss from neuron selection.
        """
        batch, seq_len, _ = z.shape
        z_refined = z.clone()
        errors_list = []
        load_balance_losses = []
        
        K = self.K if training else self.K + 1
        
        for step in range(K):
            # Generative prediction with dynamic neurons
            hidden, lb_loss_1 = self.gen_neurons_1(z_refined)
            hidden = self.relu(hidden)
            z_pred, lb_loss_2 = self.gen_neurons_2(hidden)
            
            load_balance_losses.append(lb_loss_1.get('load_balance_loss', torch.tensor(0.0)))
            load_balance_losses.append(lb_loss_2.get('load_balance_loss', torch.tensor(0.0)))
            
            # Prediction error
            error = z_refined - z_pred
            errors_list.append(error)
            
            # Update with damping
            z_refined = z_refined - self.alpha * error * (1.0 + self.damping)
        
        errors = torch.stack(errors_list, dim=1)  # (batch, K, seq_len, d_model)
        total_lb_loss = sum(load_balance_losses) / len(load_balance_losses) if load_balance_losses else torch.tensor(0.0, device=z.device)
        
        return z_refined, errors, total_lb_loss
