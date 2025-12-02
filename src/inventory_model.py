# model.py
"""
Per-item DeepSets Actor–Critic for restaurant inventory RL (variable-size item sets).

This version produces **per-item actions** via a shared decoder head:
    dec([φ(z_i), g_env]) -> (μ_i, log sig_i)   for each item i

- φ(z_i): per-item embedding
- g_env:  pooled (permutation-invariant) global context for that env
- Actions are emitted for ALL items present in the batch; sizes can vary by env.
- Log-probabilities are summed per env for PPO-style updates.

References:
  - DeepSets invariant pooling (sum/mean/max):
    Zaheer et al., 2017 "Deep Sets"  https://arxiv.org/abs/1703.06114
    (good summary: https://www.scibits.blog/posts/deepsets/)
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from torch_geometric.nn import aggr


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------

def mlp(in_dim: int, hidden_dim: int, out_dim: int, n_layers: int = 2, dropout: float = 0.0) -> nn.Sequential:
    layers = []
    d = in_dim
    for _ in range(max(0, n_layers - 1)):
        layers += [nn.Linear(d, hidden_dim), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        d = hidden_dim
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


# ----------------------------------------------------------------------
# Encoders and Aggregators
# ----------------------------------------------------------------------

class ItemEncoder(nn.Module):
    """
    Per-item encoder φ(z_i): maps raw item features to an embedding.

    Inspired by DeepSets encoders (Zaheer et al., 2017; SciBits blog).
    """
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [num_items, input_dim]
        return self.net(x)  # [num_items, hidden_dim]


class DeepSetsAggregator(nn.Module):
    """
    Permutation-invariant env-level pooling over item embeddings using PyG aggregators.

    Fixed aggregations:
      - Sum / Mean / Max  (DeepSets: Zaheer et al., 2017)
    """
    def __init__(
        self,
        hidden_dim: int,
        use_sum: bool = True,
        use_mean: bool = True,
        use_max: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_sum, self.use_mean, self.use_max = use_sum, use_mean, use_max

        if use_sum:
            self.sum_aggr = aggr.SumAggregation()
        if use_mean:
            self.mean_aggr = aggr.MeanAggregation()
        if use_max:
            self.max_aggr = aggr.MaxAggregation()

        out_dim = 0
        if use_sum:
            out_dim += hidden_dim
        if use_mean:
            out_dim += hidden_dim
        if use_max:
            out_dim += hidden_dim
        self._out_dim = out_dim

    @property
    def out_dim(self) -> int:
        return self._out_dim

    def forward(self, x: torch.Tensor, batch: torch.Tensor, batch_size: Optional[int] = None) -> torch.Tensor:
        if batch_size is None:
            batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 0

        outs = []
        if self.use_sum:
            outs.append(self.sum_aggr(x, index=batch, dim=0, dim_size=batch_size))
        if self.use_mean:
            outs.append(self.mean_aggr(x, index=batch, dim=0, dim_size=batch_size))
        if self.use_max:
            outs.append(self.max_aggr(x, index=batch, dim=0, dim_size=batch_size))

        return torch.cat(outs, dim=-1)  # [batch_size, out_dim]


# ----------------------------------------------------------------------
# Per-item Policy Head + Value Head
# ----------------------------------------------------------------------

class GaussianPolicyHead(nn.Module):
    """
    Shared per-item decoder head:
      inputs:  concat([item_embed_i, env_context_for_item_i])  -> [μ_i, logσ_i]

    - Outputs parameters for a factorised Gaussian action **per item**.
    - Supports variable-size item sets via 'item_batch' mapping.

    NOTE: projection to feasibility (budget/capacity/MOQ) should be handled
    in env.py or a post-policy wrapper; the Gaussian lives in a latent space.

    log_std is clamped to keep exploration sane.
    """
    def __init__(
        self,
        item_embed_dim: int,
        env_ctx_dim: int,
        per_item_action_dim: int = 1,
        hidden_dim: int = 128,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        in_dim = item_embed_dim + env_ctx_dim
        self.core = mlp(in_dim, hidden_dim, hidden_dim, n_layers=2, dropout=0.0)
        self.mu = nn.Linear(hidden_dim, per_item_action_dim)
        self.log_std = nn.Linear(hidden_dim, per_item_action_dim)
        self.lmin = log_std_min
        self.lmax = log_std_max

    def forward(self, item_embed: torch.Tensor, env_ctx_for_items: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            item_embed:        [num_items, item_embed_dim]
            env_ctx_for_items: [num_items, env_ctx_dim]  (context cloned by item_batch)

        Returns:
            mu_items:      [num_items, per_item_action_dim]
            log_std_items: [num_items, per_item_action_dim]
        """
        x = torch.cat([item_embed, env_ctx_for_items], dim=-1)
        h = F.relu(self.core(x))
        mu = self.mu(h)
        log_std = self.log_std(h).clamp(self.lmin, self.lmax)
        return mu, log_std


class ValueHead(nn.Module):
    """
    State-value head V(s) over env-level pooled context (+ global features).
    """
    def __init__(self, in_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.mlp = mlp(in_dim, hidden_dim, hidden_dim, n_layers=2, dropout=0.0)
        self.v = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.mlp(x))
        return self.v(h)  # [batch_size, 1]


# ----------------------------------------------------------------------
# Config + Actor-Critic wrapper
# ----------------------------------------------------------------------

@dataclass
class InventoryModelConfig:
    # Input dimensions
    item_input_dim: int
    global_input_dim: int

    # Per-item action dimension (e.g., 1 = raw-ingredient order qty; extend as needed)
    per_item_action_dim: int = 1

    # Item encoder
    item_hidden_dim: int = 64
    item_dropout: float = 0.0

    # Aggregation options
    use_sum: bool = True
    use_mean: bool = True
    use_max: bool = True

    # Heads
    actor_hidden_dim: int = 128
    critic_hidden_dim: int = 128
    log_std_min: float = -5.0
    log_std_max: float = 2.0


class InventoryActorCritic(nn.Module):
    """
    DeepSets per-item Actor–Critic:

      φ:  shared per-item encoder
      AGG: env-level pooling (sum/mean/max)
      dec: shared per-item Gaussian head   (uses [φ(z_i), g_env] per item)
      V:   env-level value head on pooled context

    Expected obs from train.py/env.py:
        obs = {
            "item_features":   FloatTensor[num_items_total, item_input_dim],
            "item_batch":      LongTensor[num_items_total],  # env index per item
            "global_features": FloatTensor[batch_size, global_input_dim],
        }
      where max(item_batch)+1 == batch_size.

    Forward returns:
        {
          "mu_items":      [num_items_total, per_item_action_dim],
          "log_std_items": [num_items_total, per_item_action_dim],
          "value":         [batch_size],
        }

    Convenience:
      - act(): samples per-item actions and returns per-env log-probs (sum over items in env)
      - evaluate_actions(): given per-item actions, returns per-env log-probs/entropy and values
    """
    def __init__(self, config: InventoryModelConfig):
        super().__init__()
        self.cfg = config

        # Shared per-item encoder φ
        self.item_encoder = ItemEncoder(
            input_dim=config.item_input_dim,
            hidden_dim=config.item_hidden_dim,
            dropout=config.item_dropout,
        )

        # Env-level aggregators
        self.actor_agg = DeepSetsAggregator(
            hidden_dim=config.item_hidden_dim,
            use_sum=config.use_sum,
            use_mean=config.use_mean,
            use_max=config.use_max,
        )
        self.critic_agg = DeepSetsAggregator(
            hidden_dim=config.item_hidden_dim,
            use_sum=config.use_sum,
            use_mean=config.use_mean,
            use_max=config.use_max,
        )

        # We append "set size" scalar N and global features to env context for both heads
        self.env_ctx_dim_actor = self.actor_agg.out_dim + 1 + config.global_input_dim
        self.env_ctx_dim_critic = self.critic_agg.out_dim + 1 + config.global_input_dim

        # Per-item Gaussian policy head (shared across items)
        self.item_policy = GaussianPolicyHead(
            item_embed_dim=config.item_hidden_dim,
            env_ctx_dim=self.env_ctx_dim_actor,
            per_item_action_dim=config.per_item_action_dim,
            hidden_dim=config.actor_hidden_dim,
            log_std_min=config.log_std_min,
            log_std_max=config.log_std_max,
        )

        # Env-value head
        self.value_head = ValueHead(
            in_dim=self.env_ctx_dim_critic,
            hidden_dim=config.critic_hidden_dim,
        )

        # For per-env reductions (sum log-probs over items)
        self.sum_aggr = aggr.SumAggregation()

    # ---------- helpers ----------

    def _env_contexts(
        self,
        item_features: torch.Tensor,
        item_batch: torch.Tensor,
        global_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Build env-level contexts and item-level replicated contexts.

        Returns:
            {
              "item_embed":        [num_items, item_hidden_dim],
              "env_ctx_actor":     [batch_size, env_ctx_dim_actor],
              "env_ctx_critic":    [batch_size, env_ctx_dim_critic],
              "env_ctx_for_items": [num_items, env_ctx_dim_actor],  # gathered by item_batch
            }
        """
        batch_size = global_features.size(0)
        device = global_features.device

        # Encode items
        item_embed = self.item_encoder(item_features)  # [num_items, item_hidden_dim]

        # Set size per env (N)
        set_sizes = torch.bincount(item_batch, minlength=batch_size).float().unsqueeze(-1).to(device)  # [B,1]

        # Pool per env
        pooled_actor = self.actor_agg(item_embed, batch=item_batch, batch_size=batch_size)   # [B, A_out]
        pooled_critic = self.critic_agg(item_embed, batch=item_batch, batch_size=batch_size) # [B, C_out]

        # Env contexts (concat global + pooled + set_size)
        env_ctx_actor = torch.cat([global_features, pooled_actor, set_sizes], dim=-1)   # [B, env_ctx_dim_actor]
        env_ctx_critic = torch.cat([global_features, pooled_critic, set_sizes], dim=-1) # [B, env_ctx_dim_critic]

        # Replicate actor env context per item (index with item_batch)
        env_ctx_for_items = env_ctx_actor[item_batch]  # [num_items, env_ctx_dim_actor]

        return {
            "item_embed": item_embed,
            "env_ctx_actor": env_ctx_actor,
            "env_ctx_critic": env_ctx_critic,
            "env_ctx_for_items": env_ctx_for_items,
        }

    # ---------- forward ----------

    def forward(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        item_features = obs["item_features"]          # [num_items, item_input_dim]
        item_batch = obs["item_batch"].long()         # [num_items]
        global_features = obs["global_features"]      # [batch_size, global_input_dim]

        ctx = self._env_contexts(item_features, item_batch, global_features)

        # Per-item Gaussian params
        mu_items, log_std_items = self.item_policy(ctx["item_embed"], ctx["env_ctx_for_items"])  # [num_items, A_d]

        # Env value
        value = self.value_head(ctx["env_ctx_critic"]).squeeze(-1)  # [batch_size]

        return {
            "mu_items": mu_items,
            "log_std_items": log_std_items,
            "value": value,
        }

    # ---------- API for train.py ----------

    @torch.no_grad()
    def act(self, obs: Dict[str, torch.Tensor], deterministic: bool = False) -> Dict[str, Any]:
        """
        Sample per-item actions and compute per-env aggregates.

        Returns:
            {
              "item_actions":   [num_items, per_item_action_dim],
              "item_log_probs": [num_items],                       # per-item summed over action dims
              "env_log_probs":  [batch_size],                      # sum over items in each env
              "mu_items":       [num_items, A_d],
              "log_std_items":  [num_items, A_d],
              "value":          [batch_size],
            }
        """
        out = self.forward(obs)
        mu_items, log_std_items, value = out["mu_items"], out["log_std_items"], out["value"]
        std_items = log_std_items.exp()

        dist = Normal(mu_items, std_items)  # factorised over action dims
        if deterministic:
            actions = mu_items
        else:
            actions = dist.rsample()
        # Sum log-prob over action dims to get per-item log-prob
        item_log_probs = dist.log_prob(actions).sum(dim=-1)  # [num_items]

        # Reduce per env
        item_batch = obs["item_batch"].long()
        batch_size = obs["global_features"].size(0)
        env_log_probs = self.sum_aggr(item_log_probs, index=item_batch, dim=0, dim_size=batch_size)  # [B]

        return {
            "item_actions": actions,
            "item_log_probs": item_log_probs,
            "env_log_probs": env_log_probs,
            "mu_items": mu_items,
            "log_std_items": log_std_items,
            "value": value,
        }

    def evaluate_actions(
        self,
        obs: Dict[str, torch.Tensor],
        item_actions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Evaluate log-probs/entropy for **given per-item actions** (PPO update).

        Args:
            obs:
                - "item_features":   [num_items, item_input_dim]
                - "item_batch":      [num_items]
                - "global_features": [batch_size, global_input_dim]
            item_actions: [num_items, per_item_action_dim]

        Returns:
            {
              "item_log_probs": [num_items],
              "env_log_probs":  [batch_size],
              "item_entropy":   [num_items],   # summed over action dims
              "env_entropy":    [batch_size],  # sum over items
              "value":          [batch_size],
            }
        """
        out = self.forward(obs)
        mu_items, log_std_items, value = out["mu_items"], out["log_std_items"], out["value"]
        std_items = log_std_items.exp()

        dist = Normal(mu_items, std_items)
        item_log_probs = dist.log_prob(item_actions).sum(dim=-1)  # [num_items]
        item_entropy = dist.entropy().sum(dim=-1)                 # [num_items]

        # Reduce to env level
        item_batch = obs["item_batch"].long()
        batch_size = obs["global_features"].size(0)
        env_log_probs = self.sum_aggr(item_log_probs, index=item_batch, dim=0, dim_size=batch_size)
        env_entropy   = self.sum_aggr(item_entropy,   index=item_batch, dim=0, dim_size=batch_size)

        return {
            "item_log_probs": item_log_probs,
            "env_log_probs": env_log_probs,
            "item_entropy": item_entropy,
            "env_entropy": env_entropy,
            "value": value,
        }
