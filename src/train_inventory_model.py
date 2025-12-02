
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
import time
import random
from typing import Dict, List, Tuple

from env import KitchenEnv
from inventory_model import InventoryActorCritic, InventoryModelConfig
from dataloader import load_historical_demands

class ExpertPolicy:
    """
    Expert policy using demand forecasting + safety stock heuristic.
    
    For dishes: produce = forecast_demand + safety_stock
    For raw ingredients: order = recipe_requirements(expected_dish_production) + safety_stock
    """
    def __init__(self, env: KitchenEnv, safety_factor: float = 1.5):
        self.env = env
        self.safety_factor = safety_factor
        
        # Pre-compute recipe requirements (B matrix transpose)
        self.B = env.B  # [n_raw, n_prep] - recipe matrix
        
    def get_action(self, obs: Dict) -> np.ndarray:
        """
        Compute expert action based on demand forecast + safety stock.
        
        Returns:
            action: [n_raw + n_prep] array of order/production quantities
        """
        # Extract forecasts if available, otherwise use base demand
        if 'forecast_mu' in obs and obs['forecast_mu'] is not None:
            # Use ensemble mean prediction
            forecast_mu = obs['forecast_mu']  # [n_models, n_prep]
            forecast_sigma = obs['forecast_sigma']  # [n_models, n_prep]
            
            # Aggregate across models (mean)
            expected_demand = forecast_mu.mean(axis=0)  # [n_prep]
            demand_std = np.sqrt((forecast_sigma**2).mean(axis=0) + forecast_mu.var(axis=0))  # [n_prep]
        else:
            # Fallback: use base demand with some assumed variance
            expected_demand = self.env.base_demand
            demand_std = 0.3 * expected_demand  # Assume 30% CV
        
        # Dish production: expected demand + safety stock
        # Safety stock = z * std (using z=1.5 for ~86% service level)
        safety_stock_prep = self.safety_factor * demand_std
        
        # Check current inventory
        current_prep_inv = np.array([inv.sum() for inv in self.env.prep_inv])
        
        # Target production: cover demand + safety stock - current inventory
        target_prep = expected_demand + safety_stock_prep - current_prep_inv
        target_prep = np.maximum(target_prep, 0)  # Don't produce negative amounts
        
        # Raw ingredient ordering: compute requirements for target production
        # raw_need = B @ target_prep
        raw_need = self.B @ target_prep  # [n_raw]
        
        # Add safety stock for raw ingredients (buffer for variability)
        raw_std = np.sqrt((self.B**2) @ (demand_std**2))  # Propagate uncertainty
        safety_stock_raw = self.safety_factor * raw_std
        
        # Check current raw inventory
        current_raw_inv = np.array([inv.sum() for inv in self.env.raw_inv])
        
        # Target ordering: cover needs + safety stock - current inventory
        target_raw = raw_need + safety_stock_raw - current_raw_inv
        target_raw = np.maximum(target_raw, 0)
        
        # Combine into action vector: [raw_orders, prep_production]
        action = np.concatenate([target_raw, target_prep])
        
        return action.astype(np.float32)


class BehaviorCloningTrainer:
    """
    Behavior cloning trainer for supervised pretraining from expert demonstrations.
    """
    def __init__(self, agent, expert_policy: ExpertPolicy):
        self.agent = agent
        self.expert = expert_policy
        
    def collect_demonstrations(self, num_episodes: int = 100) -> Tuple[List, List]:
        """
        Collect expert demonstrations by running expert policy.
        
        Returns:
            observations: List of observations
            expert_actions: List of expert actions
        """
        observations = []
        expert_actions = []
        
        env = self.expert.env
        
        for ep in range(num_episodes):
            obs, _ = env.reset()
            ep_reward = 0
            
            while True:
                # Get expert action
                action = self.expert.get_action(obs)
                expert_actions.append(action)
                observations.append(obs)
                
                # Step environment
                next_obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                obs = next_obs
                
                if terminated or truncated:
                    break
            
            if (ep + 1) % 20 == 0:
                print(f"Expert Demo {ep+1}/{num_episodes} | Reward: {ep_reward:.2f}")
        
        print(f"Collected {len(observations)} expert demonstrations")
        return observations, expert_actions
    
    def train(self, observations: List, expert_actions: List, 
              epochs: int = 50, batch_size: int = 64, lr: float = 3e-4) -> List[float]:
        """
        Train model using behavior cloning (supervised learning).
        
        Returns:
            losses: List of training losses per epoch
        """
        # Create optimizer for BC (separate from RL optimizer)
        bc_optimizer = optim.Adam(self.agent.model.parameters(), lr=lr)
        
        losses = []
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(len(observations))
            epoch_loss = 0
            n_batches = 0
            
            for start_idx in range(0, len(observations), batch_size):
                end_idx = min(start_idx + batch_size, len(observations))
                batch_indices = indices[start_idx:end_idx]
                
                # Prepare batch
                batch_obs = [observations[i] for i in batch_indices]
                batch_expert_actions = [expert_actions[i] for i in batch_indices]
                
                # Convert observations to model inputs
                batch_item_features = []
                batch_item_batch = []
                batch_global_features = []
                
                for b, obs in enumerate(batch_obs):
                    inputs = self.agent.obs_to_tensor(obs)
                    batch_item_features.append(inputs['item_features'])
                    batch_item_batch.append(inputs['item_batch'] + b)
                    batch_global_features.append(inputs['global_features'])
                
                batch_item_features = torch.cat(batch_item_features, dim=0)
                batch_item_batch = torch.cat(batch_item_batch, dim=0)
                batch_global_features = torch.cat(batch_global_features, dim=0)
                
                full_obs = {
                    "item_features": batch_item_features,
                    "item_batch": batch_item_batch,
                    "global_features": batch_global_features
                }
                
                # Convert expert actions to tensor
                batch_expert_actions_np = np.concatenate(batch_expert_actions)
                batch_expert_actions_tensor = torch.tensor(
                    batch_expert_actions_np, dtype=torch.float32, device=self.agent.device
                ).view(-1, 1)
                
                # Forward pass
                out = self.agent.model.forward(full_obs)
                mu_items = out['mu_items']  # [num_items, 1]
                
                # MSE loss between predicted mean and expert actions
                loss = F.mse_loss(mu_items, batch_expert_actions_tensor)
                
                # Backward pass
                bc_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.model.parameters(), self.agent.max_grad_norm)
                bc_optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            avg_loss = epoch_loss / n_batches if n_batches > 0 else 0
            losses.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"BC Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")
        
        return losses


class RolloutBuffer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.advantages = []
        self.returns = []
    
    def add(self, obs, action, reward, value, log_prob, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def store_episode(self, obs_list, act_list, val_list, log_prob_list, ret_list, adv_list):
        self.obs.extend(obs_list)
        self.actions.extend(act_list)
        self.values.extend(val_list)
        self.log_probs.extend(log_prob_list)
        self.returns.extend(ret_list)
        self.advantages.extend(adv_list)
        
    def get_batches(self, batch_size):
        indices = np.arange(len(self.obs))
        np.random.shuffle(indices)
        
        for start_idx in range(0, len(self.obs), batch_size):
            end_idx = start_idx + batch_size
            batch_indices = indices[start_idx:end_idx]
            
            yield (
                [self.obs[i] for i in batch_indices],
                [self.actions[i] for i in batch_indices],
                [self.values[i] for i in batch_indices],
                [self.log_probs[i] for i in batch_indices],
                [self.returns[i] for i in batch_indices],
                [self.advantages[i] for i in batch_indices],
            )
    
    def __len__(self):
        return len(self.obs)

class PPOAgent:
    def __init__(self, env, config, lr=3e-4, gamma=0.99, clip_eps=0.2, ent_coef=0.01, val_coef=0.5, max_grad_norm=0.5):
        self.env = env
        self.config = config
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.ent_coef = ent_coef
        self.val_coef = val_coef
        self.max_grad_norm = max_grad_norm
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = InventoryActorCritic(config).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Determine max shelf life for padding
        self.max_shelf_life = 0
        for inv_list in [env.raw_inv, env.prep_inv]:
            for inv in inv_list:
                self.max_shelf_life = max(self.max_shelf_life, len(inv))
        
        # Pre-calculate static features for items to save time
        # Order: Raw ingredients then Dishes
        # Features: [is_prep, cost, price, labor_req, shelf_life_len, ...inventory (padded)...]
        self.n_raw = env.n_raw
        self.n_prep = env.n_prep
        self.n_items = self.n_raw + self.n_prep
        
        self.static_features = np.zeros((self.n_items, 5), dtype=np.float32)
        
        # Raw items
        for i in range(self.n_raw):
            # is_prep=0, cost=cost, price=0, labor=0, shelf_life=len
            self.static_features[i, 0] = 0.0
            self.static_features[i, 1] = env.raw_cost[i]
            self.static_features[i, 2] = 0.0
            self.static_features[i, 3] = 0.0
            self.static_features[i, 4] = len(env.raw_inv[i])
            
        # Prep items
        for j in range(self.n_prep):
            idx = self.n_raw + j
            # is_prep=1, cost=0, price=price, labor=req, shelf_life=len
            self.static_features[idx, 0] = 1.0
            self.static_features[idx, 1] = 0.0
            self.static_features[idx, 2] = env.prep_price[j]
            self.static_features[idx, 3] = env.labor_req[j]
            self.static_features[idx, 4] = len(env.prep_inv[j])

    def obs_to_tensor(self, obs):
        """
        Convert gym observation to model inputs.
        obs structure: {'raw_inv': {...}, 'prep_inv': {...}, 'budget': [...]}
        """
        batch_size = 1 # Single env support for now
        
        # 1. Global Features (Budget)
        # Normalize budget roughly (e.g. / 10000)
        budget = obs['budget'] / 10000.0
        global_features = torch.tensor(budget, dtype=torch.float32, device=self.device).unsqueeze(0) # [1, 1]
        
        # 2. Item Features
        # Collect inventory vectors
        inv_vectors = []
        
        # Order must match env.action_space: Raw then Prep
        for i in range(self.n_raw):
            inv = obs['raw_inv'][f'raw_{i}']
            pad_len = self.max_shelf_life - len(inv)
            if pad_len > 0:
                inv = np.concatenate([inv, np.zeros(pad_len)])
            inv_vectors.append(inv)
            
        for j in range(self.n_prep):
            inv = obs['prep_inv'][f'prep_{j}']
            pad_len = self.max_shelf_life - len(inv)
            if pad_len > 0:
                inv = np.concatenate([inv, np.zeros(pad_len)])
            inv_vectors.append(inv)
            
        inv_block = np.stack(inv_vectors) # [n_items, max_shelf_life]
        
        # Concatenate static and dynamic features
        # static: [n_items, 5], inv: [n_items, max_shelf_life]
        item_feats_np = np.concatenate([self.static_features, inv_block], axis=1)
        item_features = torch.tensor(item_feats_np, dtype=torch.float32, device=self.device)
        
        # 3. Item Batch (all 0)
        item_batch = torch.zeros(self.n_items, dtype=torch.long, device=self.device)
        
        return {
            "item_features": item_features,
            "item_batch": item_batch,
            "global_features": global_features
        }

    def get_action(self, obs, deterministic=False):
        model_inputs = self.obs_to_tensor(obs)
        with torch.no_grad():
            out = self.model.act(model_inputs, deterministic=deterministic)
        
        # out['item_actions'] is [n_items, 1]
        actions = out['item_actions'].cpu().numpy().flatten()
        value = out['value'].cpu().item()
        log_prob = out['env_log_probs'].cpu().item()
        
        return actions, value, log_prob

    def update(self, buffer, batch_size=64, epochs=4):
        """
        Update policy using data in the rollout buffer (Mini-batch PPO)
        """
        total_loss = 0
        n_updates = 0
        
        for _ in range(epochs):
            for batch in buffer.get_batches(batch_size):
                obs_list, act_list, val_list, old_lp_list, ret_list, adv_list = batch
                
                # Collate Observations
                # We need to stack item_features and adjust item_batch indices
                batch_item_features = []
                batch_item_batch = []
                batch_global_features = []
                
                for b, obs in enumerate(obs_list):
                    inputs = self.obs_to_tensor(obs)
                    batch_item_features.append(inputs['item_features'])
                    batch_item_batch.append(inputs['item_batch'] + b)
                    batch_global_features.append(inputs['global_features'])
                    
                batch_item_features = torch.cat(batch_item_features, dim=0)
                batch_item_batch = torch.cat(batch_item_batch, dim=0)
                batch_global_features = torch.cat(batch_global_features, dim=0)
                
                full_obs = {
                    "item_features": batch_item_features,
                    "item_batch": batch_item_batch,
                    "global_features": batch_global_features
                }
                
                # Collate other tensors
                batch_actions = torch.tensor(np.concatenate(act_list), dtype=torch.float32, device=self.device).view(-1, 1)
                batch_old_log_probs = torch.tensor(old_lp_list, dtype=torch.float32, device=self.device)
                batch_returns = torch.tensor(ret_list, dtype=torch.float32, device=self.device)
                batch_advantages = torch.tensor(adv_list, dtype=torch.float32, device=self.device)
                
                # Forward pass
                out = self.model.evaluate_actions(full_obs, batch_actions)
                new_log_probs = out['env_log_probs']
                entropy = out['env_entropy']
                values = out['value']
                
                # Ratio
                ratio = (new_log_probs - batch_old_log_probs).exp()
                
                # Loss
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                value_loss = 0.5 * (batch_returns - values).pow(2).mean()
                entropy_loss = -entropy.mean()
                
                loss = policy_loss + self.val_coef * value_loss + self.ent_coef * entropy_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_loss += loss.item()
                n_updates += 1
                
        return total_loss / n_updates if n_updates > 0 else 0

def compute_gae(rewards, values, next_value, gamma, lam):
    advantages = []
    gae = 0
    values = values + [next_value]
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i+1] - values[i]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    return advantages, [a + v for a, v in zip(advantages, values[:-1])]

def train(use_pretraining: bool = True, bc_episodes: int = 100, bc_epochs: int = 50):
    """
    Train inventory management agent with optional behavior cloning pretraining.
    
    Args:
        use_pretraining: Whether to use behavior cloning pretraining
        bc_episodes: Number of expert demonstration episodes to collect
        bc_epochs: Number of behavior cloning training epochs
    """
    # Init Env
    # Load historical data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "..", "data")
    
    # Ensure data dir exists
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        
    past_demands = load_historical_demands(data_dir)
    if past_demands:
        print(f"Loaded historical demand for dishes: {list(past_demands.keys())}")
    else:
        print("No historical demand found, using synthetic defaults.")

    env = KitchenEnv(past_demands=past_demands)
    
    # Determine dimensions
    max_shelf_life = 0
    for inv_list in [env.raw_inv, env.prep_inv]:
        for inv in inv_list:
            max_shelf_life = max(max_shelf_life, len(inv))
    
    item_feat_dim = 5 + max_shelf_life 
    global_dim = 1 
    
    config = InventoryModelConfig(
        item_input_dim=item_feat_dim,
        global_input_dim=global_dim,
        per_item_action_dim=1,
        item_hidden_dim=64,
        actor_hidden_dim=128,
        critic_hidden_dim=128
    )
    
    agent = PPOAgent(env, config)
    
    # ============================================================
    # BEHAVIOR CLONING PRETRAINING
    # ============================================================
    if use_pretraining:
        print("\n" + "="*60)
        print("STAGE 1: BEHAVIOR CLONING PRETRAINING")
        print("="*60)
        
        # Create expert policy
        expert = ExpertPolicy(env, safety_factor=1.5)
        bc_trainer = BehaviorCloningTrainer(agent, expert)
        
        # Collect expert demonstrations
        print(f"\nCollecting {bc_episodes} expert demonstrations...")
        observations, expert_actions = bc_trainer.collect_demonstrations(num_episodes=bc_episodes)
        
        # Train with behavior cloning
        print(f"\nTraining with behavior cloning for {bc_epochs} epochs...")
        bc_losses = bc_trainer.train(observations, expert_actions, epochs=bc_epochs)
        
        # Save pretrained model
        torch.save(agent.model.state_dict(), "pretrained_bc_model.pth")
        print("\nBehavior cloning pretraining complete!")
        print(f"Final BC Loss: {bc_losses[-1]:.4f}")
        print("Pretrained model saved to pretrained_bc_model.pth")
        
        # Plot BC losses
        plt.figure(figsize=(10, 5))
        plt.plot(bc_losses)
        plt.title("Behavior Cloning Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.yscale('log')
        plt.grid(True)
        plt.savefig("bc_training_curve.png")
        print("BC training curve saved to bc_training_curve.png")
        plt.close()
    
    # ============================================================
    # REINFORCEMENT LEARNING FINE-TUNING
    # ============================================================
    print("\n" + "="*60)
    print("STAGE 2: PPO REINFORCEMENT LEARNING")
    print("="*60 + "\n")
    
    buffer = RolloutBuffer()
    
    num_steps = 15000  # Total timesteps
    buffer_size = 600  # Update every N steps (e.g. 20 episodes)
    batch_size = 64    # Mini-batch size
    epochs = 4
    
    print(f"Starting training for {num_steps} steps...")
    
    global_step = 0
    ep_rewards = []
    
    obs, _ = env.reset()
    ep_rew = 0
    
    obs_list, act_list, logp_list, val_list, rew_list = [], [], [], [], []
    
    while global_step < num_steps:
        
        # Collect data
        action, value, log_prob = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        obs_list.append(obs)
        act_list.append(action)
        logp_list.append(log_prob)
        val_list.append(value)
        rew_list.append(reward)
        
        ep_rew += reward
        global_step += 1
        obs = next_obs
        
        # Handle Episode End
        if terminated or truncated:
            # Bootstrap
            _, next_value, _ = agent.get_action(obs)
            advantages, returns = compute_gae(rew_list, val_list, next_value, 0.99, 0.95)
            
            buffer.store_episode(obs_list, act_list, val_list, logp_list, returns, advantages)
            
            ep_rewards.append(ep_rew)
            print(f"Step {global_step} | Ep Reward: {ep_rew:.2f}")
            
            # Reset episode lists
            obs_list, act_list, logp_list, val_list, rew_list = [], [], [], [], []
            obs, _ = env.reset()
            ep_rew = 0

        # Update Policy if Buffer Full
        if len(buffer) >= buffer_size:
            loss = agent.update(buffer, batch_size=batch_size, epochs=epochs)
            print(f"Updated Agent | Loss: {loss:.4f}")
            buffer.reset()

    # Save final model
    torch.save(agent.model.state_dict(), "ppo_inventory_model.pth")
    print("\nFinal model saved to ppo_inventory_model.pth")
    
    # Plot RL training rewards
    plt.figure(figsize=(10,5))
    plt.plot(ep_rewards)
    plt.title("PPO Training Rewards (After BC Pretraining)" if use_pretraining else "PPO Training Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.savefig("ppo_training_curve.png")
    print("PPO training curve saved to ppo_training_curve.png")

if __name__ == "__main__":
    train()
