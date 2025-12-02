import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import copy

class DemandDataset(Dataset):
    """
    Creates sliding windows for LSTM training.
    Returns (sequence, target) pairs.
    """
    def __init__(self, data: np.ndarray, window_size: int):
        """
        Args:
            data: shape (n_days, n_dishes)
            window_size: length of input sequence
        """
        self.data = torch.FloatTensor(data)
        self.window_size = window_size
        
    def __len__(self):
        return len(self.data) - self.window_size
        
    def __getitem__(self, idx):
        # Input: window of days [t, t+w]
        x = self.data[idx : idx + self.window_size]
        # Target: next day [t+w]
        y = self.data[idx + self.window_size]
        return x, y

class GaussianLSTM(nn.Module):
    """
    Multivariate LSTM predicting Probabilistic Demand (Mu, Sigma) for all dishes.
    """
    def __init__(self, n_dishes: int, hidden_dim: int = 64, num_layers: int = 1):
        super().__init__()
        self.n_dishes = n_dishes
        
        self.lstm = nn.LSTM(
            input_size=n_dishes,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Output head: predicts mu and log_sigma for each dish
        # Output dim = 2 * n_dishes
        self.head = nn.Linear(hidden_dim, n_dishes * 2)
        
    def forward(self, x):
        # x: (batch, window, n_dishes)
        out, _ = self.lstm(x)
        
        # Take last time step features
        last_step = out[:, -1, :] # (batch, hidden)
        
        # Predict params
        params = self.head(last_step) # (batch, 2 * n_dishes)
        
        # Split into mu and log_sigma
        mu, log_sigma = torch.chunk(params, 2, dim=-1)
        
        # Clamp log_sigma for stability and exponentiate
        sigma = torch.exp(torch.clamp(log_sigma, min=-5.0, max=2.0))
        
        return mu, sigma

    def loss_function(self, mu, sigma, targets):
        """
        Negative Log Likelihood of Gaussian.
        targets: (batch, n_dishes)
        """
        # NLL = 0.5 * (log(2pi) + 2log(sigma) + (y-mu)^2/sigma^2)
        # We can ignore constant terms for optimization
        var = sigma.pow(2)
        nll = 0.5 * (torch.log(var) + (targets - mu).pow(2) / var)
        return nll.mean()

class BootstrappedDemandModel:
    """
    Ensemble of Gaussian LSTMs trained on bootstrapped data.
    Captures both aleatoric (inherent noise) and epistemic (model) uncertainty.
    """
    def __init__(
        self, 
        n_dishes: int, 
        window_size: int = 7,
        n_models: int = 5,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        epochs: int = 100,
        batch_size: int = 32,
        device: str = "cpu"
    ):
        self.n_dishes = n_dishes
        self.window_size = window_size
        self.n_models = n_models
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        
        # Initialize ensemble
        self.models = [
            GaussianLSTM(n_dishes, hidden_dim).to(device) 
            for _ in range(n_models)
        ]
        
        self.is_fitted = False

    def fit(self, historical_demands: Dict[int, List[float]]):
        """
        Train the ensemble on historical data.
        
        Args:
            historical_demands: Dict {dish_id: [day1, day2, ...]}
        """
        # 1. Convert Dict to aligned Numpy Array (n_days, n_dishes)
        # Assume all dishes have same length history for simplicity
        # or truncate to min length
        dish_ids = sorted(historical_demands.keys())
        if not dish_ids:
            print("No data provided to fit.")
            return
            
        min_len = min(len(v) for v in historical_demands.values())
        if min_len <= self.window_size:
            print(f"Not enough history ({min_len}) for window size {self.window_size}")
            return
            
        data_matrix = np.zeros((min_len, self.n_dishes), dtype=np.float32)
        for i, d_id in enumerate(dish_ids):
            data_matrix[:, i] = historical_demands[d_id][:min_len]
            
        # 2. Train each model on a bootstrap sample
        print(f"Training {self.n_models} models on {min_len} days of history...")
        
        full_dataset = DemandDataset(data_matrix, self.window_size)
        
        for i, model in enumerate(self.models):
            optimizer = optim.Adam(model.parameters(), lr=self.lr)
            model.train()
            
            # Bootstrap: Sample indices with replacement
            indices = np.random.choice(len(full_dataset), size=len(full_dataset), replace=True)
            subset = torch.utils.data.Subset(full_dataset, indices.tolist())
            loader = DataLoader(subset, batch_size=self.batch_size, shuffle=True)
            
            epoch_loss = 0.0
            for epoch in range(self.epochs):
                running_loss = 0.0
                batches = 0
                for x, y in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    
                    optimizer.zero_grad()
                    mu, sigma = model(x)
                    loss = model.loss_function(mu, sigma, y)
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item()
                    batches += 1
                
                if batches > 0:
                    epoch_loss = running_loss / batches

            print(f"Model {i+1}/{self.n_models} fitted. Final Loss: {epoch_loss:.4f}")
            
        self.is_fitted = True

    def predict(self, recent_history: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict next day's demand parameters using the ensemble.
        
        Args:
            recent_history: shape (window_size, n_dishes)
            
        Returns:
            mu_pred: (n_models, n_dishes)
            sigma_pred: (n_models, n_dishes)
        """
        if not self.is_fitted:
            # Return dummy values if not fitted
            # Shape needs to match expected output
            return (np.zeros((self.n_models, self.n_dishes)), 
                    np.ones((self.n_models, self.n_dishes)))
            
        x = torch.FloatTensor(recent_history).unsqueeze(0).to(self.device) # (1, window, n_dishes)
        
        mus = []
        sigmas = []
        
        with torch.no_grad():
            for model in self.models:
                model.eval()
                mu, sigma = model(x)
                mus.append(mu.cpu().numpy())
                sigmas.append(sigma.cpu().numpy())
                
        mus = np.concatenate(mus, axis=0)     # (n_models, n_dishes)
        sigmas = np.concatenate(sigmas, axis=0) # (n_models, n_dishes)
        
        # Return full ensemble predictions to preserve rich uncertainty information
        # Shape: (n_models, n_dishes) for both
        return mus, sigmas

    def save(self, path: str):
        """Save ensemble state dicts."""
        state = {
            'models': [m.state_dict() for m in self.models],
            'config': {
                'n_dishes': self.n_dishes,
                'window_size': self.window_size,
                'n_models': self.n_models,
                'hidden_dim': self.hidden_dim
            },
            'is_fitted': self.is_fitted
        }
        torch.save(state, path)
        
    @classmethod
    def load(cls, path: str, device: str = "cpu"):
        """Load trained ensemble."""
        state = torch.load(path, map_location=device)
        cfg = state['config']
        
        instance = cls(
            n_dishes=cfg['n_dishes'],
            window_size=cfg['window_size'],
            n_models=cfg['n_models'],
            hidden_dim=cfg['hidden_dim'],
            device=device
        )
        
        for model, state_dict in zip(instance.models, state['models']):
            model.load_state_dict(state_dict)
            
        instance.is_fitted = state['is_fitted']
        return instance
