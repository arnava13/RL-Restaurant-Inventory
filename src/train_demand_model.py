import os
import torch
import numpy as np
from dataloader import load_historical_demands
from demand_model import BootstrappedDemandModel

def train_demand_model():
    # Configuration
    DATA_DIR = "data" 
    MODEL_SAVE_PATH = "models/demand_model.pt"
    
    # Setup paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data")
    save_dir = os.path.join(project_root, "models")
    save_path = os.path.join(save_dir, "demand_model.pt")
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    print(f"Loading historical data from {data_dir}...")
    history = load_historical_demands(data_dir)
    
    if not history:
        print(f"No data found in {data_dir}. Generating synthetic data for demonstration...")
        # Synthetic data: 2 dishes, 100 days
        # Dish 0: Mean 12, slight trend
        # Dish 1: Mean 30, seasonality
        # WE INJECT EXTRA NOISE HERE to ensure the LSTM doesn't perfectly match the Env GP
        # This simulates "forecast error" or "unknown unknowns"
        t = np.arange(100)
        history = {
            0: [np.random.poisson(12 + 0.05*i + np.random.normal(0, 2)) for i in t],
            1: [np.random.poisson(30 + 5*np.sin(2*np.pi*i/7) + np.random.normal(0, 5)) for i in t]
        }

    # Determine n_dishes from data
    n_dishes = len(history)
    print(f"Found history for {n_dishes} dishes.")

    # Initialize model
    model = BootstrappedDemandModel(
        n_dishes=n_dishes,
        window_size=7,
        n_models=5,
        hidden_dim=64,
        lr=1e-3,
        epochs=50,
        batch_size=16
    )

    # Train
    print("Starting training...")
    model.fit(history)

    # Save
    print(f"Saving model to {save_path}...")
    model.save(save_path)
    print("Done!")

if __name__ == "__main__":
    train_demand_model()
