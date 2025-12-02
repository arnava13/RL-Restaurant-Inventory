"""
Kitchen Inventory Management Environment with Gaussian Process Demand Forecasting

This module implements a Gymnasium-compatible environment for simulating kitchen inventory
management with realistic demand patterns. It models ingredients with varying shelf lives,
prepared dishes with recipes, and uses Gaussian Process regression to generate dynamic,
time-varying demand patterns.

Key Components:
- Ingredient: Raw materials with shelf life, arrival delays, and costs
- Dish: Prepared items with recipes, labor requirements, and prices
- DemandGP: Gaussian Process model for generating realistic demand with trends/seasonality
- KitchenEnv: Main Gym environment for inventory optimization

The environment supports:
- Multi-item inventory tracking with spoilage
- Production planning with labor and capacity constraints
- GP-based demand forecasting with trend and seasonal components
- Reward computation based on profit, holding costs, spoilage, and shortages

"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from IPython import display
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, ConstantKernel as C, DotProduct, ExpSineSquared
)

class Ingredient:
    def __init__(self, name, ingredient_id,
                 shelf_life_base,
                 shelf_life_var=lambda: 0,
                 arrival_delay_base=0,
                 arrival_delay_var=lambda: 0,
                 cost=0.0):

        self.name = name
        self.id = int(ingredient_id)

        # Shelf life
        self.shelf_life_base = int(shelf_life_base)
        self.shelf_life_var = shelf_life_var

        # Arrival delay
        self.arrival_delay_base = int(arrival_delay_base)
        self.arrival_delay_var = arrival_delay_var

        self.cost = float(cost)

    def shelf_life(self):
        """ base + random variation ≥1 """
        return max(1, self.shelf_life_base + int(self.shelf_life_var()))

    def arrival_delay(self):
        """ base + random variation ≥0 """
        return max(0, self.arrival_delay_base + int(self.arrival_delay_var()))

class Dish:
    def __init__(self, name, dish_id,
                 shelf_life_base,
                 shelf_life_var=lambda: 0,
                 arrival_delay_base=0,
                 arrival_delay_var=lambda: 0,
                 price=0.0,
                 labor_req=0.0,
                 recipe=None):

        self.name = name
        self.id = int(dish_id)

        # Shelf life
        self.shelf_life_base = int(shelf_life_base)
        self.shelf_life_var = shelf_life_var

        # Production delay
        self.arrival_delay_base = int(arrival_delay_base)
        self.arrival_delay_var = arrival_delay_var

        self.price = float(price)
        self.labor_req = float(labor_req)
        self.recipe = recipe or {}

    def shelf_life(self):
        return max(1, self.shelf_life_base + int(self.shelf_life_var()))

    def arrival_delay(self):
        """Delay before dish enters inventory (e.g., prep time variability)"""
        return max(0, self.arrival_delay_base + int(self.arrival_delay_var()))

class DemandGP:
    def __init__(self, base_demand, use_trend=True, use_seasonality=True,
                 period=7, noise=0.1):

        # ---- Base kernel (smooth changes) ----
        kernel = C(1.0, (0.01, 100)) * RBF(length_scale=3.0)

        # ---- Add trend ----
        if use_trend:
            kernel += DotProduct()

        # ---- Add seasonality ----
        if use_seasonality:
            kernel += ExpSineSquared(length_scale=1.0, periodicity=period)

        self.gp = GaussianProcessRegressor(kernel=kernel,
                                           alpha=noise,
                                           normalize_y=True)

        # initial data point
        self.X = np.array([[0]])
        self.y = np.array([np.log(base_demand)])
        self.gp.fit(self.X, self.y)

    def sample(self, day):
        t = np.array([[day]])
        mean, std = self.gp.predict(t, return_std=True)
        f = np.random.normal(mean[0], std[0]) # type: ignore
        return np.random.poisson(np.exp(f))

    def update(self, day, obs_demand):
        t = np.array([[day]])
        new_y = np.log(max(1, obs_demand))
        self.X = np.vstack([self.X, t])
        self.y = np.hstack([self.y, new_y])
        self.gp.fit(self.X, self.y)

class KitchenEnv(gym.Env):
    """
    Kitchen inventory & production environment
    incorporating Gaussian-Process-driven demand.
    """

    metadata = {"render_modes": ["human"], "render_fps": 2}

    def __init__(
        self,
        ingredients=None,
        dishes=None,
        capacity=300,
        labor_limit=10,
        holding=0.3,
        spoil_penalty=2.0,
        short_penalty=2.0,
        labor_cost=0.0,
        use_gp_demand=True,
        base_demand=None,
        demand_noise=0.01,
        past_demands=None,
        demand_model_path=None, # NEW: Path to trained LSTM model
    ):
        super().__init__()

        self.use_gp_demand = use_gp_demand
        self.past_demands = past_demands or {}
        self.demand_model = None
        self.history_window = None
        self.window_size = 7 # Default, will be updated from model config

        # Load Demand Model if provided
        if demand_model_path:
            from demand_model import BootstrappedDemandModel
            try:
                self.demand_model = BootstrappedDemandModel.load(demand_model_path)
                self.window_size = self.demand_model.window_size
                print(f"Loaded demand model from {demand_model_path}")
            except Exception as e:
                print(f"Failed to load demand model: {e}")

        # -----------------------------
        # Default ingredients & dishes
        # -----------------------------
        if ingredients is None or dishes is None:
            ing0 = Ingredient("raw_0", 0, shelf_life_base=3, cost=10.0)
            ing1 = Ingredient("raw_1", 1, shelf_life_base=7, cost=5.0)
            ingredients = [ing0, ing1]

            dish0 = Dish(
                "prep_0", 0, shelf_life_base=2,
                price=40.0, labor_req=3.0,
                recipe={ing0: 0.2, ing1: 0.1},
            )
            dish1 = Dish(
                "prep_1", 1, shelf_life_base=5,
                price=30.0, labor_req=2.0,
                recipe={ing0: 0.1, ing1: 0.2},
            )
            dishes = [dish0, dish1]

        self.ingredients = ingredients
        self.dishes = dishes

        # -----------------------------
        # Core arrays
        # -----------------------------
        self.n_raw = len(self.ingredients)
        self.n_prep = len(self.dishes)

        self.raw_shelf_life = np.array(
            [ing.shelf_life() for ing in self.ingredients], dtype=np.int32
        )
        self.raw_cost = np.array([ing.cost for ing in self.ingredients], dtype=np.float32)

        self.prep_shelf_life = np.array(
            [d.shelf_life() for d in self.dishes], dtype=np.int32
        )
        self.prep_price = np.array([d.price for d in self.dishes], dtype=np.float32)
        self.labor_req = np.array([d.labor_req for d in self.dishes], dtype=np.float32)

        self.B = self._build_recipe_matrix()

        self.capacity = capacity
        self.labor_limit = labor_limit

        # Action = [purchase_raw..., produce_dish...]
        self.action_space = spaces.Box(
            low=0, high=1000,
            shape=(self.n_raw + self.n_prep,),
            dtype=np.float32,
        )

        # Observations
        obs_dict = {
            "raw_inv": spaces.Dict({
                f"raw_{i}": spaces.Box(
                    0, np.inf,
                    shape=(int(self.raw_shelf_life[i]),),
                    dtype=np.float32,
                )
                for i in range(self.n_raw)
            }),
            "prep_inv": spaces.Dict({
                f"prep_{j}": spaces.Box(
                    0, np.inf,
                    shape=(int(self.prep_shelf_life[j]),),
                    dtype=np.float32,
                )
                for j in range(self.n_prep)
            }),
            "budget": spaces.Box(0, 1e6, shape=(1,), dtype=np.float32),
        }
        
        # Add forecast to observation space if model exists
        if self.demand_model:
            # Shape: 2 arrays of (n_models, n_dishes) -> Flattened or kept structured?
            # Let's flatten for Gym space simplicity or keep as Box.
            # Output is (n_models, n_dishes). 
            n_models = self.demand_model.n_models
            obs_dict["forecast_mu"] = spaces.Box(
                -np.inf, np.inf, shape=(n_models, self.n_prep), dtype=np.float32
            )
            obs_dict["forecast_sigma"] = spaces.Box(
                0, np.inf, shape=(n_models, self.n_prep), dtype=np.float32
            )

        self.observation_space = spaces.Dict(obs_dict)

        # ----------------------------------------
        # COST PARAMETERS
        # ----------------------------------------
        self.holding = float(holding)
        self.spoil_penalty = float(spoil_penalty)
        self.short_penalty = float(short_penalty)
        self.labor_cost = float(labor_cost)

        # ----------------------------------------
        # GP DEMAND SETUP: one GP per dish
        # ----------------------------------------
        self.demand_noise = demand_noise

        if base_demand is None:
            # Default demand for each dish
            self.base_demand = np.array([8.0] * self.n_prep)
        else:
            self.base_demand = np.array(base_demand)

        if use_gp_demand:
            self._init_gp()

        # ----------------------------------------
        self.reset()

    # ============================================
    #   Gaussian Process: Initialize Models
    # ============================================
    def _init_gp(self):
        self.gp_models = []
        self.gp_X = []
        self.gp_y = []

        for j in range(self.n_prep):
            # Kernel per dish
            kernel = C(1.0, (0.01, 100)) * RBF(length_scale=5.0)
            
            # Add trend/seasonality components if desired (matching original logic)
            # The original __init__ didn't expose these flags, but DemandGP class did.
            # We'll stick to a reasonable default or what was in the original _init_gp
            # Original _init_gp: kernel = C(1.0, (0.01, 20)) * RBF(length_scale=5.0)
            # It didn't add trend/seasonality explicitly in the previous _init_gp, 
            # but DemandGP class has them. 
            # Let's assume we want a robust kernel.
            kernel += DotProduct(sigma_0=1.0) # Trend
            kernel += ExpSineSquared(length_scale=1.0, periodicity=7) # Seasonality

            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=self.demand_noise,
                normalize_y=True,
            )

            dish_id = self.dishes[j].id
            past_data = self.past_demands.get(dish_id, [])
            
            if past_data and len(past_data) > 0:
                # History available
                n = len(past_data)
                # Time: -n, -n+1, ..., -1
                t_hist = np.arange(-n, 0).reshape(-1, 1)
                y_hist = np.log(np.maximum(1, np.array(past_data)))
                
                X0 = t_hist
                y0 = y_hist
            else:
                # No history: use base_demand at t=0
                X0 = np.array([[0]])
                y0 = np.array([np.log(self.base_demand[j])])

            gp.fit(X0, y0)

            self.gp_models.append(gp)
            self.gp_X.append(X0)
            self.gp_y.append(y0)

    # ============================================
    #   GP Demand Generator
    # ============================================
    def generate_gp_demand(self):
        t = np.array([[self.day]])
        demand = np.zeros(self.n_prep, dtype=np.float32)

        for j in range(self.n_prep):

            gp = self.gp_models[j]

            # Predict latent GP demand f(t)
            mean, std = gp.predict(t, return_std=True)
            f = np.random.normal(mean[0], std[0])

            lam = float(np.exp(f))  # Poisson intensity
            lam = max(0.1, lam)
            demand[j] = np.random.poisson(lam)

        return demand

    # ============================================
    #   GP Updater
    # ============================================
    def update_gp(self, demand):
        t = np.array([[self.day]])

        for j in range(self.n_prep):
            X = self.gp_X[j]
            y = self.gp_y[j]

            new_y = np.log(max(1, float(demand[j])))

            X_new = np.vstack([X, t])
            y_new = np.hstack([y, new_y])

            self.gp_models[j].fit(X_new, y_new)

            self.gp_X[j] = X_new
            self.gp_y[j] = y_new

    # ============================================
    #   RECIPE MATRIX
    # ============================================
    def _build_recipe_matrix(self):
        B = np.zeros((self.n_raw, self.n_prep), dtype=np.float32)
        ing_index = {ing.id: idx for idx, ing in enumerate(self.ingredients)}
        dish_index = {d.id: j for j, d in enumerate(self.dishes)}

        for d in self.dishes:
            j = dish_index[d.id]
            for ing, qty in d.recipe.items():
                i = ing_index[ing.id]
                B[i, j] = float(qty)

        return B

    # ============================================
    #   RESET
    # ============================================
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.raw_inv  = [np.zeros(int(life), dtype=np.float32) for life in self.raw_shelf_life]
        self.prep_inv = [np.zeros(int(life), dtype=np.float32) for life in self.prep_shelf_life]

        self.arrival_raw = [[] for _ in range(self.n_raw)]
        self.arrival_prep = [[] for _ in range(self.n_prep)]

        self.day = 0
        self.budget = 10000.0
        self.labor_left = self.labor_limit

        # Initialize history window for LSTM
        if self.demand_model:
            # Fill with 0s or base_demand initially
            self.history_window = np.zeros((self.window_size, self.n_prep), dtype=np.float32)
            for i in range(self.window_size):
                # Add some noise to base demand to make it realistic
                self.history_window[i] = self.base_demand * (1 + np.random.normal(0, 0.1, size=self.n_prep))
                
            self.current_forecast = self.demand_model.predict(self.history_window)
        else:
            self.history_window = None
            self.current_forecast = None

        return self._get_obs(), {}

    # ============================================
    #   OBS
    # ============================================
    def _get_obs(self):
        raw_obs = {f"raw_{i}": self.raw_inv[i].copy() for i in range(self.n_raw)}
        prep_obs = {f"prep_{j}": self.prep_inv[j].copy() for j in range(self.n_prep)}

        obs = {
            "raw_inv": raw_obs,
            "prep_inv": prep_obs,
            "budget": np.array([self.budget], dtype=np.float32),
        }
        
        if self.current_forecast:
            mu, sigma = self.current_forecast
            obs["forecast_mu"] = mu.astype(np.float32)
            obs["forecast_sigma"] = sigma.astype(np.float32)
            
        return obs

    # ============================================
    #   STEP
    # ============================================
    def step(self, action):

        q_raw = np.maximum(action[:self.n_raw], 0)
        y_prep = np.maximum(action[self.n_raw:], 0)
        
        # ... (rest of step implementation)
        # Need to inject history update here. 
        # But I need to see where "demand" is generated in step.
        
        # Let me read the step function again carefully or just replace the beginning and end logic.
        # It's better to use a larger context replacement to ensure correctness.
        
        spoil_raw  = np.zeros(self.n_raw, dtype=np.float32)
        spoil_prep = np.zeros(self.n_prep, dtype=np.float32)

        # ARRIVAL QUEUES -------------------------
        arrivals_raw = np.zeros(self.n_raw)
        for i in range(self.n_raw):
            newQ = []
            for delay, qty in self.arrival_raw[i]:
                if delay <= 0:
                    arrivals_raw[i] += qty
                else:
                    newQ.append((delay-1, qty))
            self.arrival_raw[i] = newQ

        arrivals_prep = np.zeros(self.n_prep)
        for j in range(self.n_prep):
            newQ = []
            for delay, qty in self.arrival_prep[j]:
                if delay <= 0:
                    arrivals_prep[j] += qty
                else:
                    newQ.append((delay-1, qty))
            self.arrival_prep[j] = newQ

        # ADD NEW ARRIVALS -----------------------
        for i in range(self.n_raw):
            qty = float(q_raw[i])
            if qty > 0:
                delay = self.ingredients[i].arrival_delay()
                if delay == 0:
                    arrivals_raw[i] += qty
                else:
                    self.arrival_raw[i].append((delay-1, qty))

        for j in range(self.n_prep):
            qty = float(y_prep[j])
            if qty > 0:
                delay = self.dishes[j].arrival_delay()
                if delay == 0:
                    arrivals_prep[j] += qty
                else:
                    self.arrival_prep[j].append((delay-1, qty))

        # RAW AGING ------------------------------
        for i in range(self.n_raw):
            spoil_raw[i] = self.raw_inv[i][-1]
            self.raw_inv[i][1:] = self.raw_inv[i][:-1]
            self.raw_inv[i][0] += arrivals_raw[i]

        # RAW CONSUMPTION -------------------------
        raw_need = (self.B @ y_prep).astype(np.float32)

        for i in range(self.n_raw):
            used = min(self.raw_inv[i].sum(), raw_need[i])
            self.raw_inv[i] = self._fifo_consume(self.raw_inv[i], used)

        # PREP AGING ------------------------------
        for j in range(self.n_prep):
            spoil_prep[j] = self.prep_inv[j][-1]
            self.prep_inv[j][1:] = self.prep_inv[j][:-1]
            self.prep_inv[j][0] = arrivals_prep[j]

        # DEMAND ---------------------------------
        if self.use_gp_demand:
            demand = self.generate_gp_demand()
            self.update_gp(demand)
        else:
            demand = np.random.poisson(8, size=self.n_prep)
            
        # UPDATE HISTORY WINDOW FOR LSTM
        if self.demand_model and self.history_window is not None:
            # Shift history: remove oldest day, add new demand
            # self.history_window shape: (window_size, n_prep)
            self.history_window = np.roll(self.history_window, -1, axis=0)
            self.history_window[-1] = demand
            
            # Generate forecast for NEXT step (t+1)
            self.current_forecast = self.demand_model.predict(self.history_window)

        sold = np.zeros(self.n_prep)
        short = np.zeros(self.n_prep)

        for j in range(self.n_prep):
            available = self.prep_inv[j].sum()
            sell = min(float(available), float(demand[j]))
            sold[j] = sell
            short[j] = demand[j] - sell
            self.prep_inv[j] = self._fifo_consume(self.prep_inv[j], sell)

        # REWARD ---------------------------------
        revenue = float(np.dot(self.prep_price, sold))
        cost_raw = float(np.dot(self.raw_cost, q_raw))

        total_raw_units = sum(vec.sum() for vec in self.raw_inv)
        total_prep_units = sum(vec.sum() for vec in self.prep_inv)

        hold_cost = self.holding * (total_raw_units + total_prep_units)
        spoil_cost = self.spoil_penalty * (spoil_raw.sum() + spoil_prep.sum())
        short_cost = self.short_penalty * short.sum()

        labor_used = float(np.dot(self.labor_req, y_prep))
        labor_cost_term = self.labor_cost * labor_used

        reward = (
            revenue
            - cost_raw
            - hold_cost
            - spoil_cost
            - short_cost
            - labor_cost_term
        )

        self.day += 1
        terminated = self.day >= 30

        return self._get_obs(), float(reward), terminated, False, {
            "revenue": revenue,
            "sold": float(sold.sum()),
            "spoil": float(spoil_raw.sum() + spoil_prep.sum()),
            "short": float(short.sum()),
            "labor": labor_used,
        }

    # ============================================
    #   FIFO CONSUMPTION
    # ============================================
    @staticmethod
    def _fifo_consume(vec, qty):
        remaining = qty
        for a in range(len(vec)):
            if remaining <= 1e-12:
                break
            used = min(vec[a], remaining)
            vec[a] -= used
            remaining -= used
        return vec

# Add Ingredients
ing0 = Ingredient(
    "fish", 0,
    shelf_life_base=2,
    shelf_life_var=lambda: np.random.randint(0, 2),    # 0–1 extra days
    arrival_delay_base=0,
    arrival_delay_var=lambda: np.random.randint(0, 3), # 0–2 day arrival delay
    cost=12.0
)

ing1 = Ingredient(
    "rice", 1,
    shelf_life_base=10,
    shelf_life_var=lambda: 0,                          # no shelf-life randomness
    arrival_delay_base=0,
    arrival_delay_var=lambda: 0,                       # always arrives immediately
    cost=3.0
)

ing2 = Ingredient(
    "veggies", 2,
    shelf_life_base=5,
    shelf_life_var=lambda: np.random.randint(-1, 2),   # 4–6 days
    arrival_delay_base=0,
    arrival_delay_var=lambda: np.random.randint(0, 2), # 0–1 day delay
    cost=4.0
)

# Add dishes
dish0 = Dish(
    "sushi", 0,
    shelf_life_base=1,
    shelf_life_var=lambda: 0,                  # always 1 day shelf life
    arrival_delay_base=0,
    arrival_delay_var=lambda: np.random.randint(0, 2),  # 0–1 day delay
    price=50,
    labor_req=4,
    recipe={ing0: 0.3, ing1: 0.2}
)

dish1 = Dish(
    "bento", 1,
    shelf_life_base=2,
    shelf_life_var=lambda: 0,                  # always 2 day shelf life
    arrival_delay_base=0,
    arrival_delay_var=lambda: np.random.randint(0, 3),  # 0–2 day delay
    price=45,
    labor_req=3,
    recipe={ing0: 0.2, ing1: 0.3, ing2: 0.4}
)

env = KitchenEnv(
    ingredients=[ing0, ing1, ing2],
    dishes=[dish0, dish1],
    base_demand=[12, 30],   # dish 0 demand = 12/day, dish 1 = 30/day
    demand_noise=0.1,       # GP smoothness
    use_gp_demand=True,   # turn GP demand on/off here
)
obs, info = env.reset()

n_days = 30
rewards = []

for day in range(n_days):
    action = env.action_space.sample()  # random policy
    obs, reward, terminated, truncated, info = env.step(action)

    rewards.append(reward)

    if terminated or truncated:
        break


plt.figure(figsize=(8,5))
plt.plot(rewards, label="Daily Reward (Profit/Loss)", color="orange", linewidth=2)
plt.xlabel("Day")
plt.ylabel("Reward")
plt.title("Daily Reward Over Time")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

def create_demand_gp(base_demand,
                     noise=0.1,
                     use_trend=True,
                     trend_strength=1.0,
                     use_seasonality=True,
                     season_period=7,
                     season_scale=1.0,
                     smooth_length=100.0,
                     past_demands=None,       # past demand as input
                     ):
    """
    Create GP model with optional trend/seasonality and optional past demand.
    If past_demands is None → GP only trains on base_demand.
    If past_demands provided → GP trains using those historical points.
    """

    # -------------------------
    # 1. Kernel
    # -------------------------
    kernel = C(1.0, (0.01, 100)) * RBF(length_scale=smooth_length)

    if use_trend:
        kernel += trend_strength * DotProduct(sigma_0=1.0)

    if use_seasonality:
        kernel += season_scale * ExpSineSquared(
            periodicity=season_period,
            length_scale=1.0
        )

    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=noise,
        normalize_y=True
    )

    # -------------------------
    # Default training point
    # -------------------------
    X = [0]
    y = [np.log(base_demand)]

    # -------------------------
    if past_demands is not None and len(past_demands) > 0:
        t_hist = np.arange(1, len(past_demands) + 1)

        X = np.concatenate([X, t_hist])

        y_hist = np.log(np.clip(past_demands, 1, None))
        y = np.concatenate([y, y_hist])

    X = np.array(X).reshape(-1, 1)
    y = np.array(y)

    # -------------------------
    # Fit GP
    # -------------------------
    gp.fit(X, y)
    return gp

# --- past demand for both dishes
past0 = [11, 12, 13, 10, 14, 16, 15]
past1 = [28, 29, 31, 27, 30, 32, 33]

gp_models = [
    create_demand_gp(base_demand=12, noise=0.05, use_trend=True, past_demands=past0),
    create_demand_gp(base_demand=30, noise=0.05, use_trend=True, past_demands=past1)
]

# ---- Predict next 60 days ----
predictions = []
days_list = []

for past, gp in zip([past0, past1], gp_models):
    days = np.arange(len(past), len(past) + 60)
    days_list.append(days)
    preds = []

    for t in days:
        mean, std = gp.predict(np.array([[t]]), return_std=True)
        f = np.random.normal(mean[0], min(std[0], 0.2)) # type: ignore
        lam = np.exp(f)
        lam = min(lam, 80)
        preds.append(np.random.poisson(max(lam, 0.1)))

    predictions.append(np.array(preds))

# ---- Plot ----
plt.figure(figsize=(12, 6))

# Dish 0
plt.plot(range(len(past0)), past0, "o-", label="Dish 0 Past", color="black")
plt.plot(days_list[0], predictions[0], "s-", label="Dish 0 Pred", color="red")

# Dish 1
plt.plot(range(len(past1)), past1, "o-", label="Dish 1 Past", color="blue")
plt.plot(days_list[1], predictions[1], "s-", label="Dish 1 Pred", color="orange")

plt.xlabel("Day")
plt.ylabel("Demand")
plt.title("Two-Dish GP Demand Forecast Based on Past Observations")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()