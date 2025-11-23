import numpy as np
import gymnasium as gym
from gymnasium import spaces

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

class KitchenEnv(gym.Env):
    """
    Kitchen inventory & production environment.
    Supports random shelf life and random arrival delays
    for both ingredients and dishes.
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
    ):
        super().__init__()

        # ---------- default config ----------
        if ingredients is None or dishes is None:
            ing0 = Ingredient("raw_0", 0, shelf_life_base=3, cost=10.0)
            ing1 = Ingredient("raw_1", 1, shelf_life_base=7, cost=5.0)
            ingredients = [ing0, ing1]

            dish0 = Dish(
                "prep_0", 0,
                shelf_life_base=2,
                price=40.0,
                labor_req=3.0,
                recipe={ing0: 0.2, ing1: 0.1},
            )
            dish1 = Dish(
                "prep_1", 1,
                shelf_life_base=5,
                price=30.0,
                labor_req=2.0,
                recipe={ing0: 0.1, ing1: 0.2},
            )
            dishes = [dish0, dish1]

        # store objects
        self.ingredients = ingredients
        self.dishes = dishes

        # ---------- derive arrays ----------
        self.n_raw = len(self.ingredients)
        self.n_prep = len(self.dishes)

        # raw parameters
        self.raw_shelf_life = np.array(
            [ing.shelf_life() for ing in self.ingredients], dtype=np.int32
        )
        self.raw_cost = np.array([ing.cost for ing in self.ingredients], dtype=np.float32)

        # prep parameters
        self.prep_shelf_life = np.array(
            [d.shelf_life() for d in self.dishes], dtype=np.int32
        )
        self.prep_price = np.array([d.price for d in self.dishes], dtype=np.float32)
        self.labor_req = np.array([d.labor_req for d in self.dishes], dtype=np.float32)

        # recipe matrix B
        self.B = self._build_recipe_matrix()

        # environment constants
        self.capacity = capacity
        self.labor_limit = labor_limit

        # action space
        self.action_space = spaces.Box(
            low=0,
            high=1000,
            shape=(self.n_raw + self.n_prep,),
            dtype=np.float32,
        )

        # observation space (no pipeline)
        self.observation_space = spaces.Dict({
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
        })

        # costs
        self.holding = float(holding)
        self.spoil_penalty = float(spoil_penalty)
        self.short_penalty = float(short_penalty)
        self.labor_cost = float(labor_cost)

        # arrival queues (raw & prepared)
        self.arrival_raw = [[] for _ in range(self.n_raw)]
        self.arrival_prep = [[] for _ in range(self.n_prep)]

        self.reset()

    # ---------- build B ----------
    def _build_recipe_matrix(self):
        B = np.zeros((len(self.ingredients), len(self.dishes)), dtype=np.float32)
        ing_index = {ing.id: idx for idx, ing in enumerate(self.ingredients)}
        dish_index = {dish.id: jdx for jdx, dish in enumerate(self.dishes)}

        for dish in self.dishes:
            j = dish_index[dish.id]
            for ing, qty in dish.recipe.items():
                i = ing_index[ing.id]
                B[i, j] = float(qty)
        return B

    # ---------- RESET ----------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.raw_inv  = [np.zeros(int(life), dtype=np.float32) for life in self.raw_shelf_life]
        self.prep_inv = [np.zeros(int(life), dtype=np.float32) for life in self.prep_shelf_life]

        self.arrival_raw = [[] for _ in range(self.n_raw)]
        self.arrival_prep = [[] for _ in range(self.n_prep)]

        self.day = 0
        self.budget = 10000.0
        self.labor_left = self.labor_limit

        return self._get_obs(), {}

    # ---------- OBS ----------
    def _get_obs(self):
        raw_obs = {f"raw_{i}": self.raw_inv[i].copy() for i in range(self.n_raw)}
        prep_obs = {f"prep_{j}": self.prep_inv[j].copy() for j in range(self.n_prep)}

        return {
            "raw_inv": raw_obs,
            "prep_inv": prep_obs,
            "budget": np.array([self.budget], dtype=np.float32),
        }

    # ---------- STEP ----------
    def step(self, action):
        q_raw = np.maximum(action[:self.n_raw], 0)
        y_prep = np.maximum(action[self.n_raw:], 0)

        spoil_raw  = np.zeros(self.n_raw, dtype=np.float32)
        spoil_prep = np.zeros(self.n_prep, dtype=np.float32)

        # -------------------------------------
        # PROCESS ARRIVAL QUEUES (raw + prep)
        # -------------------------------------
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

        # -------------------------------------
        # ADD TODAY’S NEW ORDERS TO QUEUE
        # -------------------------------------
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

        # -------------------------------------
        # RAW AGING + ARRIVAL
        # -------------------------------------
        for i in range(self.n_raw):
            spoil_raw[i] = self.raw_inv[i][-1]
            self.raw_inv[i][1:] = self.raw_inv[i][:-1]
            self.raw_inv[i][0] += arrivals_raw[i]

        # -------------------------------------
        # RAW CONSUMED FOR PRODUCTION
        # -------------------------------------
        raw_need = (self.B @ y_prep).astype(np.float32)
        for i in range(self.n_raw):
            used = min(self.raw_inv[i].sum(), raw_need[i])
            self.raw_inv[i] = self._fifo_consume(self.raw_inv[i], used)

        # -------------------------------------
        # PREP AGING + ARRIVAL
        # -------------------------------------
        for j in range(self.n_prep):
            spoil_prep[j] = self.prep_inv[j][-1]
            self.prep_inv[j][1:] = self.prep_inv[j][:-1]
            self.prep_inv[j][0] = arrivals_prep[j]

        # -------------------------------------
        # DEMAND
        # -------------------------------------
        demand = np.random.poisson(8, size=self.n_prep)
        sold = np.zeros(self.n_prep)
        short = np.zeros(self.n_prep)

        for j in range(self.n_prep):
            available = self.prep_inv[j].sum()
            sell = min(demand[j], available)
            sold[j] = sell
            short[j] = demand[j] - sell
            self.prep_inv[j] = self._fifo_consume(self.prep_inv[j], sell)

        # -------------------------------------
        # COSTS & REWARD
        # -------------------------------------
        revenue = float(np.dot(self.prep_price, sold))
        cost_raw = float(np.dot(self.raw_cost, q_raw))

        total_raw_units = sum(vec.sum() for vec in self.raw_inv)
        total_prep_units = sum(vec.sum() for vec in self.prep_inv)
        hold_cost = self.holding * (total_raw_units + total_prep_units)

        total_spoil = spoil_raw.sum() + spoil_prep.sum()
        spoil_cost = self.spoil_penalty * total_spoil

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
            "spoil": float(total_spoil),
            "short": float(short.sum()),
            "labor": labor_used,
        }

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