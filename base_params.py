import jax
import jax.numpy as jnp 
import envs.environnment


KRW_TO_USD = 0.0007 
GJ_TO_MHW_COAL = 8
GJ_TO_MHW_GAS = 6

EF_BASE = 0.85
EG_BASE = 0.37
CF_BASE = 47500*KRW_TO_USD
CG_BASE = 96600*KRW_TO_USD
KAPPA_BASE = 25911*KRW_TO_USD

BM_FUEL = 0.8
BM_GREEN = 0.4
BM_UNIFORM = 0.7

BM_RATIO = {'large': 0.75, 'base':0.13}


CAP_XI_BASE = 100
CAP_F_BASE = CAP_XI_BASE*EF_BASE*1.1 #Can buy 1.1 the amount of allowances it would need if it produces CAP_XI_BASE
SIGMA_EPS_BASE = CAP_XI_BASE*EF_BASE/20

A0_BASE = 8637*KRW_TO_USD 

# P_mean = 160000*KRW_TO_USD
# P_hz = 10
# P_amplitude = 40000*KRW_TO_USD
# EPS0_P = 10

# def generate_prices_base(key, T):
#     times = (jnp.arange(T + 1, dtype = float)/T)
#     return jnp.maximum(P_mean + P_amplitude*jnp.sin(times*P_hz) + jax.random.normal(key, (int(T+1),))*EPS0_P, 0)

# P_SCALE_BASE = jax.std(generate_prices_base(jax.random.PRNGKey(27), T_BASE))

THETA_P = 0.01
MU_P = 79
SIGMA_P = 6
P0 = 75


def generate_prices_ou(key, T):
    """
    Generates a price path using the Ornstein-Uhlenbeck process.
    T: Number of time steps (days)
    P_start: The initial price to start the simulation
    """
    dt = 365/T
    # 1. Generate all random shocks at once for JAX efficiency
    # shocks = np.sqrt(dt)*jax.random.normal(key, (int(T),))
    shocks = jax.random.normal(key, (int(T),))
    
    # 2. Define the scan function (Recursive step)
    # The 'state' is the price at time t-1
    def step(current_p, epsilon):
        # OU formula: dP = theta * (mu - P) * dt + sigma * dW
        # Assuming dt = 1 (daily steps)
        next_p = current_p + THETA_P * (MU_P - current_p)*dt + SIGMA_P * epsilon
        
        # Ensure prices stay non-negative
        next_p = jnp.maximum(next_p, 0.0)
        
        return next_p, next_p

    # 3. Use jax.lax.scan to loop through T steps efficiently
    _, price_path = jax.lax.scan(step, P0, shocks)
    
    # Prepend the starting price to the result
    return jnp.concatenate([jnp.array([P0]), price_path])


P_SCALE_BASE = P0
A_SCALE_BASE = A0_BASE


THETA0_100_BASE = CAP_XI_BASE*EF_BASE
A_FLOOR_NULL = 0
A_FLOOR_BASE = A0_BASE


def market_impact_base(f):
    return f/(CAP_F_BASE)

def white_noise_A_base(key, T):
    return jax.random.normal(key, (T,)) * 0.25

def idiosyncratic_noise_base(key, sigma_eps, T):
    return jnp.abs(jax.random.normal(key, (T,)) * sigma_eps)

#Theta0,ef,eg,cf,cg,cap_f,cap_xi,sigma_eps,tec_control
PRIVATE_GENERATOR = [THETA0_100_BASE, EF_BASE, EG_BASE, CF_BASE, CG_BASE, CAP_F_BASE, CAP_XI_BASE, SIGMA_EPS_BASE, -1.0]
TEC_PRIVATE = 21/154

BIG_PUBLIC_GENERATOR = [7*THETA0_100_BASE, EF_BASE, EG_BASE, CF_BASE, CG_BASE, CAP_F_BASE*7, CAP_XI_BASE*7, SIGMA_EPS_BASE*7, -1.0]
TEC_PUBLIC = 141/188

# agent_params_60_eta08 = [THETA0_100_BASE*0.6, EF_BASE, EG_BASE, CF_BASE, CG_BASE, CAP_F_BASE, CAP_XI_BASE, SIGMA_EPS_BASE, 0.8]
# agent_params_60_eta06 = [THETA0_100_BASE*0.6, EF_BASE, EG_BASE, CF_BASE, CG_BASE, CAP_F_BASE, CAP_XI_BASE, SIGMA_EPS_BASE, 0.2]

MARKET_MAKER = [0, EF_BASE, EG_BASE, CF_BASE, CG_BASE, CAP_F_BASE,1e-5, 1e-5, 0]


 

# Templates define the core architecture (Capacities, Costs, Emissions)
agent_templates = {
    'base': PRIVATE_GENERATOR,
    'large': BIG_PUBLIC_GENERATOR,
    'mm': MARKET_MAKER
}

fp_configs = {
    'test':  {
    'list_T': [10], 
    'list_iterations': [100],
    'list_lr': [1e-4],
    'batch_size': 500, 
    'list_gamma': [0, 0]
    }, 
    'medium': {
    'list_T': [10, 50, 80, 100],
    'list_iterations': [2000, 300, 100, 10],
    'list_lr': [1e-4, 1e-5, 1e-5, 1e-5], 
    'batch_size': 500, 
    'list_gamma': [0, 0]}, 

    'full': {
    'list_T': [10, 100, 200, 365],
    'list_iterations': [500, 500, 500, 3000],
    'list_lr': [1e-4, 1e-5, 1e-5, 1e-5, 1e-5], 
    'batch_size': 500, 
    'list_gamma': [0, 0]
}
}

# ENV_BASE = envs.environnment.ExogenousMarketEnvJAX(
#     kappa=KAPPA_BASE, T=T_BASE, 
#     agent_params_list=[PRIVATE_GENERATOR], 
#     agent_counts=[50], 
#     generate_P_func=generate_prices_base, 
#     A0=A0_BASE, P0=P_mean, Afloor= A_FLOOR_NULL,
#     market_impact_func=market_impact_base,
#     generate_eps0_func=white_noise_A_base,
#     generate_eps_idiosyncratic_func=idiosyncratic_noise_base, 
#     A_scale=A_SCALE_BASE, 
#     P_scale=P_SCALE_BASE
# )


import json
import os
from itertools import product

def generate_scenarios(scenarios_folder, base_config, sweep_params):
    """
    Generates JSON files for all combinations of sweep_params.
    
    Args:
        scenarios_folder: Folder to save the JSON files.
        base_config: Dictionary of default values.
        sweep_params: Dictionary where keys are parameter names 
                      and values are lists of values to test.
    """
    os.makedirs(scenarios_folder, exist_ok=True)
    
    # Extract keys and the lists of values
    keys = sweep_params.keys()
    values = sweep_params.values()
    
    # Generate every combination
    # e.g., if ratios=[0.4, 0.6] and etas=[-1, 0], this gives 4 combinations
    combinations = list(product(*values))
    
    for i, combo in enumerate(combinations):
        # Create a copy of the base config
        scenario = base_config.copy()
        
        # Create a unique name based on the combination
        name_parts = []
        
        # Apply the sweep values to the scenario
        for key, value in zip(keys, combo):
            scenario[key] = value
            # Build a string for the filename (e.g., "ratio_0.6_eta_0.0")
            name_parts.append(f"{key}_{value}")
        
        scenario_name = "_".join(name_parts)
        file_path = os.path.join(scenarios_folder, f"{scenario_name}.json")
        
        # Save to JSON
        with open(file_path, 'w') as f:
            json.dump(scenario, f, indent=4)
            
    print(f"Successfully generated {len(combinations)} scenarios in '{scenarios_folder}'")
