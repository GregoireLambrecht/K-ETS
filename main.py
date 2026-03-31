import jax
import jax.random as jrand
import equinox as eqx
import json
import os
import base_params
from envs.environnment import ExogenousMarketEnvJAX # Fixed typo in environment
from args_parser import get_parser, load_all_from_json # Use the new loading logic
from utils import fictitious_play, evaluate_and_plot
import numpy as np

def main():
    # 1. Get the path from the parser
    parser = get_parser()
    args = parser.parse_args()
    
    # 2. Load EVERYTHING from the JSON path
    # config is now a dictionary containing agent_params_list, fp_config, etc.
    config = load_all_from_json(args)
    
    # 3. Create the directory for this specific scenario
    os.makedirs(config['folder_name'], exist_ok=True)

    # 4. Initialize Environment with parsed params from config dict
    env = ExogenousMarketEnvJAX(
        kappa=base_params.KAPPA_BASE, 
        T=base_params.T_BASE, 
        agent_params_list=config['agent_params_list'], 
        agent_counts=config['agent_counts'], 
        generate_P_func=base_params.generate_prices_ou, 
        A0=base_params.A0_BASE, 
        P0=base_params.P_mean, 
        Afloor=config['Afloor'],
        market_impact_func=base_params.market_impact_base,
        generate_eps0_func=base_params.white_noise_A_base,
        generate_eps_idiosyncratic_func=base_params.idiosyncratic_noise_base, 
        A_scale=base_params.A_SCALE_BASE, 
        P_scale=base_params.P_SCALE_BASE
    )

    print(f"Starting simulation: {config['scenario_name']}")
    print(f"Saving to: {config['folder_name']}")

    # 5. Run Fictitious Play
    # Using the fp_config object mapped from the string in the JSON
    
    ensemble = fictitious_play(
        env=env, 
        config=config['fp_config'], 
        nb_policy=config['fp_iterations'], 
        key=jrand.PRNGKey(42),
        plot_report=False,
        # save_path=config['folder_name'] 
    )

    # # --- 6. Saving ---
    # # Save the trained policies
    model_path = f"{config['folder_name']}/{config['scenario_name']}.eqx"
    eqx.tree_serialise_leaves(model_path, env.policies)

    env_init_params = dict(
    kappa=base_params.KAPPA_BASE,
    T=base_params.T_BASE,
    agent_params_list=config["agent_params_list"],
    agent_counts=config["agent_counts"],
    generate_P_func="generate_prices_ou",
    A0=base_params.A0_BASE,
    P0=base_params.P_mean,
    Afloor=config["Afloor"],
    market_impact_func="market_impact_base",
    generate_eps0_func="white_noise_A_base",
    generate_eps_idiosyncratic_func="idiosyncratic_noise_base",
    A_scale=base_params.A_SCALE_BASE,
    P_scale=base_params.P_SCALE_BASE,
    )

    model_path_env = f"{config['folder_name']}/{config['scenario_name']}_env.json"
    with open(model_path_env, "w") as f:
        json.dump(env_init_params, f, indent=2)
    
    
    # # Save metadata for the audit trail
    with open(f"{config['folder_name']}/details.txt", "w") as f:
        f.write("K-ETS Simulation Details\n")
        f.write("========================\n")
        f.write(f"Scenario Name: {config['scenario_name']}\n")
        f.write(f"Agent Types: {config['agent_types']}\n")
        f.write(f"Counts: {config['agent_counts']}\n")
        f.write(f"Initial Ratios: {config['initial_allowances_ratio']}\n")
        f.write(f"BM Uniform: {config.get('BM_uniform', True)}\n")
        f.write(f"Control Tech (Eta): {config['control_technology']}\n")
        f.write(f"FP Config Type: {config.get('fp_config_str', 'default')}\n")

    states, actions, A_history, rewards = evaluate_and_plot(
        env,
        env.policies,
        num_simulations=100,
        key=jax.random.PRNGKey(33),
        plot_report=False
    )

    results_path = f"{config['folder_name']}/{config['scenario_name']}_data.npz"

    # Save all arrays into a single compressed file
    # We convert to numpy explicitly to ensure they are off-device 
    # and compatible with standard loading tools
    np.savez_compressed(
        results_path,
        states=np.array(states),
        actions=np.array(actions),
        A_history=np.array(A_history),
        rewards=np.array(rewards)
    )

if __name__ == "__main__":
    main()


# data = np.load("your_folder/your_scenario_results.npz")
# actions = data['actions']