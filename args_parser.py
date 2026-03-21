import argparse
import json
import os
import base_params

def get_parser():
    parser = argparse.ArgumentParser(description="K-ETS Scenario Loader")
    # This is the ONLY thing the parser needs
    parser.add_argument('--scenario_path', type=str, required=True, 
                        help='Full path to the scenario JSON file')
    return parser

def load_all_from_json(args):
    """
    Reads the JSON path from args, overrides defaults, 
    and builds the agent_params_list exactly as specified.
    """
    if not os.path.exists(args.scenario_path):
        raise FileNotFoundError(f"Scenario file not found: {args.scenario_path}")

    with open(args.scenario_path, 'r') as f:
        # This is now our source of truth
        config = json.load(f)

    # 1. Validation: Ensure all lists in the JSON match in length
    # We use config['key'] because these variables are INSIDE the JSON now
    nb_types = len(config['agent_types'])
    # required_lists = ['initial_allowances_ratio', 'control_technology', 'agent_counts', 'Afloor', "fp_config_str", "BM_uniform", "agent_types", "fp_iterations"]
    
    # if not all(len(config[lst]) == nb_types for lst in required_lists):
    #     raise ValueError(f"All lists in {args.scenario_path} must have length {nb_types}!")

    agent_params_list = []
    
    for k in range(nb_types):
        # 2. Get a FRESH copy of the template from base_params
        template_key = config['agent_types'][k]
        agent = list(base_params.agent_templates[template_key])
        
        # 3. Apply overrides (Respecting your Index 0 and Index 8 logic)
        
        # Index 0: THETA0 (Multiply by ratio)
        agent[0] *= config['initial_allowances_ratio'][k]
        
        if template_key != 'mm':
            # Respect your BM_uniform logic from the JSON toggle
            if config.get('BM_uniform', True): 
                agent[0] *= base_params.BM_UNIFORM
            else: 
                # Benchmarking calculation based on fuel/green ratios
                bm_val = (base_params.BM_FUEL * base_params.BM_RATIO[template_key] + 
                          base_params.BM_GREEN * (1 - base_params.BM_RATIO[template_key]))
                agent[0] *= bm_val
            
        # Index 8: tech_control (eta) - Absolute override
        agent[8] = config['control_technology'][k]
        
        agent_params_list.append(agent)

    # 4. Final Bundle
    config['agent_params_list'] = agent_params_list
    
    # Map the string (e.g., "full") to the actual config object in base_params
    fp_key = config.get('fp_config_str', 'full')
    config['fp_config'] = base_params.fp_configs[fp_key]
    
    # Save the output folder name based on the JSON filename
    scenario_name = os.path.splitext(os.path.basename(args.scenario_path))[0]
    config['scenario_name'] = scenario_name
    config['folder_name'] = os.path.join(config.get('results_root', 'results'), scenario_name)

    return config