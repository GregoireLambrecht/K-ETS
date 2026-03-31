import jax
import jax.numpy as jnp
from jax import vmap
import math

class ExogenousMarketEnvJAX:
    def __init__(self, kappa, T, agent_params_list, agent_counts, generate_P_func, A0, P0, Afloor, market_impact_func, generate_eps0_func, generate_eps_idiosyncratic_func,A_scale=1.0, P_scale=1.0):
        # 1. Store basic constants
        self.kappa = kappa
        self.T = float(T)
        self.A_scale = A_scale
        self.P_scale = P_scale
        self.A0 = A0
        self.P0 = P0
        self.Afloor = Afloor
        self.dt = 365/self.T
        
        # 2. Market Composition (params_list shape: [N_types, 9])
        #Theta0/T,ef,eg,cf,cg,cap_f,cap_xi,sigma_eps
        self.agent_params_list = jnp.array(agent_params_list)
        # In __init__
        self.agent_counts_list = [int(c) for c in agent_counts]
        self.total_agents_static = int(sum(self.agent_counts_list))
        self.nb_agent_types = len(self.agent_counts_list)

        # Expected signature: (key, P0, T) -> (T+1,) trajectory
        self.get_P_trajectory = lambda key : generate_P_func(key,self.T)
        self.market_impact = market_impact_func   # Expected: (total_f) -> float
        self.generate_eps0 = generate_eps0_func   # Expected: (key, T) -> (T,) trajectory
        # New Signature: (key, sigma_eps, T) -> (T,) trajectory
        self.generate_idiosyncratic_noise = generate_eps_idiosyncratic_func
        

        # 3. Automatically derive Market Maximums from agent_params_list
        # agent_params mapping: [theta0(0), ef(1), eg(2), cf(3), cg(4), cap_f(5), cap_xi(6), sigma(7), tech(8)]
        self.max_cap_f = jnp.max(self.agent_params_list[:, 5])
        self.max_cap_xi = jnp.max(self.agent_params_list[:, 6])
        self.max_sigma_eps = jnp.max(self.agent_params_list[:, 7])

        self.policies = []

    @staticmethod
    def create_initial_state(agent_params, A0, P0,T):
        """
        Pure function: no 'self' allowed here if we want to vmap easily.
        """
        # [t, theta, xi_f, xi_g, eps, A, P, ef, eg, cf, cg, cap_f, cap_xi, sigma, tech]
        return jnp.concatenate([
            jnp.array([0.0, agent_params[0]*T, 0.0, 0.0, 0.0]), # Indices 0-4
            jnp.array([A0, P0]),                             # Indices 5-6
            agent_params[1:]                                 # Indices 7-14
        ])

    def get_market_initial_states(self, K):
            # Create one block of agents (N1 + N2 + ...)
            type_initial_states = jax.vmap(self.create_initial_state, in_axes=(0, None, None, None))(
                self.agent_params_list, self.A0, self.P0, self.T
            )
            
            # FIX: We use a standard Python list for repeats and 
            # provide total_repeat_length as a static integer.
            base_block = jnp.repeat(
                type_initial_states, 
                jnp.array(self.agent_counts_list), 
                axis=0, 
                total_repeat_length=int(sum(self.agent_counts_list))
            )
            
            # Repeat the entire block K times
            return jnp.tile(base_block, (K, 1))
    
    # def initialize_state_training(self, key, B):
    #     """
    #     Returns B initial states where each state's type is chosen randomly 
    #     from the available agent_params_list.
    #     """
    #     k1, k2 = jax.random.split(key)
        
    #     # 1. Randomly choose a type index for each of the B agents
    #     # We use jnp.arange(self.nb_agent_types) to pick from types 0, 1, ...
    #     type_indices = jax.random.choice(k1, jnp.arange(self.nb_agent_types), shape=(B,))
        
    #     # 2. Map the indices to the actual parameters
    #     batch_params = self.agent_params_list[type_indices]
        
    #     # 3. Generate initial states for this heterogeneous batch
    #     # We vmap over the chosen parameters
    #     X0 = jax.vmap(self.create_initial_state, in_axes=(0, None, None, None))(
    #         batch_params, self.A0, self.P0, self.T
    #     )
        
    #     return X0
    
    def initialize_state_training(self, key, B):
        """
        Samples B initial states by generating random agent parameters
        within a range derived from the agent_params_list.
        """
        k_params, k_init = jax.random.split(key)

        # 1. Identify the bounds for each parameter across your defined types
        # agent_params_list shape: (num_types, num_params)
        param_mins = jnp.min(self.agent_params_list, axis=0)
        param_maxs = jnp.max(self.agent_params_list, axis=0)

        # 2. Define the sampling range [min * 0.8, max * 1.2]
        # This creates a "buffer zone" around your theoretical types
        low_bounds = param_mins * 0.8
        high_bounds = param_maxs * 1.2

        # 3. Sample B sets of parameters uniformly
        # Shape: (B, num_params)
        batch_params = jax.random.uniform(
            k_params, 
            shape=(B, self.agent_params_list.shape[1]), 
            minval=low_bounds, 
            maxval=high_bounds
        )

        # 4. Generate initial states
        # Each agent now has a unique identity based on these sampled params
        X0 = jax.vmap(self.create_initial_state, in_axes=(0, None, None, None))(
            batch_params, self.A0, self.P0, self.T
        )

        return X0

    def initialize_state_training_gamma(self, key, B, gamma=0.0):
        """
        Samples B initial states.
        gamma = 1.0: Samples exactly from the types in agent_params_list.
        gamma = 0.0: Samples uniformly across the bounded parameter space.
        """
        k_type, k_noise = jax.random.split(key)

        # 1. Randomly assign each of the B agents to one of the defined types
        # This ensures that even with noise, the agents cluster around valid archetypes
        type_indices = jax.random.choice(k_type, jnp.arange(self.nb_agent_types), shape=(B,))
        base_params = self.agent_params_list[type_indices] # (B, num_params)

        # 2. Define the global boundaries (as you had before)
        param_mins = jnp.min(self.agent_params_list, axis=0) * 0.8
        param_maxs = jnp.max(self.agent_params_list, axis=0) * 1.2

        # 3. Generate uniform noise in the range [param_mins, param_maxs]
        random_params = jax.random.uniform(
            k_noise, 
            shape=(B, self.agent_params_list.shape[1]), 
            minval=param_mins, 
            maxval=param_maxs
        )

        # 4. Interpolate based on gamma
        # If gamma=1, we get base_params. If gamma=0, we get random_params.
        # Formula: Params = gamma * Base + (1 - gamma) * Random
        batch_params = gamma * base_params + (1.0 - gamma) * random_params

        # 5. Generate initial states
        X0 = jax.vmap(self.create_initial_state, in_axes=(0, None, None, None))(
            batch_params, self.A0, self.P0, self.T
        )

        return X0
    
    def compute_next_A(self, A_t, array_f, eps0_t):
        """
        Price Update Logic:
        A_{t+1} = A_t + mean(MarketImpact(f_n)) + eps0_t
        
        array_f: (Total_Agents,) array of individual trades
        """
        # Apply the impact function to every trade in the array
        individual_impacts = vmap(self.market_impact)(array_f)
        
        # The price shift is the average of these impacts
        mean_impact = jnp.mean(individual_impacts)
        
        return jnp.maximum(A_t + mean_impact*self.dt + eps0_t*math.sqrt(self.dt), self.Afloor)

    def get_eps0_trajectory(self, key):
        """Generates the exogenous volatility for the allowance price."""
        return self.generate_eps0(key, self.T)

    @staticmethod
    def instant_allowances(x):
        """x: (15,) -> [t, theta, xi_f, xi_g, eps, A, P, e1, e2, c1, c2, cap_f, cap_xi, sigma, tech]"""
        prod_value = x[2:4] @ x[7:9]
        return x[1] - prod_value

    @staticmethod
    def running_reward(x, a):
        """x: (15,), a: (3,)"""
        prod_coef = x[6] - x[9:11] 
        quantities = a[1] * jnp.array([a[2], 1.0 - a[2]])
        prod_revenue = jnp.dot(quantities, prod_coef)
        trade_revenue = -a[0] * x[5] 
        return prod_revenue + trade_revenue
    

    @staticmethod
    def running_reward_training(x, a, gamma):
        """
        x: (15,), a: (3,), gamma: float (penalty coefficient)
        """
        # --- Standard Market Reward ---
        prod_coef = x[6] - x[9:11] 
        quantities = a[1] * jnp.array([a[2], 1.0 - a[2]])
        prod_revenue = jnp.dot(quantities, prod_coef)
        trade_revenue = -a[0] * x[5]
        market_reward = prod_revenue + trade_revenue

        trading_cost = -gamma*a[0]**2

        return market_reward - trading_cost

    def terminal_reward(self, x):
        emissions = jnp.dot(x[2:4], x[7:9]) + x[4]
        allowances = x[1]
        return -self.kappa * jnp.clip(emissions - allowances, a_min=0.0)
    
    
    def terminal_reward_training(self, x, gamma):
        emissions = jnp.dot(x[2:4], x[7:9]) + x[4]
        allowances = x[1]
        
        shortage = jnp.clip(emissions - allowances, a_min=0.0)
        surplus = jnp.clip(allowances - emissions, a_min=0.0)
        
        # Shortage is usually a 'Hard' market penalty (kappa)
        # Surplus is your 'Training' penalty (gamma)
        return -self.kappa * shortage - gamma * surplus**2

    @staticmethod
    def single_step_dynamics(x, a, eps, next_A, next_P):
        t_next = x[0] + 1.0
        theta_next = x[1] + a[0]
        xi_next = x[2:4] + a[1] * jnp.array([a[2], 1.0 - a[2]])
        eps_next = x[4] + eps
        static_part = x[7:15]
        
        return jnp.concatenate([
            jnp.array([t_next, theta_next]),
            xi_next,
            jnp.array([eps_next]),
            jnp.array([next_A]),
            jnp.array([next_P]),
            static_part
        ])

    def normalize_state(self, x):
        """
        Normalizes a single state vector.
        Dynamic variables are scaled by time/capacity/price scales.
        Agent-specific capacities are scaled by market-wide maximums.
        """
        # Dynamic variables
        t_norm = x[0] / self.T
        theta_norm = x[1] / (self.T * x[11])    # Scale by agent's own T*cap_f
        xi_norm = x[2:4] / (self.T * x[12] + 1e-4)     # Scale by agent's own T*cap_xi
        eps_norm = x[4] / (jnp.sqrt(self.T) * x[13] + 1e-4) # Scale by agent's own T*sigma_eps
        A_norm = x[5] / self.A_scale
        P_norm = x[6] / self.P_scale
        
        # Agent-specific capacity parameters (Normalized by Market Max)
        cap_f_norm = x[11] / self.max_cap_f
        cap_xi_norm = x[12] / self.max_cap_xi
        sigma_eps_norm = x[13] / self.max_sigma_eps
        
        # Concatenate back: 
        # [t, theta, xi_f, xi_g, eps, A, P] (normalized)
        # [e1, e2, c1, c2] (unscaled features)
        # [cap_f, cap_xi, sigma] (normalized by market max)
        # [tech_control] (raw flag)
        return jnp.concatenate([
            jnp.array([t_norm, theta_norm]),
            xi_norm,
            jnp.array([eps_norm, A_norm, P_norm]),
            x[7:11],                            # e1, e2, c1, c2
            jnp.array([cap_f_norm, cap_xi_norm, sigma_eps_norm]),
            jnp.atleast_1d(x[14])               # tech_control
        ])

    @staticmethod
    def unnormalize_action(x, hat_a):
        # theta is x[1], cap_f is x[11], cap_xi is x[12]
        a0 = jnp.where(hat_a[0] < 0, hat_a[0] * x[1], hat_a[0] * x[11])
        a1 = hat_a[1] * x[12]
        return jnp.array([a0, a1, hat_a[2]])
    

    # def rollout_market(self, key, policies):
    #         T_int = int(self.T)
    #         K = len(policies)
    #         agents_per_block = sum(self.agent_counts_list) # N1 + N2
            
    #         k_p, k_eps0, k_agents = jax.random.split(key, 3)
            
    #         P_traj = self.get_P_trajectory(k_p)
    #         eps0_traj = self.generate_eps0(k_eps0, T_int)
            
    #         # X0 is now (K * agents_per_block, 15)
    #         X0 = self.get_market_initial_states(K) 
            
    #         agent_keys = jax.random.split(k_agents, K * agents_per_block)
    #         all_eps_ids = jax.vmap(self.generate_idiosyncratic_noise, in_axes=(0, 0, None))(
    #             agent_keys, X0[:, 13], T_int
    #         )

    #         def step_fn(carry, t_idx):
    #             current_X, current_A = carry
    #             all_actions = []
                
    #             # Loop through each policy
    #             for k in range(K):
    #                 # Slice the block of all agent types assigned to policy k
    #                 block_X = jax.lax.dynamic_slice(
    #                     current_X, 
    #                     (k * agents_per_block, 0), 
    #                     (agents_per_block, 15)
    #                 )
                    
    #                 # Apply policy_k to this entire block (all types)
    #                 norm_X = jax.vmap(self.normalize_state)(block_X)
    #                 raw_a = jax.vmap(policies[k])(norm_X) 
    #                 actions = jax.vmap(self.unnormalize_action)(block_X, raw_a)
                    
    #                 all_actions.append(actions)
                
    #             # Combine actions from ALL policies and ALL types
    #             actions_t = jnp.concatenate(all_actions, axis=0)
                
    #             # B. Endogenous Price Update (A)
    #             # A_t responds to the average trade of the whole multi-policy ensemble
    #             next_A = self.compute_next_A(current_A, actions_t[:, 0], eps0_traj[t_idx])
                
    #             # C. Agent Dynamics
    #             next_P = P_traj[t_idx + 1]
    #             next_X = jax.vmap(self.single_step_dynamics, in_axes=(0, 0, 0, None, None))(
    #                 current_X, actions_t, all_eps_ids[:, t_idx], next_A, next_P
    #             )
                
    #             return (next_X, next_A), (current_X, actions_t, current_A)

    #         last_state, trajectory = jax.lax.scan(step_fn, (X0, self.A0), jnp.arange(T_int))
    #         return trajectory

    def rollout_market(self, key, policies):
        T_int = int(self.T)
        K = len(policies)
        total_agents = K * self.total_agents_static
        
        k_p, k_eps0, k_agents = jax.random.split(key, 3)
        
        P_traj = self.get_P_trajectory(k_p)
        eps0_traj = self.generate_eps0(k_eps0, T_int)
        
        X0 = self.get_market_initial_states(K) 
        
        agent_keys = jax.random.split(k_agents, total_agents)
        all_eps_ids = jax.vmap(self.generate_idiosyncratic_noise, in_axes=(0, 0, None))(
            agent_keys, X0[:, 13], T_int
        )

        # Initial carry now includes a reward accumulator of shape (total_agents,)
        initial_carry = (X0, self.A0, jnp.zeros(total_agents))

        def step_fn(carry, t_idx):
            current_X, current_A, current_reward = carry
            all_actions = []
            
            for k in range(K):
                block_X = jax.lax.dynamic_slice(
                    current_X, 
                    (k * self.total_agents_static, 0), 
                    (self.total_agents_static, 15)
                )
                
                norm_X = jax.vmap(self.normalize_state)(block_X)
                raw_a = jax.vmap(policies[k])(norm_X) 
                actions = jax.vmap(self.unnormalize_action)(block_X, raw_a)
                all_actions.append(actions)
            
            actions_t = jnp.concatenate(all_actions, axis=0)
            
            # --- NEW: Compute running rewards ---
            # running_reward is vmapped over all agents in the ensemble
            step_rewards = jax.vmap(self.running_reward)(current_X, actions_t)
            new_cumulative_reward = current_reward + step_rewards
            
            next_A = self.compute_next_A(current_A, actions_t[:, 0], eps0_traj[t_idx])
            next_P = P_traj[t_idx + 1]
            
            next_X = jax.vmap(self.single_step_dynamics, in_axes=(0, 0, 0, None, None))(
                current_X, actions_t, all_eps_ids[:, t_idx], next_A, next_P
            )
            
            return (next_X, next_A, new_cumulative_reward), (current_X, actions_t, current_A)

        # Run the scan
        (last_X, last_A, total_running_reward), trajectory = jax.lax.scan(
            step_fn, initial_carry, jnp.arange(T_int)
        )
        
        # --- NEW: Compute terminal rewards ---
        # terminal_reward is added to the final accumulated running rewards
        # print(jax.vmap(self.terminal_reward)(last_X))
        # print(total_running_reward)
        final_rewards = total_running_reward + jax.vmap(self.terminal_reward)(last_X)
        
        # Unpack trajectory for consistency
        states_traj, actions_traj, A_traj = trajectory
        
        # We append the final state and price to the trajectory to match T+1 length
        full_states = jnp.concatenate([states_traj, last_X[None, ...]], axis=0)
        full_A = jnp.append(A_traj, last_A)
        
        return full_states, actions_traj, full_A, final_rewards
    

    def generate_A(self, key):
        _, _, A,_ = self.rollout_market(key, self.policies)
        return A


