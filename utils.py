
from envs.environnment import *
import envs.environnment
import matplotlib.pyplot as plt
from envs.models import *
import envs.models
import optax
import seaborn as sns
import pandas as pd


def generate_report_jax(env, states, actions, reward=None, save=False, save_title='report.png'):
    T_plus_1, total_agents, _ = states.shape
    T = T_plus_1 - 1
    K = total_agents//env.total_agents_static
    
    # Pre-calculate type labels
    type_labels_list = []
    for type_idx, count in enumerate(env.agent_counts_list):
        type_labels_list.extend([f"Type {type_idx}"] * int(count))

    type_labels_list = type_labels_list * K
    
    all_data = []
    for k in range(total_agents):
        # 1. Ensure we convert the JAX slice to a standard Numpy array
        # 2. Append a single np.nan to the end to reach length T+1
        f_padded = np.append(np.array(actions[:, k, 0]), np.nan)
        xi_padded = np.append(np.array(actions[:, k, 1]), np.nan)
        eta_padded = np.append(np.array(actions[:, k, 2]), np.nan)
                            

        # 2. Build the dictionary. 
        # Every entry MUST have length T_plus_1
        d = {
            "t": np.arange(T_plus_1),
            r"$\theta$": np.array(states[:, k, 1]),
            r"$\xi^f$": np.array(states[:, k, 2]),
            r"$\xi^g$": np.array(states[:, k, 3]),
            "A": np.array(states[:,k,5]),
            "P": np.array(states[:, k, 6]),
            "f": f_padded,
            r"$\xi$": xi_padded,
            r"$\eta$": eta_padded,
            "idx_agent": [k] * T_plus_1,
            "Type": [type_labels_list[k]] * T_plus_1,  # Broadcaster to full length, 
            r"$\bar\epsilon$": np.array(states[:, k, 4])
        }
        # return d
        
        # Create DF for this agent and add to list
        # return d
        all_data.append(pd.DataFrame(d))

    # Combine all agents into one panel
    data = pd.concat(all_data, ignore_index=True)

    # --- Plotting ---
    columns_to_plot = [r"$\theta$", r"$\xi^f$", r"$\xi^g$",r"$\bar\epsilon$", "A", "P",'f',  r"$\eta$"]
    n_cols = 3
    n_rows = (len(columns_to_plot) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()
    palette = sns.color_palette("husl", env.nb_agent_types)

    for i, col in enumerate(columns_to_plot):
        ax = axes[i]
        sns.lineplot(
            data=data, x="t", y=col, hue="Type", units="idx_agent", 
            estimator=None, palette=palette, alpha=0.4, ax=ax, legend=(i == 0)
        )
        ax.set_title(col, fontsize=14)
        
        # Consistent Scaling
        if col == 'f': ax.set_ylim(top = float(env.max_cap_f) * 1.5)
        if col == r"$\eta$": ax.set_ylim(-0.05, 1.05)
        if col in ["A", "P", r"$\theta$"]: ax.set_ylim(bottom=0)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    avg_reward_str = f"{reward.mean():.2e}" if reward is not None else "N/A"

    params_str = (
    f"$A_{{floor}}$: {env.Afloor:.1f} | "
    f"$\kappa$: {env.kappa:.2e} | "
    )

    plt.suptitle(
        f"Market Universe Report | Agents: {total_agents}\n"
        f"{params_str} | Avg Reward: {avg_reward_str}", 
        fontsize=16, y=0.98
    )

    # plt.suptitle(f"Market Universe Report | Agents: {total_agents} | T: {T} | Avg Reward: {avg_reward_str}", fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if save:
        plt.savefig(save_title, dpi=150)
    plt.show()

    return data


# def train_single_nn_core(env, model, optimizer, opt_state, nb_iterations, lr, batch_size, key):
#     model_params, model_static = eqx.partition(model, eqx.is_array)

#     def loss_fn(params, step_key):
#         active_model = eqx.combine(params, model_static)
#         k_init, k_p, k_eps, k_A = jax.random.split(step_key, 4)
        
#         # Sample heterogeneous batch
#         X0 = env.initialize_state_training(k_init, batch_size)
#         P_traj = env.get_P_trajectory(k_p)
#         # A_traj = env.generate_A(k_A, [active_model]) 
#         A_traj = env.generate_A(k_A)

#         def step_fn(carry, t_idx):
#             X_t, cum_reward = carry
#             norm_X = jax.vmap(env.normalize_state)(X_t)
#             a_t = jax.vmap(env.unnormalize_action)(X_t, jax.vmap(active_model)(norm_X))
            
#             r_t = jax.vmap(env.running_reward)(X_t, a_t)
#             next_X = jax.vmap(env.single_step_dynamics, in_axes=(0, 0, 0, None, None))(
#                 X_t, a_t, 
#                 jax.vmap(env.generate_idiosyncratic_noise, in_axes=(0, 0, None))(
#                     jax.random.split(k_eps, batch_size), X0[:, 13], int(env.T)
#                 )[:, t_idx], 
#                 A_traj[t_idx+1], P_traj[t_idx+1]
#             )
#             return (next_X, cum_reward + r_t), None

#         (X_final, total_reward), _ = jax.lax.scan(step_fn, (X0, jnp.zeros(batch_size)), jnp.arange(int(env.T)))
#         total_reward += jax.vmap(env.terminal_reward)(X_final)
#         return -jnp.mean(total_reward) / (env.T * 1e5)

#     @jax.jit
#     def update_step(params, state, step_key, current_lr):
#         loss_val, grads = jax.value_and_grad(loss_fn)(params, step_key)
        
#         # We manually scale the updates by the current horizon's LR
#         # This is more robust than re-initializing the optimizer
#         updates, new_state = optimizer.update(grads, state, params)
#         scaled_updates = jax.tree_util.tree_map(lambda g: g * (current_lr / lr), updates)
        
#         new_params = eqx.apply_updates(params, scaled_updates)
#         return new_params, new_state, loss_val

#     curr_params = model_params
#     losses = []
#     for i in range(nb_iterations):
#         step_key = jax.random.fold_in(key, i)
#         curr_params, opt_state, loss_val = update_step(curr_params, opt_state, step_key, lr)
#         losses.append(loss_val)
        
#     return eqx.combine(curr_params, model_static), opt_state, losses


def train_single_nn_core(env, model, optimizer, opt_state, nb_iterations, lr, batch_size, key, use_curriculum=True):
    model_params, model_static = eqx.partition(model, eqx.is_array)
    T_int = int(env.T)

    def loss_fn(params, step_key, current_gamma):
        active_model = eqx.combine(params, model_static)
        k_init, k_p, k_eps, k_A = jax.random.split(step_key, 4)
        
        # Pass the dynamic linear gamma to the sampler
        X0 = env.initialize_state_training_gamma(k_init, batch_size, gamma=current_gamma)
        
        P_traj = env.get_P_trajectory(k_p)
        A_traj = env.generate_A(k_A)

        agent_keys = jax.random.split(k_eps, batch_size)
        noise_traj = jax.vmap(env.generate_idiosyncratic_noise, in_axes=(0, 0, None))(
            agent_keys, X0[:, 13], T_int
        )

        def step_fn(carry, t_idx):
            X_t, cum_reward = carry
            norm_X = jax.vmap(env.normalize_state)(X_t)
            raw_a = jax.vmap(active_model)(norm_X)
            a_t = jax.vmap(env.unnormalize_action)(X_t, raw_a)
            
            r_t = jax.vmap(env.running_reward)(X_t, a_t)
            
            next_X = jax.vmap(env.single_step_dynamics, in_axes=(0, 0, 0, None, None))(
                X_t, a_t, noise_traj[:, t_idx], A_traj[t_idx+1], P_traj[t_idx+1]
            )
            return (next_X, cum_reward + r_t), None

        (X_final, total_reward), _ = jax.lax.scan(step_fn, (X0, jnp.zeros(batch_size)), jnp.arange(T_int))
        total_reward += jax.vmap(env.terminal_reward)(X_final)
        
        return -jnp.mean(total_reward) / (env.T * 1e5)

    @jax.jit
    def train_loop(params, state, loop_key):
        def one_iteration(carry, i):
            p, s = carry
            step_key = jax.random.fold_in(loop_key, i)
            
            # --- Linear Ramp Logic ---
            # i goes from 0 to (nb_iterations - 1)
            # current_gamma moves from 0.0 to ~1.0
            if use_curriculum:
                current_gamma = i / (nb_iterations - 1.0)
            else:
                current_gamma = 1.0
            
            loss_val, grads = jax.value_and_grad(loss_fn)(p, step_key, current_gamma)
            
            updates, new_s = optimizer.update(grads, s, p)
            new_p = eqx.apply_updates(p, updates)
            
            return (new_p, new_s), (loss_val, current_gamma)

        (final_p, final_s), (losses, gammas) = jax.lax.scan(one_iteration, (params, state), jnp.arange(nb_iterations))
        return final_p, final_s, losses, gammas

    curr_params, opt_state, losses, gammas = train_loop(model_params, opt_state, key)
    
    return eqx.combine(curr_params, model_static), opt_state, losses


def train_multigrid_horizon(env, config, key):
    """
    Trains a single NN over a sequence of increasing time horizons, 
    maintaining the optimizer state (momentum, etc.) throughout.
    """
    k_model, k_train = jax.random.split(key)
    
    # 1. Initialize Model and Optimizer ONCE
    model = ActionNN(state_dim=15, key=k_model)
    
    # Start with the first LR in the config
    optimizer = optax.adam(config['list_lr'][0])
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    master_loss_history = []

    # 2. Loop through horizons
    for i, T_horizon in enumerate(config['list_T']):
        print(f"\n--- Training Horizon T = {T_horizon} ---")
        
        # Update Environment Horizon
        env.T = float(T_horizon)
        
        # Update Learning Rate in the existing opt_state
        # This keeps the Adam 'm' and 'v' buffers but changes the step size
        new_lr = config['list_lr'][i]
        opt_state = jax.tree_util.tree_map(
            lambda x: x if not isinstance(x, jnp.ndarray) else x, 
            opt_state
        ) # Ensure state is ready for update
        
        # 3. Train for this horizon
        # We pass the model and the opt_state to resume exactly where we left off
        if i == 0:
            use_curriculum=True
        else: 
            use_curriculum=False
        model, opt_state, losses = train_single_nn_core(
            env=env,
            model=model,
            optimizer=optimizer,
            opt_state=opt_state,
            nb_iterations=config['list_iterations'][i],
            lr=new_lr, # Passed to update_step to scale gradients
            batch_size=config['batch_size'],
            key=jax.random.fold_in(k_train, i), 
            use_curriculum=use_curriculum
        )
        
        master_loss_history.append(losses)
        print(f"Final Loss for T={T_horizon}: {losses[-1]:.6f}")

    # plt.plot(jnp.concatenate(master_loss_history))

    return model, master_loss_history


# def train_social_optimum_core(env, model, optimizer, opt_state, nb_iterations, key):
#     model_params, model_static = eqx.partition(model, eqx.is_array)
#     T_int = int(env.T)
    
#     # Use the actual number of agents defined in your environment
#     total_nb_agents = int(jnp.sum(jnp.array(env.agent_counts_list)))

#     def loss_fn(params, step_key):
#         active_model = eqx.combine(params, model_static)
#         k_init, k_p, k_eps, k_eps0 = jax.random.split(step_key, 4)
        
#         # 1. Initialize the exact number of agents in the economy
#         X0 = env.initialize_state_training(k_init, total_nb_agents)
        
#         # 2. Market trajectories
#         P_traj = env.get_P_trajectory(k_p)     
#         eps0_traj = env.generate_eps0(k_eps0, T_int) 

#         # 3. Individual agent noise
#         agent_keys = jax.random.split(k_eps, total_nb_agents)
#         noise_traj = jax.vmap(env.generate_idiosyncratic_noise, in_axes=(0, 0, None))(
#             agent_keys, X0[:, 13], T_int
#         )

#         def step_fn(carry, t_idx):
#             X_t, A_t, cum_reward = carry
            
#             # Update all agents with current endogenous price A_t
#             X_t = X_t.at[:, 5].set(A_t)
            
#             # All agents follow the SAME policy (Social Planner's perspective)
#             norm_X = jax.vmap(env.normalize_state)(X_t)
#             raw_a = jax.vmap(active_model)(norm_X)
#             a_t = jax.vmap(env.unnormalize_action)(X_t, raw_a)
            
#             # Price Discovery: Compute impact based on the WHOLE economy's trades
#             array_f = a_t[:, 0] 
#             A_next = env.compute_next_A(A_t, array_f, eps0_traj[t_idx])
            
#             # Rewards
#             r_t = jax.vmap(env.running_reward)(X_t, a_t)
            
#             # Step dynamics with the updated price
#             next_X = jax.vmap(env.single_step_dynamics, in_axes=(0, 0, 0, None, None))(
#                 X_t, a_t, noise_traj[:, t_idx], A_next, P_traj[t_idx+1]
#             )
            
#             return (next_X, A_next, cum_reward + r_t), None

#         # Execute trajectory
#         (X_final, A_final, total_reward), _ = jax.lax.scan(
#             step_fn, 
#             (X0, env.A0, jnp.zeros(total_nb_agents)), 
#             jnp.arange(T_int)
#         )
        
#         total_reward += jax.vmap(env.terminal_reward)(X_final)
        
#         # Social Optimum: Maximize the average utility of the entire population
#         return -jnp.mean(total_reward) / (env.T * 1e5)

#     @jax.jit
#     def train_loop(params, state, loop_key):
#         def one_iteration(carry, i):
#             p, s = carry
#             step_key = jax.random.fold_in(loop_key, i)
#             loss_val, grads = jax.value_and_grad(loss_fn)(p, step_key)
#             updates, new_s = optimizer.update(grads, s, p)
#             new_p = eqx.apply_updates(p, updates)
#             return (new_p, new_s), loss_val

#         (final_p, final_s), losses = jax.lax.scan(one_iteration, (params, state), jnp.arange(nb_iterations))
#         return final_p, final_s, losses

#     curr_params, opt_state, losses = train_loop(model_params, opt_state, key)
#     return eqx.combine(curr_params, model_static), opt_state, losses


def train_social_optimum_core(env, model, optimizer, opt_state, nb_iterations, key, use_curriculum=True):
    model_params, model_static = eqx.partition(model, eqx.is_array)
    T_int = int(env.T)
    
    total_nb_agents = int(jnp.sum(jnp.array(env.agent_counts_list)))

    # Loss function now accepts the dynamic gamma
    def loss_fn(params, step_key, current_gamma):
        active_model = eqx.combine(params, model_static)
        k_init, k_p, k_eps, k_eps0 = jax.random.split(step_key, 4)
        
        # 1. Initialize with dynamic gamma
        X0 = env.initialize_state_training_gamma(k_init, total_nb_agents, gamma=current_gamma)
        
        P_traj = env.get_P_trajectory(k_p)     
        eps0_traj = env.generate_eps0(k_eps0, T_int) 

        agent_keys = jax.random.split(k_eps, total_nb_agents)
        noise_traj = jax.vmap(env.generate_idiosyncratic_noise, in_axes=(0, 0, None))(
            agent_keys, X0[:, 13], T_int
        )

        def step_fn(carry, t_idx):
            X_t, A_t, cum_reward = carry
            
            # Update all agents with current price
            X_t = X_t.at[:, 5].set(A_t)
            
            norm_X = jax.vmap(env.normalize_state)(X_t)
            raw_a = jax.vmap(active_model)(norm_X)
            a_t = jax.vmap(env.unnormalize_action)(X_t, raw_a)
            
            # Total trade impact
            array_f = a_t[:, 0] 
            A_next = env.compute_next_A(A_t, array_f, eps0_traj[t_idx])
            
            r_t = jax.vmap(env.running_reward)(X_t, a_t)
            
            next_X = jax.vmap(env.single_step_dynamics, in_axes=(0, 0, 0, None, None))(
                X_t, a_t, noise_traj[:, t_idx], A_next, P_traj[t_idx+1]
            )
            
            return (next_X, A_next, cum_reward + r_t), None

        (X_final, A_final, total_reward), _ = jax.lax.scan(
            step_fn, 
            (X0, env.A0, jnp.zeros(total_nb_agents)), 
            jnp.arange(T_int)
        )
        
        total_reward += jax.vmap(env.terminal_reward)(X_final)
        return -jnp.mean(total_reward) / (env.T * 1e5)

    @jax.jit
    def train_loop(params, state, loop_key):
        def one_iteration(carry, i):
            p, s = carry
            step_key = jax.random.fold_in(loop_key, i)
            
            # --- Linear Gamma Ramp ---
            # Moves from 0.0 at i=0 to 1.0 at i=max
            if use_curriculum:
                current_gamma = i / jnp.maximum(nb_iterations - 1.0, 1.0)
            else:
                current_gamma = 1.0
                
            loss_val, grads = jax.value_and_grad(loss_fn)(p, step_key, current_gamma)
            updates, new_s = optimizer.update(grads, s, p)
            new_p = eqx.apply_updates(p, updates)
            
            return (new_p, new_s), (loss_val, current_gamma)

        (final_p, final_s), (losses, gammas) = jax.lax.scan(one_iteration, (params, state), jnp.arange(nb_iterations))
        return final_p, final_s, losses, gammas

    curr_params, opt_state, losses, gammas = train_loop(model_params, opt_state, key)
    return eqx.combine(curr_params, model_static), opt_state, losses


def fictitious_play(env, config, nb_policy, key, plot_report=True):
    """
    nb_players: Number of iterations of Fictitious Play (total NNs in ensemble)
    config: The multigrid configuration (list_T, list_lr, etc.)
    """
    k_fp = key
    # Start with an empty list of policies
    # The first player will train against the environment's default logic 
    # (or a random policy if env.policies is empty)
    if len(env.policies) == 0:
        k_fp, k_init = jax.random.split(k_fp)
        # env.policies = [ActionNN(state_dim=15, key=k_init)]


        model = ActionNN(state_dim=15, key=k_init)
        optimizer = optax.adam(1e-4)
        opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

        env.T = 30
        model, opt_state, losses = train_social_optimum_core(
            env=env,
            model=model,
            optimizer=optimizer,
            opt_state=opt_state,
            nb_iterations=2000,
            key=jax.random.PRNGKey(11), 
            use_curriculum=True
        )

        env.policies.append(model)
    
    for p in range(nb_policy):
        print(f"\n{'='*30}")
        print(f"FICTITIOUS PLAY ITERATION: {p+1}/{nb_policy}")
        print(f"{'='*30}")
        
        # 1. Train the current player using Multigrid Horizon
        k_train, k_fp = jax.random.split(k_fp)
        
        # Train the new policy
        new_policy, _ = train_multigrid_horizon(env, config, k_train)
        
        # 2. Add the new policy to the ensemble
        env.policies.append(new_policy)
        
        # 3. Optional: Plot a report of the market with the latest player
        if plot_report:
            print(f"\nGenerating Market Report for Iteration {p+1}...")
            # Use the final T from the curriculum for the report
            env.T = float(config['list_T'][-1])
            k_rollout, k_fp = jax.random.split(k_fp)
            
            # Rollout one universe
            states, actions, A_history, rewards = env.rollout_market(k_rollout, env.policies)
            
            # Generate the panel
            generate_report_jax(
                env, 
                states, 
                actions, 
                rewards, 
                save_title=f"fp_iteration_{p+1}.png"
            )


from scipy.stats import gaussian_kde

# def plot_normalized_kde(data, ax, label, color):
#     # 1. Calculate the KDE
#     kde = gaussian_kde(data)
    
#     # 2. Create an x-axis range for plotting
#     min_data = min(data)
#     max_data = max(data)
#     if min_data < 0: 
#         min_x = min_data*1.1 
#     else:
#         min_x = min_data*0.9 
#     if max_data < 0: 
#         max_x = max_data*0.9 
#     else:
#         max_x = max_data*1.1 
#     x_range = np.linspace(min_x, max_x, 500)
#     y_values = kde(x_range)
    
#     # 3. Normalize: Divide by the maximum value to force peak to 1
#     y_norm = y_values / np.max(y_values)
    
#     # 4. Plot using standard matplotlib (looks just like Seaborn)
#     ax.plot(x_range, y_norm, color=color, lw=1)
#     ax.fill_between(x_range, y_norm, color=color, alpha=0.3, label = label)
#     ax.set_ylim(0, 1.05) # Keeps the peak visible at the top


def plot_normalized_kde(data, ax, label, color, fill = True):
    # Check if data is constant to avoid KDE errors
    if np.all(data == data[0]):
        ax.axvline(data[0], color=color, lw=2, label=label)
        return
    
    # 2. Create a smart x-axis range
    std_data = np.std(data)
    # If variance is very low, force a window so we can see the spike
    if std_data < 1e-3: 
        margin = max(abs(np.mean(data)) * 0.1, 1.0)
    else:
        margin = std_data * 3  # Show 3 standard deviations

    # 1. Calculate the KDE
    kde = gaussian_kde(data)
        
    min_x = np.min(data) - margin
    max_x = np.max(data) + margin
    
    x_range = np.linspace(min_x, max_x, 1000) # Increased resolution
    y_values = kde(x_range)
    
    # 3. Normalize
    max_y = np.max(y_values)
    if max_y > 0:
        y_norm = y_values / max_y
    else:
        y_norm = y_values

    # 4. Plot
    if fill:
        ax.plot(x_range, y_norm, color=color, lw=1, linestyle = '-')
        ax.fill_between(x_range, y_norm, color=color, alpha=0.3, label=label)
    else: 
        ax.plot(x_range, y_norm, color=color, lw=1, linestyle = 'dashdot', label = label)
    
    ax.set_ylim(0, 1.05)

def plot_unit_spike(ax, x_coord, color, label, linestyle = '-'):
    """
    Plots a vertical line at x_coord that starts at 0 
    and ends exactly at 1 on the y-axis.
    """
    ax.vlines(x=x_coord, ymin=0, ymax=1, color=color, 
              lw=2, linestyle=linestyle, label=label)
    
    # Optional: ensure the y-axis shows the full height
    ax.set_ylim(0, 1.05)


# def plot_densities_jax(env, states, actions, rewards):
#     B, T_plus_1, N_total, _ = states.shape
#     T = T_plus_1 - 1
#     K = N_total // env.total_agents_static
#     to_np = lambda x: np.array(x)
    
#     # Define colors for different types
#     # palette = sns.color_palette("husl", len(env.agent_counts_list))
#     palette = sns.color_palette("muted", len(env.agent_counts_list))

#     fig, axes = plt.subplots(3, 3, figsize=(18, 10))
#     axes = axes.flatten()

#     # --- 1. GLOBAL METRICS (Independent of Type) ---

#     # Plot 1: Allowance Price
#     A_samples = to_np(states[:, :, 0, 5]).flatten()
#     # sns.kdeplot(A_samples, fill=True, ax=axes[0])
#     plot_normalized_kde(A_samples, axes[0], '', '#1f77b4')
#     if hasattr(env, 'Afloor'):
#         axes[0].axvline(env.Afloor, linestyle='--', label='Floor', color = 'black')
#     axes[0].axvline(np.mean(A_samples), color='red', label=f'Mean: ${np.mean(A_samples):.1f}')
#     axes[0].set_title("Market Allowance Prices ($)")
#     axes[0].legend()

#     # Plot 2: Churn Rate
#     trading_vol = np.abs(to_np(actions[..., 0])).sum(axis=1) 
#     init_q = to_np(states[:, 0, :, 1])
#     churn = (trading_vol / (init_q + 1e-9)).mean(axis=1) 
#     # sns.kdeplot(churn, fill=True, ax=axes[1])
#     plot_normalized_kde(churn, axes[1], '', '#1f77b4')
#     axes[1].axvline(np.mean(churn), color='red', label=f'Mean: {np.mean(churn):.3f}')
#     axes[1].set_title("Churn Rate")
#     axes[1].legend()

#     # A_path = to_np(states[:, :, 0, 5]) # (B, T+1)
#     # returns = np.diff(A_path, axis=1) / (A_path[:, :-1] + 1e-9)
#     # volatility = returns.std(axis=1)
#     # sns.kdeplot(volatility, fill=True, ax=axes[6],label = 'Percentage Return')
#     # perc_mean = np.mean(volatility)

#     # Plot 3: Percentage Returns (Volatility)
#     A_path = to_np(states[:, :, 0, 5])
#     returns = np.diff(A_path, axis=1) / (A_path[:, :-1] + 1e-9)
#     volatility = returns.std(axis=1)
#     # sns.kdeplot(volatility, fill=True, ax=axes[2])
#     plot_normalized_kde(volatility, axes[2], '', '#1f77b4')
#     axes[2].axvline(np.mean(volatility), color='red', label=f'Mean: {np.mean(volatility):.3f}')
#     axes[2].set_title("Percentage Return")
#     axes[2].legend()

#     # Plot 4: HHI
#     agent_vol = np.abs(to_np(actions[..., 0])).sum(axis=1) 
#     mkt_share = agent_vol / (agent_vol.sum(axis=1, keepdims=True) + 1e-9)
#     hhi = np.sum(mkt_share**2, axis=1)
#     # sns.kdeplot(hhi, fill=True, ax=axes[3])
#     plot_normalized_kde(hhi, axes[3], '', '#1f77b4')
#     mean_hhi = np.mean(hhi)
#     axes[3].axvline(mean_hhi, color='red', label=f'Mean: {mean_hhi:.4f}')
#     axes[3].axvline(1/N_total, color='black',linestyle = '--', label=f'Minimum Theoretical HHI: {(1/N_total):.4f}')
#     # axes[3].set_xlim(0, 1)
#     axes[3].set_xlim(0, 2*max(hhi))
#     axes[3].set_title("Herfindahl-Hirschman Index")
#     axes[3].legend()

#     # Plot 5: Terminal Liquidity
#     last_steps = max(1, int(T * 0.1))
#     total_v = np.abs(to_np(actions[..., 0])).sum(axis=(1, 2))
#     late_v = np.abs(to_np(actions[:, -last_steps:, :, 0])).sum(axis=(1, 2))
#     liq_share = (late_v / (total_v + 1e-9)) * 100
#     # sns.kdeplot(liq_share, fill=True, ax=axes[4])
#     plot_normalized_kde(liq_share, axes[4], '', '#1f77b4')
#     mean_liq = np.mean(liq_share)
#     axes[4].axvline(mean_liq, color='red', label=f'Mean: {mean_liq:.1f}')
#     axes[4].set_title("Terminal Liquidity Share (%)")
#     axes[4].legend()

#     # --- 2. AGENT-SPECIFIC METRICS (Split by Type) ---
    
#     agent_metrics_axes = [5, 6, 7, 8]
#     titles = ["Terminal Inventory of Allowances (k)", "Cumulative Production per Technology (GHW)", "Net Compliance Balance (k)", "Reward ($M)"]
    
#     start_idx = 0
#     for type_idx, count in enumerate(env.agent_counts_list):
#         end_idx = start_idx + int(count)
#         c = palette[type_idx]
#         label_prefix = f"Type {type_idx+1}"
#         if len(env.agent_counts_list) == 1:
#             c = '#1f77b4'
#             label_prefix = ""

#         # Slicing data for this type
#         type_states = states[:, :, start_idx:end_idx, :]
#         type_actions = actions[:, :, start_idx:end_idx, :]
#         type_rewards = rewards[:, start_idx:end_idx]
        
#         # Metric 1: Terminal Inventory (q)
#         q_final = to_np(type_states[:, -1, :, 1]).flatten()
#         q_init = type_states[0, 0, 0, 1]/1e3
#         # sns.kdeplot(q_final, ax=axes[5], color=c, fill=True, label=f"{label_prefix} Mean: {np.mean(q_final):.1f}")
#         plot_normalized_kde(q_final/1e3, axes[5],f"{label_prefix}, Mean: {np.mean(q_final/1e3):.1f}k" ,c)
#         axes[5].axvline(q_init, color=c, linestyle='--', label = f"{label_prefix}, Initial: {q_init:.1f}k")

#         # # Metric 2: Production (Fuel + Green)
#         # xf = to_np(type_states[:, -1, :, 2]).flatten()
#         # xg = to_np(type_states[:, -1, :, 3]).flatten()
#         # sns.kdeplot(xf + xg, ax=axes[6], color=c, fill=True, label=f"{label_prefix} (Mean:{np.mean(xf+xg):.1f})")

#         # Metric 2: Production (Fuel vs Green)
#         xf = to_np(type_states[:, -1, :, 2]).flatten()/1e3
#         xg = to_np(type_states[:, -1, :, 3]).flatten()/1e3

#         # Fuel Production: Solid with Fill
#         # sns.kdeplot(xf, ax=axes[6], color=c, fill=True, linestyle='-', 
#         #             label=f"{label_prefix} Fuel, Mean: {np.mean(xf):.1f}MHW")
#         if xf.std() < 1e-4:
#             plot_unit_spike(axes[6], xf.mean(), c, label = f"{label_prefix}, Fuel, Mean: {np.mean(xf):.1f}GHW", linestyle= '--')
#         else: 
#             plot_normalized_kde(xf, axes[6],f"{label_prefix}, Fuel, Mean: {np.mean(xf):.1f}GHW" ,c)

#         # Green Production: Dashed Line (No Fill to avoid overlapping mess)
#         # sns.kdeplot(xg, ax=axes[6], color=c, fill=True, linestyle='-', hatch='///', 
#         #             label=f"{label_prefix} Green, Mean: {np.mean(xg):.1f}MHW")
#         if xg.std() < 1e-4:
#             plot_unit_spike(axes[6], xg.mean(), get_flashy_color(c), label = f"{label_prefix}, Green, Mean: {np.mean(xg):.1f}GHW")
#         else: 
#             plot_normalized_kde(xg, axes[6],f"{label_prefix}, Green, Mean: {np.mean(xg):.1f}GHW" ,get_flashy_color(c))

#         # Metric 3: Net Compliance Balance
#         # q - (xi_f * ef + xi_g * eg)
#         ef = env.agent_params_list[start_idx, 1]
#         eg = env.agent_params_list[start_idx, 2]
#         balance = (to_np(type_states[:, -1, :, 1]) - (to_np(type_states[:, -1, :, 2])*ef + to_np(type_states[:, -1, :, 3])*eg)).flatten()/1e3
#         # sns.kdeplot(balance, ax=axes[7], color=c, fill=True, label=f"{label_prefix} Mean: {np.mean(balance):.1f}")
#         plot_normalized_kde(balance, axes[7],f"{label_prefix} Mean: {np.mean(balance):.1f}k" ,c)

#         # Metric 4: Rewards
#         rew = to_np(type_rewards / 1e6).flatten()
#         # sns.kdeplot(rew, ax=axes[8], color=c, fill=True, label=f"{label_prefix} Mean: ${np.mean(rew):.2f}M")
#         plot_normalized_kde(rew, axes[8],f"{label_prefix} Mean: ${np.mean(rew):.2f}M" ,c)

#         start_idx = end_idx

#     # Vertical line for balance at 0
#     axes[7].axvline(0, color='black', linestyle='--', label = 'x = 0')

#     # Final Formatting
#     for i in range(5, 9):
#         axes[i].set_title(titles[i-5])
#         axes[i].legend(fontsize='small')
    


#     plt.tight_layout()
#     plt.show()

def plot_densities_jax(env, states, actions, rewards):
    # states shape: (B, T+1, N_total, 15)
    # actions shape: (B, T, N_total, 3)
    # rewards shape: (B, N_total)
    B, T_plus_1, total_agents, _ = states.shape 
    T = T_plus_1 - 1
    K = total_agents // env.total_agents_static
    to_np = lambda x: np.array(x)
    
    palette = sns.color_palette("muted", len(env.agent_counts_list))
    fig, axes = plt.subplots(3, 3, figsize=(18, 10))
    axes = axes.flatten()

     # --- 1. GLOBAL METRICS (Independent of Type) ---

    # Plot 1: Allowance Price
    A_samples = to_np(states[:, :, 0, 5]).flatten()
    # sns.kdeplot(A_samples, fill=True, ax=axes[0])
    plot_normalized_kde(A_samples, axes[0], '', '#1f77b4')
    if hasattr(env, 'Afloor'):
        axes[0].axvline(env.Afloor, linestyle='--', label='Floor', color = 'black')
    axes[0].axvline(A_samples[0], linestyle='--', label='Initial Price', color = 'blue')
    axes[0].axvline(np.mean(A_samples), color='red', label=f'Mean: ${np.mean(A_samples):.1f}')
    axes[0].set_title("Market Allowance Prices ($)")
    axes[0].legend()

    # Plot 2: Churn Rate
    trading_vol = np.abs(to_np(actions[..., 0])).sum(axis=1) 
    init_q = to_np(states[:, 0, :, 1])
    churn = (trading_vol / (init_q + 1e-9)).mean(axis=1) 
    # sns.kdeplot(churn, fill=True, ax=axes[1])
    plot_normalized_kde(churn, axes[1], '', '#1f77b4')
    axes[1].axvline(np.mean(churn), color='red', label=f'Mean: {np.mean(churn):.3f}')
    axes[1].set_title("Churn Rate")
    axes[1].legend()

    # A_path = to_np(states[:, :, 0, 5]) # (B, T+1)
    # returns = np.diff(A_path, axis=1) / (A_path[:, :-1] + 1e-9)
    # volatility = returns.std(axis=1)
    # sns.kdeplot(volatility, fill=True, ax=axes[6],label = 'Percentage Return')
    # perc_mean = np.mean(volatility)

    # Plot 3: Percentage Returns (Volatility)
    A_path = to_np(states[:, :, 0, 5])
    returns = np.diff(A_path, axis=1) / (A_path[:, :-1] + 1e-9)
    volatility = returns.std(axis=1)
    # sns.kdeplot(volatility, fill=True, ax=axes[2])
    plot_normalized_kde(volatility, axes[2], '', '#1f77b4')
    axes[2].axvline(np.mean(volatility), color='red', label=f'Mean: {np.mean(volatility):.3f}')
    axes[2].set_title("Percentage Return")
    axes[2].legend()

    # Plot 4: HHI
    agent_vol = np.abs(to_np(actions[..., 0])).sum(axis=1) 
    mkt_share = agent_vol / (agent_vol.sum(axis=1, keepdims=True) + 1e-9)
    hhi = np.sum(mkt_share**2, axis=1)
    # sns.kdeplot(hhi, fill=True, ax=axes[3])
    plot_normalized_kde(hhi, axes[3], '', '#1f77b4')
    mean_hhi = np.mean(hhi)
    axes[3].axvline(mean_hhi, color='red', label=f'Mean: {mean_hhi:.4f}')
    axes[3].axvline(1/total_agents, color='black',linestyle = '--', label=f'Minimum Theoretical HHI: {(1/total_agents):.4f}')
    # axes[3].set_xlim(0, 1)
    axes[3].set_xlim(0, 2*max(hhi))
    axes[3].set_title("Herfindahl-Hirschman Index")
    axes[3].legend()

    # Plot 5: Terminal Liquidity
    last_steps = max(1, int(T * 0.1))
    total_v = np.abs(to_np(actions[..., 0])).sum(axis=(1, 2))
    late_v = np.abs(to_np(actions[:, -last_steps:, :, 0])).sum(axis=(1, 2))
    liq_share = (late_v / (total_v + 1e-9)) * 100
    # sns.kdeplot(liq_share, fill=True, ax=axes[4])
    plot_normalized_kde(liq_share, axes[4], '', '#1f77b4')
    mean_liq = np.mean(liq_share)
    axes[4].axvline(mean_liq, color='red', label=f'Mean: {mean_liq:.1f}')
    axes[4].set_title("Terminal Liquidity Share (%)")
    axes[4].legend()


    # --- 2. AGENT-SPECIFIC METRICS ---

    titles = ["Terminal Inventory of Allowances (k)", "Cumulative Production per Technology (GHW)", "Net Compliance Balance (k)", "Reward ($M)"]
    start_type_idx = 0
    for type_idx, count in enumerate(env.agent_counts_list):
        c = palette[type_idx]
        flashy_c = get_flashy_color(mcolors.to_hex(c))
        label_prefix = f"Type {type_idx+1}"
        
        # Select indices for this type across all policy blocks K
        all_type_indices = []
        for k in range(K):
            block_start = k * env.total_agents_static + start_type_idx
            block_end = block_start + int(count)
            all_type_indices.extend(list(range(block_start, block_end)))
        
        indices_jnp = jnp.array(all_type_indices)

        # Slicing: (B, T+1, N_type, 15) -> Flatten B and N_type for distributions
        type_states = states[:, :, indices_jnp, :]
        type_rewards = rewards[:, indices_jnp]
        
        # Metric 1: Terminal Inventory (q)
        q_final = to_np(type_states[:, -1, :, 1]).flatten() / 1e3
        # q_init: Take simulation 0, time 0, agent 0, feature 1
        q_init = to_np(type_states[0, 0, 0, 1]) / 1e3
        
        plot_normalized_kde(q_final, axes[5], f"{label_prefix}, Mean: {np.mean(q_final):.1f}k", c)
        axes[5].axvline(q_init, color=c, linestyle='--', alpha=0.5, label=f"{label_prefix} Initial")

        # Metric 2: Production (Fuel vs Green)
        xf = to_np(type_states[:, -1, :, 2]).flatten() / 1e3
        xg = to_np(type_states[:, -1, :, 3]).flatten() / 1e3

        # Fuel (Base Color)
        if xf.std() < 1e-4:
            plot_unit_spike(axes[6], xf.mean(), c, label=f"{label_prefix}, Fuel, Mean: {xf.mean():.1f}G", linestyle='--')
        else:
            plot_normalized_kde(xf, axes[6], f"{label_prefix}, Fuel, Mean: {xf.mean():.1f}G", c)

        # Green (Flashy Color)
        if xg.std() < 1e-4:
            plot_unit_spike(axes[6], xg.mean(), flashy_c, label=f"{label_prefix}, Green, Mean: {xg.mean():.1f}G")
        else:
            plot_normalized_kde(xg, axes[6], f"{label_prefix}, Green, Mean: {xg.mean():.1f}G", flashy_c, fill=False)

        # Metric 3: Net Compliance Balance
        ef = env.agent_params_list[start_type_idx, 1]
        eg = env.agent_params_list[start_type_idx, 2]
        balance = (to_np(type_states[:, -1, :, 1]) - (to_np(type_states[:, -1, :, 2])*ef + to_np(type_states[:, -1, :, 3])*eg)).flatten() / 1e3
        plot_normalized_kde(balance, axes[7], f"{label_prefix}, Mean: {np.mean(balance):.1f}k", c)


        # Metric 4: Rewards
        rew = to_np(type_rewards / 1e6).flatten()
        plot_normalized_kde(rew, axes[8], f"{label_prefix}, Mean: ${np.mean(rew):.2f}M", c)

        start_type_idx += int(count)
    axes[7].axvline(0, color='black', linestyle='--', label = 'x=0')
    # Global formatting
    for i in range(5, 9):
        axes[i].set_title(titles[i-5])
        # axes[i].legend(fontsize='x-small', ncol=2)
        axes[i].legend()
    
    plt.tight_layout()
    plt.show()

import matplotlib.colors as mcolors

def get_flashy_color(hex_color):
    """Returns a more saturated/vibrant version of the input color."""
    rgb = mcolors.hex2color(hex_color)
    hsv = mcolors.rgb_to_hsv(rgb)
    # Increase saturation (s) and value (v) to make it 'flashy'
    hsv[1] = min(hsv[1] * 1.5, 1.0) 
    hsv[2] = 1.0 
    return mcolors.hsv_to_rgb(hsv)

def evaluate_and_plot(env, policies, num_simulations=100, key=jax.random.PRNGKey(0)):
    """
    Runs multiple market simulations in parallel and plots the resulting densities.
    """
    print(f"Running {num_simulations} simulations for T={int(env.T)}...")
    
    # 1. Create a batch of keys for the simulations
    batch_keys = jax.random.split(key, num_simulations)
    
    # 2. Vectorize the rollout_market function across the batch of keys
    # in_axes=(0, None) means we vmap over the keys (0) but keep the policies list fixed (None)
    vmapped_rollout = jax.vmap(env.rollout_market, in_axes=(0, None))
    
    # 3. Execute bulk rollout
    # Returns shapes: 
    # states: (B, T+1, N, 15), actions: (B, T, N, 3), A_history: (B, T+1), rewards: (B, N)
    states, actions, A_history, rewards = vmapped_rollout(batch_keys, policies)
    
    # 4. Use our density plotting function
    print("Generating Density Plots...")
    plot_densities_jax(env, states, actions, rewards)
    
    return states, actions, A_history, rewards

# --- Usage Example ---
# Assuming 'ensemble' is the list of policies from Fictitious Play
# and 'env' is your initialized environment
# results = evaluate_and_plot(env, env.policies, num_simulations=128)


