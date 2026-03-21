import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np

class ActionNN(eqx.Module):
    layers: list
    state_dim: int = eqx.static_field()
    action_dim: int = eqx.static_field()

    def __init__(self, state_dim=12, action_dim=3, architecture=[128, 128, 128, 128, 128], key=None):
        if key is None:
            # Using a random seed if no key provided
            key = jax.random.PRNGKey(np.random.randint(0, 1000000))
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # The NN processes everything except the tech_control flag
        # State: [t, theta, xi_f, xi_g, eps, A, P, e1, e2, c1, c2, tech_control]
        nn_input_dim = state_dim - 1 
        
        dims = [nn_input_dim] + architecture + [action_dim]
        keys = jax.random.split(key, len(dims) - 1)
        
        self.layers = [
            eqx.nn.Linear(dims[i], dims[i+1], key=keys[i])
            for i in range(len(dims) - 1)
        ]

    def __call__(self, x):
        """
        x: (12,) array 
        Logic: Use x[0:11] for the NN, use x[11] as the tech_control override for eta.
        """
        # 1. Identify the tech_control index (the last dimension)
        tech_idx = self.state_dim - 1
        
        # 2. Prepare Features (t and all state vars up to but excluding tech_control)
        h = x[:tech_idx]
            
        # 3. Forward Pass
        for layer in self.layers[:-1]:
            h = jax.nn.silu(layer(h))
        
        raw_out = self.layers[-1](h)
        
        # 4. Activation Mapping
        # index 0: f (trade amount)
        # index 1: xi (total production quantity)
        # index 2: eta_pred (production mix prediction)
        f = jnp.tanh(raw_out[0])
        xi = jax.nn.sigmoid(raw_out[1])
        eta_pred = jax.nn.sigmoid(raw_out[2])
        
        # 5. Tech Control Override Logic
        # If tech_control >= 0, it forces the value of eta. 
        # If tech_control < 0 (usually -1.0), we use the NN prediction.
        eta = jnp.where(x[tech_idx] >= 0, x[tech_idx], eta_pred)
            
        return jnp.array([f, xi, eta])