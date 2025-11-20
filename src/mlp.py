# mlp.py
from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn
jax.config.update("jax_debug_nans", True)

from param_schemes import ParamScheme 


# ----------------------------
# MLP with role-aware inits
# ----------------------------
class MLP(nn.Module):
    widths: Tuple[int, ...]
    acts: Callable
    param_scheme: ParamScheme
    kernel_dims: Any
    base_kernel_dims: Any
    
    @nn.compact
    def __call__(self, x, capture_layer_acts=False):
        layer_acts = []
        # Input layer
        kinit = self.param_scheme.kernel_inits("in", self.kernel_dims, self.base_kernel_dims)
        x = nn.Dense(self.widths[0], kernel_init=kinit, use_bias=False, name="Dense_0")(x)
        if capture_layer_acts:
            layer_acts.append(jnp.abs(x).mean(axis=1))
        x = self.acts[0](x)        

        # Hidden layers
        for i, w in enumerate(self.widths[1:-1], start=1):
            kinit = self.param_scheme.kernel_inits(f"Dense_{i-1}", self.kernel_dims, self.base_kernel_dims)
            x = nn.Dense(w, kernel_init=kinit, use_bias=False, name=f"Dense_{i}")(x)
            if capture_layer_acts:
                layer_acts.append(jnp.abs(x).mean(axis=1))
            x = self.acts[i](x)
            
        # Output layer
        kinit = self.param_scheme.kernel_inits("out", self.kernel_dims, self.base_kernel_dims)
        y = nn.Dense(self.widths[-1], kernel_init=kinit, use_bias=False, name=f"Dense_{len(self.widths)-1}")(x)
        if capture_layer_acts:
            layer_acts.append(jnp.abs(y).mean(axis=1))
            stacked_act = jnp.stack(layer_acts, axis=0).T
        y = self.acts[-1](y)
        
        if capture_layer_acts:
            return y, stacked_act
        return y
