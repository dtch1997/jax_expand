""" Expanding an MLP's hidden layers. """

import jax.numpy as jnp

from jax import random
from jax_expand import mlp, mlp_expand

if __name__ == "__main__":
    # Test our implementation
    params = mlp.init_mlp_params([8, 16, 2])
    print("Original MLP:")
    mlp.pretty_print_params(params)
    print()

    # Expand the first layer
    params_new = mlp_expand.expand_mlp_layer(params, 0)
    print("Expanded MLP:")
    mlp.pretty_print_params(params_new)
    print()

    # Make predictions with the original and expanded MLPs
    inputs = random.normal(random.PRNGKey(0), (10, 8))
    outputs = mlp.mlp_predict(params, inputs)
    outputs_new = mlp.mlp_predict(params_new, inputs)
    print("Difference in predictions:")
    print(jnp.mean(jnp.abs(outputs - outputs_new)))
    print()
