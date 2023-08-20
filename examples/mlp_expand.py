""" Expanding an MLP's hidden layers. """

import jax.numpy as jnp

from jax import random
from jax_expand import mlp


def get_n_layers(params: mlp.Params):
    """Get the number of layers in a multi-layer perceptron"""
    return len(params)


def expand_mlp_layer(params: mlp.Params, layer_idx: int):
    n_layers = get_n_layers(params)
    if layer_idx < 0 or layer_idx >= n_layers:
        raise ValueError(f"Invalid layer index: {layer_idx}")
    if layer_idx == n_layers - 1:
        raise ValueError("Cannot expand the final layer")

    # Get the weights and biases for the layer we want to expand
    w_prev, b_prev = params[layer_idx]
    w_next, b_next = params[layer_idx + 1]

    # Expand w_prev, b_prev by doubling the number of output dimensions
    # The new values are initialized to the old values plus some noise
    w_key = random.split(random.PRNGKey(0))[0]
    b_key = random.split(random.PRNGKey(0))[1]
    w_prev_new = jnp.concatenate(
        [w_prev, w_prev + random.normal(w_key, w_prev.shape)], axis=1
    )
    b_prev_new = jnp.concatenate(
        [b_prev, b_prev + random.normal(b_key, b_prev.shape)], axis=0
    )

    # Expand w_next, b_next by doubling the number of input dimensions
    # The new values are initialized to zero
    w_next_new = jnp.concatenate(
        [w_next, jnp.zeros((w_next.shape[0], w_next.shape[1]))], axis=0
    )
    b_next_new = b_next

    # Return the new parameters
    params_new = params.copy()
    params_new[layer_idx] = (w_prev_new, b_prev_new)
    params_new[layer_idx + 1] = (w_next_new, b_next_new)
    return params_new


if __name__ == "__main__":
    # Test our implementation
    params = mlp.init_mlp_params([8, 16, 2])
    print("Original MLP:")
    mlp.pretty_print_params(params)
    print()

    # Expand the first layer
    params_new = expand_mlp_layer(params, 0)
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
