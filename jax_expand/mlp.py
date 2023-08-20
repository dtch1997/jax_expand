import jax.numpy as jnp
from jax import random
from typing import List


def init_mlp_params(layer_widths: List[int], keys=None):
    """Initialize the parameters of a multi-layer perceptron"""
    if keys is None:
        keys = random.split(random.PRNGKey(0), len(layer_widths))
    weights = [
        random.normal(keys[i], (layer_widths[i], layer_widths[i + 1]))
        for i in range(len(layer_widths) - 1)
    ]
    biases = [
        random.normal(keys[i], (layer_widths[i + 1],))
        for i in range(len(layer_widths) - 1)
    ]

    # Reorganize into a list of (weight, bias) pairs
    params = list(zip(weights, biases))
    return params


def pretty_print_params(params):
    """Print the parameters of a multi-layer perceptron"""
    for i, (w, b) in enumerate(params):
        print(f"Layer {i}:")
        print(f"  Weights: {w.shape}")
        print(f"  Biases: {b.shape}")


def mlp_predict(params, inputs, activation=jnp.tanh):
    """Make predictions from a multi-layer perceptron"""
    activations = inputs
    for w, b in params[:-1]:
        outputs = jnp.dot(activations, w) + b
        activations = activation(outputs)
    final_w, final_b = params[-1]
    outputs = jnp.dot(activations, final_w) + final_b
    return outputs
