import equinox as eqx
import optax
import jax

from typing import List


class Linear(eqx.Module):
    weight: jax.Array
    bias: jax.Array

    def __init__(self, in_size, out_size, key):
        wkey, bkey = jax.random.split(key)
        self.weight = jax.random.normal(wkey, (out_size, in_size))
        self.bias = jax.random.normal(bkey, (out_size,))

    def __call__(self, x):
        return self.weight @ x + self.bias


class MLP(eqx.Module):

    layers: List[Linear]

    def __init__(self, in_size, out_size, hidden_size, depth, key):

        layers = []
        layer_sizes = [in_size] + [hidden_size] * (depth - 1) + [out_size]
        in_sizes = layer_sizes[:-1]
        out_sizes = layer_sizes[1:]
        for i, (in_size, out_size) in enumerate(zip(in_sizes, out_sizes)):
            layers.append(Linear(in_size, out_size, key))

        self.layers = layers

    def __call__(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = jax.nn.relu(x)
        return x


@jax.jit
def loss_fn(model, x, y):
    pred_y = jax.vmap(model)(x)
    return jax.numpy.mean((y - pred_y) ** 2)


if __name__ == "__main__":
    in_size, hidden_size, out_size = 8, 16, 2
    model = MLP(in_size, out_size, hidden_size, 2, jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (100, in_size))
    print(jax.vmap(model)(x).shape)
