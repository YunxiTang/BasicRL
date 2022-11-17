import jax
from jax import numpy as jnp

class LinearRegressor():
    def __init__(self, w, b) -> None:
        self.weight = w
        self.bias = b

    def predict(self, x):
        return self.weight * x + self.bias

    def rms(self, xs: jnp.ndarray, ys: jnp.ndarray):
        return jnp.sqrt(jnp.sum(jnp.square(self.predict(xs) - ys)))

def flatten_linear_regressor(regressor: LinearRegressor):
    leaves = (regressor.weight, regressor.bias)
    aux = None
    return (leaves, aux)

def unflatten_linear_regressor(_aux, leaves):
    w, b = leaves
    return LinearRegressor(w, b)

jax.tree_util.register_pytree_node(
    LinearRegressor,
    flatten_linear_regressor,
    unflatten_linear_regressor,
)

def loss_fn(regressor: LinearRegressor, xs, ys):
    return regressor.rms(xs=xs, ys=ys)

if __name__ == '__main__':
    regressor = LinearRegressor(13., 0.)
    xs = jnp.array([42.0])
    ys = jnp.array([500.0])
    
    grad_fn = jax.grad(loss_fn)
    for i in range(1001):
        grads = grad_fn(regressor, xs, ys)
        regressor = jax.tree_util.tree_map(lambda x, y: x - 1e-4 * y,
                                            regressor, grads)
        print(loss_fn(regressor, xs, ys))
    print(regressor.weight, regressor.bias)