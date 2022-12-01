import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu

class myrelu(eqx.Module):
    def __init__(self):
        pass
    def __call__(self, x):
        return jax.numpy.maximum(x, 0)

class normal_dist(eqx.Module):
    mean: jnp.array
    std: jnp.array

    def __init__(self, shape, key):
        key1, key2 = jax.random.split(key, 2)
        self.mean = jax.random.normal(key1, shape)
        self.std = jax.random.normal(key2, shape)

    def __call__(self, x):
        return jax.scipy.stats.multivariate_normal.logpdf(x, mean=self.mean, cov=self.std)


class MyModule(eqx.Module):
    layers: list

    def __init__(self, key):
        key1, key2, key3 = jax.random.split(key, 3)
        self.layers = [eqx.nn.Linear(1, 8, key=key1),
                       myrelu(),
                       eqx.nn.Linear(8, 8, key=key2),
                       myrelu(),
                       eqx.nn.Linear(8, 8, key=key2),
                       myrelu(),
                       eqx.nn.Linear(8, 1, key=key3)]

    @jax.jit
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    @jax.value_and_grad
    def get_value_and_grad(self, x, y):
        pred_y = jax.vmap(self)(x)
        return jax.numpy.mean((y - pred_y) ** 2)

    def update_with_loss(self, x, y, lr):
        loss, grads = ( self.get_value_and_grad(x, y) )
        learning_rate = lr
        res = jtu.tree_map(lambda m, g: m - learning_rate * g, self, grads)
        return res, loss

x_key, y_key, model_key = jax.random.split(jax.random.PRNGKey(0), 3)
x = 2. * jax.random.normal(x_key, (100, 1))
y = 2 * x ** 2 + 1.5 + 0.1 * jax.random.normal(y_key, (100, 1))
model = MyModule(model_key)
for i in range(500):
    model, l = model.update_with_loss(x, y, 0.001)
    print('epoch {} || loss {}'.format(i, l))


# x0 = jax.random.normal(x_key, (1,))
# fx = jax.jacfwd(model, argnums=(0,))(x0)
# print(fx)
# print(model)
for i in range(5):
    print(jtu.tree_leaves(model.layers[i]))
    print("=======================")

import matplotlib.pyplot as plt
y_hat = jax.vmap(model)(x)
plt.figure(1)
plt.scatter(x[:,0], y[:,0])
plt.scatter(x[:,0], y_hat[:,0])
plt.show()

# eqx.tree_serialise_leaves("model_mlp.eqx", model)

# model_new = MyModule(model_key, 15.)
# model_loaded = eqx.tree_deserialise_leaves("model_mlp.eqx", model_new)
# print( model_loaded == model )