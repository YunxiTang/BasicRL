import jax
import equinox as eqx
import jax.numpy as jnp
from jax import vmap, grad, jit
import matplotlib.pyplot as plt
import optax
import random

def data_generation(w: jnp.array, b: jnp.array, num_samples: int):
    x = jax.random.normal(jax.random.PRNGKey(3), (num_samples, 1))
    y = w * x + b + jax.random.normal(jax.random.PRNGKey(1), (num_samples, 1))
    return x, y

def data_iter(features, labels, batch_size):
    num_examples = len(labels)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = indices[i: min(i + batch_size, num_examples)]
        yield features[batch_indices,:], labels[batch_indices,:]

def low_api_run():

    xs, ys = data_generation(1.5, 1.2, 1000)

    key = jax.random.PRNGKey(0)
    key_w, key_b = jax.random.split(key, 2)
    w = jax.random.normal(key_w, (1,))
    b = jax.random.normal(key_b, (1,))

    params = {'w': w, 'b': b}

    @jit
    def prediction(param, x):
        return param['w'] * x + param['b']
    @jit
    def loss(param, x, y):
        batched_pre = vmap(prediction, in_axes=(None, 0))
        return jnp.mean((y - batched_pre(param, x)) ** 2 / 2)

    batch_size = 100
    lr = 0.03

    plt.figure(1)
    plt.scatter(xs, ys)
    plt.scatter(xs, vmap(prediction, in_axes=(None, 0))(params, xs))

    for epoch in range(50):
        for feat, label in data_iter(xs, ys, batch_size):
            grads = grad(loss)(params, feat, label)
            params = jax.tree_util.tree_map(lambda x, g: x - lr * g, params, grads)

        print('epoch {} || loss {}'.format(epoch+1, loss(params, xs, ys)))

    plt.scatter(xs, vmap(prediction, in_axes=(None, 0))(params, xs))
    plt.show()
    return loss(params, xs, ys)

def high_api_run():
    xs, ys = data_generation(1.5, 1.2, 1000)

    class net(eqx.Module):
        layers: list

        def __init__(self, key):
            self.layers = [eqx.nn.Linear(1, 1, key=key),]

        @jax.jit
        def __call__(self, x):
            for i, lyr in enumerate(self.layers):
                x = lyr(x)
            return x

        @jit
        def get_loss(self, x, y):
            pred_y = jax.vmap(self)(x)
            return jax.numpy.mean((y - pred_y) ** 2 / 2.)
        
        @jax.value_and_grad
        def loss_and_grad(self, x, y):
            pred_y = jax.vmap(self)(x)
            return jax.numpy.mean((y - pred_y) ** 2 / 2.)

        def update_with_loss(self, x, y, lr):
            loss, grads = ( self.loss_and_grad(x, y) )
            learning_rate = lr
            res = jax.tree_util.tree_map(lambda m, g: m - learning_rate * g, self, grads)
            return res, loss

    key = jax.random.PRNGKey(0)
    model = net(key)
    batch_size = 100
    lr = 0.03
    optimizer = optax.sgd(lr)

    opt_state = optimizer.init(model)

    plt.figure(1)
    plt.scatter(xs, ys)
    plt.scatter(xs, vmap(model)(xs))
    for epoch in range(50):
        for feat, label in data_iter(xs, ys, batch_size):
            _, grads = model.loss_and_grad(feat, label)
            updates, opt_state = optimizer.update(grads, opt_state)
            model = optax.apply_updates(model, updates)
        l = model.get_loss(xs, ys)
        print('epoch {} || loss {}'.format(epoch+1, l))
    plt.scatter(xs, vmap(model)(xs))
    plt.show()
    return l
            

if __name__ == '__main__':
    final_loss1 = low_api_run()
    final_loss2 = high_api_run()
    print(final_loss1, final_loss2)
