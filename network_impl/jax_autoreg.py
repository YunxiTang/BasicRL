import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import matplotlib.pyplot as plt
import random
import optax

def sequence_generation(key, num_samples: int):
    t = np.linspace(0, 12, num_samples)
    noise = np.random.normal(0.0, 0.2, (num_samples,))
    y = np.sin(t) + noise
    return y

def dataset_generation(sequence_data, tau: int):
    num_point = sequence_data.shape[0]
    feats = np.zeros((num_point-tau, tau))
    labels = np.zeros((num_point-tau, 1))
    for i in range(num_point-tau):
        feats[i,:] = sequence_data[i:i+tau]
        labels[i,:] = sequence_data[i+tau]

    return feats, labels

def data_iter(features, labels, batch_size):
    num_examples = len(labels)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = indices[i: min(i + batch_size, num_examples)]
        yield features[batch_indices,:], labels[batch_indices,:]

class auto_regressor(eqx.Module):
    layers: list

    def __init__(self, tau, key):
        keys = jax.random.split(key, 3)
        self.layers = [eqx.nn.Linear(tau, 32, key=keys[0]),
                       eqx.nn.Linear(32, 16, key=keys[1]),
                       eqx.nn.Linear(16, 1, key=keys[2]),]

    @jax.jit
    def __call__(self, x):
        x = jax.nn.relu( self.layers[0](x) )
        x = jax.nn.relu( self.layers[1](x) )
        return self.layers[2](x)

    @jax.jit
    def get_loss(self, x, y):
        pred_y = jax.vmap(self)(x)
        return jax.numpy.mean((y - pred_y) ** 2 / 2.)
    @jax.jit
    @jax.value_and_grad
    def loss_and_grad(self, x, y):
        pred_y = jax.vmap(self)(x)
        return jax.numpy.mean((y - pred_y) ** 2 / 2.)

@jax.value_and_grad
def loss_and_grad(model, feats, labels):
    predicts = jax.vmap(model)(feats)
    return jnp.sum( (predicts - labels) ** 2 )

if __name__ == '__main__':
    key = jax.random.PRNGKey(0)
    x = sequence_generation(key, 1000)
    tau = 4
    xs, ys = dataset_generation(x, tau)

    model = auto_regressor(tau, key)
    batch_size = 200
    lr = 0.1
    optimizer = optax.sgd(lr)
    opt_state = optimizer.init(model)
    
    for epoch in range(100):
        for feat, label in data_iter(xs, ys, batch_size):
            _, grads = model.loss_and_grad(feat, label)
            updates, opt_state = optimizer.update(grads, opt_state)
            model = optax.apply_updates(model, updates)
        l = model.get_loss(xs, ys)
        print('epoch {} || loss {}'.format(epoch+1, l))

    # visualization
    valid_onestep = np.zeros((1000-tau,))
    for i in range(1000-tau):
        valid_onestep[i] = model(x[i:i+tau])

    plt.figure(1)
    plt.plot(x)
    plt.plot( np.pad(valid_onestep, (tau,0)))
    plt.grid()
    plt.show()
