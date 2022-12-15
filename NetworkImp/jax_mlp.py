"""jax-based multi-layer perceptron"""
import jax
import equinox as eqx
import jax.numpy as jnp
from jax import vmap, grad, jit
import matplotlib.pyplot as plt
import optax
from typing import Callable
from jax_utils import FlattenAndCast, NumpyLoader
import numpy as np
import torchvision

def data_loader_fashion_mnist(batch_size, resize=None):
    mnist_train = torchvision.datasets.FashionMNIST(root='..\data', train=True, 
                                                    transform=FlattenAndCast(), download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root='..\data', train=False, 
                                                   transform=FlattenAndCast(), download=True)
    train_iter = NumpyLoader(mnist_train, batch_size, shuffle=True, num_workers=4)
    test_iter = NumpyLoader(mnist_test, batch_size, shuffle=False, num_workers=4)
    return train_iter, test_iter

def one_hot(x, k, dtype=jnp.float32):
    """Create a one-hot encoding of x of size k."""
    return jnp.array(x[:, None] == jnp.arange(k), dtype)

class SoftMax(eqx.Module):
    def __call__(self, x):
        """jax-based softmax"""
        tmp = jnp.sum( jnp.exp(x - jnp.max(x)) )
        return jnp.exp(x - jnp.max(x)) / tmp 

class CrossEntropyLoss(eqx.Module):
    def __call__(self, y_hat, y):
        """jax-based cross entropy loss"""
        res = jnp.mean( -y_hat * y )
        return res 

class classifier(eqx.Module):
    layers: list
    loss_fun: Callable
    
    def __init__(self, key):
        key1, key2 = jax.random.split(key, 2)
        self.layers = [eqx.nn.Linear(28*28, 28*28, key=key1),
                       eqx.nn.Linear(28*28, 10, key=key2)]
        self.loss_fun = CrossEntropyLoss()

    @jax.jit
    def __call__(self, x):
        x = self.layers[0](x)
        x = jnp.tanh(x)
        x = self.layers[1](x)
        res = x -  jax.scipy.special.logsumexp(x)
        return res
         
    @jit
    def get_loss(self, x, y):
        pred_y = jax.vmap(self)(x)
        return self.loss_fun(pred_y, y)

    @jax.value_and_grad
    def loss_and_grad(self, x, y):
        pred_y = jax.vmap(self)(x)
        return self.loss_fun(pred_y, y)

def accuracy(logits, labels):
    target_class = jnp.argmax(labels, axis=1)
    predicted_class = jnp.argmax(logits, axis=1)
    return jnp.mean(predicted_class == target_class)

def high_api_run(train_iter, test_iter, key, lr=0.01, scale=0.1):

    batch_size = 100
    lr = 0.01

    train_iter, test_iter = data_loader_fashion_mnist(batch_size)

    net = classifier(key)

    optimizer = optax.sgd(lr)
    opt_state = optimizer.init(net)

    for epoch in range(50):
        l_sum = 0.0
        acc_sum = 0.0
        i = 0
        for feats, labels in train_iter:
            train_labels = one_hot(np.array(labels), 10)
            loss, grads = net.loss_and_grad(feats, train_labels)
            updates, opt_state = optimizer.update(grads, opt_state)
            net = optax.apply_updates(net, updates)
            l_sum += float(loss)

            acc = accuracy(jax.vmap(net)(feats), train_labels)
            acc_sum += float(acc)
            i += 1

        print('epoch {} || loss {} || training accuracy {}'.format(epoch+1, l_sum, acc_sum/i))

if __name__ == '__main__':
    print('==========')
    batch_size = 50
    train_iter, test_iter = data_loader_fashion_mnist(batch_size)
    key = jax.random.PRNGKey(0)
    high_api_run(train_iter, test_iter, key)
