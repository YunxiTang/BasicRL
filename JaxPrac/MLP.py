import jax
from flax import linen as nn
from jax import numpy as jnp
from jax import jit, vmap, grad
import random
import numpy as np
from typing import Sequence

# data generator
def data_synthetic():
    return 0


# define a data iterator
def data_iter(batch_size, features, labels):
    num_examples = len(labels)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = indices[i: min(i + batch_size, num_examples)]
        yield features[batch_indices], labels[batch_indices]

# define a model
class simpleMLP(nn.Module):
    
    def setup(self):
        self.layers = [nn.Dense(5), nn.Dense(10), nn.Dense(3)]

    def __call__(self, inputs):
        x = inputs
        for i, lyr in enumerate(self.layers):
            x = lyr(x)
            if i != len(self.layers)-1:
                x = nn.relu(x)
        return x

# define loss
def loss(y_hat, y):
    l = jnp.mean((y_hat-y)**2)
    return l

def sgd(param, lr):
    return 0

def train(init_param, net, train_iter, loss, updater):
    return 0

if __name__ == '__main__':
    model = simpleMLP()
    key1, key2 = jax.random.split(jax.random.PRNGKey(0), 2)
    dummy_x = jax.random.normal(key1, (4, 4))
    param = model.init(key2, dummy_x)
    print(param)
    y = model.apply(param, dummy_x)
    print(y)
   
