"""hands on softmax regression"""
import matplotlib.pyplot as plt
from jax import jit, grad, vmap, random
from jax import numpy as jnp
import numpy as np
from torch.utils import data
from torchvision.datasets import MNIST
from jax.scipy.special import logsumexp

# hyperparameters
def random_layer_params(m, n, key, scale=1e-2):
    # A helper function to randomly initialize weights and biases
    # for a dense neural network layer
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (n, m), dtype=jnp.float32), scale * random.normal(b_key, (n, ), dtype=jnp.float32)

# Initialize all layers for a fully-connected neural network
def init_network(sizes, key):
    keys = random.split(key, len(sizes))
    return [random_layer_params(m, n, key) for m, n, key in zip(sizes[:-1], sizes[1:], keys)]

def relu(x):
    return jnp.maximum(0, x)

def net(params, image):
    activations = image
    for w, b in params[:-1]:
        output = jnp.dot(w, activations) + b
        activations = relu(output)
    final_w, final_b = params[-1]
    logits = jnp.dot(final_w, activations) + final_b
    return logits - logsumexp(logits)

layer_sizes = [784, 512, 512, 10]
lr = 0.01
num_epochs = 3
batch_size = 128
n_targets = 10

params = init_network(layer_sizes, random.PRNGKey(0))
random_flattened_image = random.normal(random.PRNGKey(1), (28 * 28,))
preds = net(params, random_flattened_image)
print(preds.shape)

# batched version
batched_predict = vmap(net, in_axes=(None, 0))
random_flattened_images = random.normal(random.PRNGKey(1), (10, 28 * 28))

batched_preds = batched_predict(params, random_flattened_images)
print(batched_preds.shape)

def one_hot(x, k, dtype=jnp.float32):
  """Create a one-hot encoding of x of size k."""
  return jnp.array(x[:, None] == jnp.arange(k), dtype)

def accuracy(params, images, targets):
  target_class = jnp.argmax(targets, axis=1)
  predicted_class = jnp.argmax(batched_predict(params, images), axis=1)
  return jnp.mean(predicted_class == target_class)

def loss(params, images, targets):
  preds = batched_predict(params, images)
  return -jnp.mean(preds * targets)

@jit
def update(params, x, y):
  grads = grad(loss)(params, x, y)
  return [(w - lr * dw, b - lr * db) for (w, b), (dw, db) in zip(params, grads)]



def numpy_collate(batch):
  if isinstance(batch[0], np.ndarray):
    return np.stack(batch)
  elif isinstance(batch[0], (tuple,list)):
    transposed = zip(*batch)
    return [numpy_collate(samples) for samples in transposed]
  else:
    return np.array(batch)

class NumpyLoader(data.DataLoader):
  def __init__(self, dataset, batch_size=1,
                shuffle=False, sampler=None,
                batch_sampler=None, num_workers=0,
                pin_memory=False, drop_last=False,
                timeout=0, worker_init_fn=None):
    super(self.__class__, self).__init__(dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=numpy_collate,
        pin_memory=pin_memory,
        drop_last=drop_last,
        timeout=timeout,
        worker_init_fn=worker_init_fn)

class FlattenAndCast(object):
  def __call__(self, pic):
    return np.ravel(np.array(pic, dtype=jnp.float32))

mnist_dataset = MNIST('/tmp/mnist/', download=True, transform=FlattenAndCast())
training_generator = NumpyLoader(mnist_dataset, batch_size=batch_size, num_workers=0)

train_images = np.array(mnist_dataset.train_data).reshape(len(mnist_dataset.train_data), -1)
train_labels = one_hot(np.array(mnist_dataset.train_labels), n_targets)

# Get full test dataset
mnist_dataset_test = MNIST('/tmp/mnist/', download=True, train=False)
test_images = jnp.array(mnist_dataset_test.test_data.numpy().reshape(len(mnist_dataset_test.test_data), -1), dtype=jnp.float32)
test_labels = one_hot(np.array(mnist_dataset_test.test_labels), n_targets)

print(test_images[0])
print(test_labels[0])

import time

for epoch in range(num_epochs):
  start_time = time.time()
  for x, y in training_generator:
    y = one_hot(y, n_targets)
    params = update(params, x, y)
  epoch_time = time.time() - start_time

  train_acc = accuracy(params, train_images, train_labels)
  test_acc = accuracy(params, test_images, test_labels)
  print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
  print("Training set accuracy {}".format(train_acc))
  print("Test set accuracy {}".format(test_acc))
