"""jax-based simple classification"""
import jax
import equinox as eqx
import jax.numpy as jnp
from jax import vmap, grad, jit
import matplotlib.pyplot as plt
import optax
import random

from torch_softmax import data_loader_fashion_mnist
