import flax
from flax import linen as nn
import jax.numpy as jnp

# =========== activation function ============
class Sigmoid(nn.Module):
    def __call__(self, x):
        return 1. / ( 1. + jnp.exp(-x) )
    
class Tanh(nn.Module):
    def __call__(self, x):
        x_exp, neg_x_exp = jnp.exp(x), jnp.exp(-x)
        return (x_exp - neg_x_exp) / (x_exp + neg_x_exp)
    
class ReLU(nn.Module):
    def __call__(self, x):
        return jnp.maximum(x, 0)
    
class LeakyReLU(nn.Module):
    alpha: float = 0.1
    
    def __call__(self, x):
        return jnp.where(x > 0, x, self.alpha * x)
    
class ELU(nn.Module):

    def __call__(self, x):
        return jnp.where(x > 0, x, jnp.exp(x)-1)

class Swish(nn.Module):

    def __call__(self, x):
        return x * nn.sigmoid(x)