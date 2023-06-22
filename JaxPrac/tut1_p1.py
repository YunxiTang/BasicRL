## Standard libraries
import numpy as np
import time

## Imports for plotting
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import seaborn as sns
sns.set()

## Progress bar
from tqdm.auto import tqdm

#=========== jax basic info
import jax
import jax.numpy as jnp
print("Using jax with version of ", jax.__version__)
print("available devices: ", jax.devices())

a = jnp.array([1., 2., 3.], dtype=jnp.float32)
b = jnp.arange(0, 7)
b_cpu = jax.device_get(b)
print(b_cpu.__class__, type(b_cpu))
b_gpu = jax.device_put(b_cpu)
print(b_gpu.__class__, type(b_gpu))
print(f'Device put: {b_gpu.__class__} on {b_gpu.device()}')

#========== Immutable tensor
# A DeviceArray object is immutable
x = jnp.array([0., 0.])
print(id(x), x)
x = x.at[0].set(12.)
print(id(x), x) # id is changed

# =========== Pseudo Random Numbers ==========
# JAX takes a different approach by explicitly passing and iterating the PRNG state
# create a PRNG state for the seed 42
rng = jax.random.PRNGKey(21)
print(type(rng), rng)
jax_random_number_1 = jax.random.normal(rng, (1, 2))
jax_random_number_2 = jax.random.normal(rng, (1, 2))
print("jax_random_number_1: ", jax_random_number_1)
print('jax_random_number_2: ', jax_random_number_2) # the same

# Typical random numbers in NumPy
np.random.seed(42)
np_random_number_1 = np.random.normal(0, 1., (1, 2))
np_random_number_2 = np.random.normal(0, 1., (1, 2))
print('NumPy - Random number 1:', np_random_number_1)
print('NumPy - Random number 2:', np_random_number_2) # not the same

# the right way
rng, key1, key2 = jax.random.split(rng, 3)
jax_random_number_1 = jax.random.normal(key1, (1, 2))
jax_random_number_2 = jax.random.normal(key2, (1, 2))
print("jax_random_number_1: ", jax_random_number_1)
print('jax_random_number_2: ', jax_random_number_2) # not the same

# ============= Function transformations with Jaxpr 
# The intermediate jaxpr representation defines a computation graph,
def simple_graph(x):
    x = x + 2
    x = x ** 2
    x = x + 3
    y = x.mean()
    return y

inp = jnp.array([1.,2.])
print(simple_graph(inp))
# use make_jaxpr to view jaxpr representation of this function
#  jaxpr only understand side-effect-free code
print(jax.make_jaxpr(simple_graph)(inp))
# { lambda ; a:f32[2]. let
#     b:f32[2] = add a 2.0
#     c:f32[2] = integer_pow[y=2] b
#     d:f32[2] = add c 3.0
#     e:f32[] = reduce_sum[axes=(0,)] d
#     f:f32[] = div e 2.0
#   in (f,) }

# ================ Automatic differentiation
grad_function_of_simple_graph = jax.grad(simple_graph)
gradients = grad_function_of_simple_graph(inp)
print('Gradient', gradients)
# jaxpr of the grad_function
jax.make_jaxpr(grad_function_of_simple_graph)(inp)

# both value and grad
val_grad_function = jax.value_and_grad(simple_graph)
print( jax.make_jaxpr(val_grad_function)(inp) )
print( val_grad_function(inp) )

# ======== jit
jitted_function = jax.jit(simple_graph)

xo = jax.random.normal(key1, (50000,))
ts = time.time()
simple_graph(xo)
print(time.time() - ts)

jitted_function(xo) # jit firstly
ts = time.time()
jitted_function(xo)
print(time.time() - ts)