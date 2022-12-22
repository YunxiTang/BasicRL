import equinox as eqx
from jax import random
import jax.numpy as jnp
import jax
import numpy as np

m_true = 1.0
l_true = 0.5
c_model = 0.003

m_model = 0.7
l_model = 0.4
c_model = 0.002

g = 9.81

# Inverted pendulum
class InvertedPendulum(eqx.Module):
    residual_model: eqx.Module
    input_dim: int = eqx.static_field()
    output_dim: int = eqx.static_field()

    def __init__(self, input_dim, output_dim, key):
        keys = random.split(key, 3)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.residual_model = [eqx.nn.Linear(3, 2, key=keys[0]),
                               eqx.nn.Linear(2, 2, key=keys[1]),
                               eqx.nn.Linear(2, 2, key=keys[2])]

    @jax.jit
    def __call__(self, input):
        q = input[:, 0]
        qdot = input[:, 1]
        u = input[:, 3]
        I = 1 / 3 * m_model * l_model ** 2
        xdot_nom = jnp.concatenate([qdot, (u - m_model * g * l_model * jnp.sin(q) - c_model * qdot) / I])
        xdot_res = self.residual_model(input)
        return xdot_nom + xdot_res

if __name__ == '__main__':
    key = random.PRNGKey(seed=0)
    pend = InvertedPendulum(3, 2, key=key)
    
    print(jax.tree_util.tree_structure(pend))

    a = np.array([[1,2,3], 
                  [1,2,3]])
    print( np.pad( a, pad_width=((0, 2), (0, 3)) ) )