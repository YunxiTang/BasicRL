### test iLQR wirtten in JAX
import jax
import jax.numpy as jnp
from jax import jit, vmap, grad, jacfwd
import numpy as np
import matplotlib.pyplot as plt
import time

############### time-invariant params ##############
dt = 0.01
# time grids
N = 10           
max_iter = 10
Reg = 1e-6
x_dim = 4
u_dim = 2


Q = jnp.diag(jnp.array([0.1, 0.1, 0.1, 0.1]))
Qf = jnp.diag(jnp.array([100, 100, 100, 100])) * 0.1
R = jnp.diag(jnp.array([0.1, 0.1]))

x0 = jnp.array([0.0, 0.0, 0.0, 0.0])
xf = jnp.array([2.0, 3.0, 0.*3.14/2., 0.0])

############### user-defined functions ##############
@jax.jit
def path_cost(x, u):
    # path cost
    cost = 0.5 * (x - xf) @ Q @ jnp.transpose( x - xf) + 0.5 * u @ R @ jnp.transpose( u )
    return cost * dt

@jax.jit
def final_cost(x):
    # final cost
    cost = 0.5 * (x - xf) @ Qf @ jnp.transpose( x - xf )
    return cost

@jax.jit
def discrete_dynamics(q, u):
    x = q[0]
    y = q[1]
    theta = q[2]
    v = q[3]
    u_the = u[0]
    u_v = u[1]
    x_next = x + dt * v * jnp.sin(theta)
    y_next = y + dt * v * jnp.cos(theta)
    theta_next = theta + dt * u_the * v
    v_next = v + dt * u_v
    q_next = jnp.array([x_next, y_next, theta_next, v_next])
    return q_next

############### gradient/hessian functions ##############
f_jac_func = jax.jit( jax.jacfwd(discrete_dynamics, argnums=[0, 1]) )  # fx, fu
path_cost_jac = jax.jit( jax.jacfwd(path_cost, argnums=[0, 1]))        # lx, lu
path_cost_hessian = jax.jit( jax.hessian(path_cost, argnums=[0, 1]) )  # (l_xx, l_xu), (l_ux, l_uu)
final_cost_jac = jax.jit( jax.jacfwd(final_cost))                      # lx
final_cost_hessian = jax.jit( jax.hessian(final_cost) )                # lxx

############### iLQR-related functions (in general) ##############
@jax.jit
def rollout_body(x_i, u_i):
    # function body in initial rollout
    x_next = discrete_dynamics(x_i, u_i)
    return x_next, x_next

def init_forward(x_init, u):
    # initial forward pass
    _, x_trj = jax.lax.scan(rollout_body, x_init, u)
    return jnp.vstack((x_init, x_trj))

@jax.jit
def forward_pass_body(i, aux_state):
    x_trj_ref, u_trj_ref, k_trj_ref, K_trj_ref, x_trj, u_trj = aux_state
    x = x_trj[i]
    u = u_trj_ref[i] + 0.05 * k_trj_ref[i] + K_trj_ref[i] @ (x - x_trj_ref[i])
    u_trj = u_trj.at[i].set(u)
    x_next = discrete_dynamics(x, u)
    x_trj = x_trj.at[i+1].set(x_next)
    return [x_trj_ref, u_trj_ref, k_trj_ref, K_trj_ref, x_trj, u_trj]

@jax.jit
def forward_pass(x_trj_ref, u_trj_ref, k_trj_ref, K_trj_ref):
    x_trj = jnp.empty_like(x_trj_ref)
    u_trj = jnp.empty_like(u_trj_ref)
    x_trj = x_trj.at[0].set(x_trj_ref[0])
    x_trj_ref, u_trj_ref, k_trj_ref, K_trj_ref, x_trj, u_trj = jax.lax.fori_loop(0, 
                                                                                N-1, 
                                                                                forward_pass_body, 
                                                                                [x_trj_ref, u_trj_ref, k_trj_ref, K_trj_ref, x_trj, u_trj])
    return x_trj, u_trj

@jax.jit
def compute_single_step_cost(i, aux_state):
    x_trj, u_trj, cost = aux_state
    x = x_trj[i]
    u = u_trj[i]
    cost_s = path_cost(x, u)
    cost = cost + cost_s
    return [x_trj, u_trj, cost]

@jax.jit
def traj_cost(x_trj, u_trj):
    cost = 0.
    x_trj, u_trj, cost = jax.lax.fori_loop(0, N-1, compute_single_step_cost, [x_trj, u_trj, cost])
    cost_f = final_cost(x_trj[-1])
    cost = cost + cost_f
    return cost

@jax.jit
def backward_pass_body(i, aux_state):
    V_x, V_xx, k_trj, K_trj, x_trj, u_trj, expected_cost_reduc = aux_state
    # convert the time axis
    k = N - 2 - i
    x = x_trj[k]
    u = u_trj[k]
    f_x, f_u = f_jac_func(x, u)
    l_x, l_u = path_cost_jac(x, u)
    (l_xx, l_xu), (l_ux, l_uu) = path_cost_hessian(x, u)

    Q_x = l_x + f_x.T @ V_x
    Q_u = l_u + f_u.T @ V_x

    Q_xx = l_xx + f_x.T @ V_xx @ f_x
    Q_ux = l_ux + f_u.T @ V_xx @ f_x
    Q_xu = l_xu + f_x.T @ V_xx @ f_u
    Q_uu = l_uu + f_u.T @ V_xx @ f_u

    # add regularization
    Q_uu_reg = Q_uu + jnp.eye(u_dim) * Reg

    k_ff = - jnp.linalg.inv(Q_uu_reg) @ Q_u
    K_fb = - jnp.linalg.inv(Q_uu_reg) @ Q_ux

    k_trj = k_trj.at[k].set(k_ff)
    K_trj = K_trj.at[k].set(K_fb)

    V_x = Q_x + K_fb.T @ Q_uu @ k_ff + K_fb.T @ Q_u + Q_xu @ k_ff
    V_xx = Q_xx + Q_xu @ K_fb + K_fb.T @ Q_ux + K_fb.T @ Q_uu @ K_fb
    expected_cost_reduc = expected_cost_reduc + 1/2 * k_ff.T @ Q_uu @ k_ff + k_ff.T @ Q_u
    return [V_x, V_xx, k_trj, K_trj, x_trj, u_trj, expected_cost_reduc]

@jax.jit
def backward_pass(x_trj, u_trj):
    expected_cost_reduc = 0.
    k_trj = jnp.empty_like(u_trj)
    K_trj = jnp.empty((N-1, u_dim, x_dim))
    V_x = final_cost_jac(x_trj[-1])
    V_xx = final_cost_hessian(x_trj[-1])
    V_x, V_xx, k_trj, K_trj, x_trj, u_trj, expected_cost_reduc = jax.lax.fori_loop(0, 
                                                                                    N-1, backward_pass_body, 
                                                                                    [V_x, V_xx, k_trj, K_trj, x_trj, u_trj, expected_cost_reduc])
        
    return k_trj, K_trj, expected_cost_reduc

@jax.jit
def ilqr_body(i, aux_state):
    x_trj, u_trj, cost_trace = aux_state
    k_trj, K_trj, expected_cost_reduc = backward_pass(x_trj, u_trj)
    x_trj_updated, u_trj_updated = forward_pass(x_trj_ref=x_trj, u_trj_ref=u_trj, k_trj_ref=k_trj, K_trj_ref=K_trj)
    cost = traj_cost(x_trj_updated, u_trj_updated)
    cost_trace = cost_trace.at[i].set(cost)
    # TODO add some stop condition (Line-Search)
    return [x_trj_updated, u_trj_updated, cost_trace]

def run_ilqr(x_init):
    # run iLQR
    cost_trace = jnp.zeros((max_iter,))
    u_trj = np.random.randn(N-1, u_dim) * 1.
    x_trj = init_forward(x_init, u_trj)
    cost_trace = cost_trace.at[0].set(traj_cost(x_trj, u_trj))
    x_trj, u_trj, cost_trace = jax.lax.fori_loop(1, max_iter + 1, ilqr_body, [x_trj, u_trj, cost_trace])
    return x_trj, u_trj, cost_trace

x_batch = jnp.array([[0.0, 0.0, 0.8, 0.0], 
                     [1.0, 0.0, 0.0, 0.0], 
                     [0.5, 0.5, 0.0, 0.4], 
                     [0.1, 1.0, 0.0, 0.0],
                     [0.0, 1.0, 0.2, 0.0],
                     [2.0, 1.0, 0.0, 0.0],
                     [0.0, 1.3, 0.4, 0.0],
                     [0.9, 1.0, 0.0, 0.0],
                     [3.0, 1.0, 0.4, 0.0],
                     [0.0, 2.0, 0.0, 0.0],
                     [0.3, 1.0, 0.1, 0.0]] * 50)

for i in range(10):
    ts = time.time()
    xsol, usol, cost_hist = jax.vmap( jax.jit(run_ilqr), in_axes=[0] )(x_batch)
    t_elapsed = time.time() - ts
    print(t_elapsed)

plt.figure(1)
for i in range(11):
      plt.plot(xsol[i,:,:])
plt.legend(['x','y','theta','v'])
plt.grid()
print(xsol[0,-1,:])
plt.show()

plt.figure(2)
for i in range(11):
      plt.plot(usol[i,:,:])
plt.grid()
plt.show()

plt.figure(3)
for i in range(11):
      plt.plot(cost_hist[i,:])
plt.grid()
plt.show()