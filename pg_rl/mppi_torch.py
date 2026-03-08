import gym
import numpy as np
from numpy.random import normal

# Set up system parameters
dt = 0.02
n_steps = 50
n_samples = 100
Sigma = 1.0
lamba = 0.1

# Set up cost function
def running_cost(x, u):
    # Pendulum cost function
    theta = x[0]
    theta_dot = x[1]
    return theta**2 + 0.1*theta_dot**2 + 0.001*u**2

def terminal_cost(x):
    # Pendulum cost function
    theta = x[0]
    theta_dot = x[1]
    return theta**2 + 0.1*theta_dot**2

# Set up OpenAI Gym environment
env_sim = gym.make('Pendulum-v1', render_mode="human")
env_ctrl = gym.make('Pendulum-v1')

env_sim.reset()
env_ctrl.reset()

# Initialize random number generator
rng = np.random.default_rng()

# Initialize control sequence and costs
U = np.zeros((n_samples, n_steps))
Epsilon = np.zeros((n_samples, n_steps))
C = np.zeros((n_samples,))

# Main loop
for t in range(1000):
    # Get current state
    curr_x = env_sim.state

    # ============= controller =============== #
    # Generate control sequence and costs for each sample traj
    for i in range(n_samples):
        # Initialize state and cost for this sample
        env_ctrl.state = curr_x
        cost_i = 0.0
        epsilons = normal(loc=0.0, scale=Sigma, size=(n_steps,))
        Epsilon[i] = epsilons
        # Generate control sequence and costs for each time step
        for j in range(n_steps):
            # Sample control sequence
            u = U[i, j] + epsilons[j]
            # Apply control and update state
            
            x, _, _, _ = env_ctrl.step([u])
            # Save control and cost
            U[i, j] = u
            # Update cost
            cost_i += running_cost(x, u) + lamba * u * epsilons[j] / Sigma

        cost_i += terminal_cost(x)
        C[i] = cost_i

    beta = np.min(C)
    # Compute weights and control signal
    exp_c = np.exp(-1.0 / lamba * (C - beta))
    yita = np.sum(exp_c, axis=0)
    w = exp_c / yita
    real_U = np.zeros((n_steps,))
    for k in range(n_samples):
        real_U = U[k] + np.sum( w[k] * Epsilon[k], axis=0 )
    u_star = U[:, 0]
    print(u_star)
    # Apply optimal control signal to environment
    x, _, _, _ = env_sim.step([u_star])

# Close environment
env_sim.close()
