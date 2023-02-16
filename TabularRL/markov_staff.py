"""Markov-related contents"""
import torch
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda:0')

def from_numpy(np_array: np.ndarray):
    return torch.from_numpy(np_array).float().to(device)

def to_numpy(tensor):
    return tensor.to('cpu').detach().numpy()

np.random.seed(0)

# states
States = [1, 2, 3, 4, 5, 6]

# rewards of state
Rewards = torch.tensor([-1., -2., -2., 10., 1., 0.])

# transition matrix P
P_np = np.array([
                [0.9, 0.1, 0.0, 0.0, 0.0, 0.0],
                [0.5, 0.0, 0.5, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.6, 0.0, 0.4],
                [0.0, 0.0, 0.0, 0.0, 0.3, 0.7],
                [0.0, 0.2, 0.3, 0.5, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                ], dtype=np.float32)

plt.matshow(P_np)
plt.show()

P = torch.from_numpy(P_np)

def compute_return(state_traj, state_index:int, gamma:float):
    """compute the return of a state in a traj"""
    G = 0
    for i in reversed( range(state_index, len(state_traj)) ):
        G += gamma * Rewards[state_traj[i] - 1]
    return G

# example of computation of rewards
# state sequence: s1->s2->s3->s6
traj = [1, 2, 3, 6]
start_index = 0
G = compute_return(traj, start_index, 0.5)
print(f'return of starting state: {G}')

#=========== Bellman Equation for Markov Reward Progress =========
# sovle a n-linear equation x = Ax + b
# A is the transition matrix: gamma*P
# x is the value function of state: S
# b is the reward list: R

# Solve the value function of state via Bellman Equation anytically
# for small-scale discrete problem
gamma = 0.5
V = np.linalg.inv(np.eye(len(States)) - gamma*P_np) @ to_numpy(Rewards)

print(f'Value function: {V}')

# check for value of state 4
print(f'checking : {V[3]}=={to_numpy(Rewards[3]) + gamma * P_np[3,:] @ V} ' )

# ======== Bellman Expected Equation for Markov Decesion Making Progress (MDP) ====
# Example

# state space
S = ["s1", "s2", "s3", "s4", "s5"]  
# action space
A = ["保持s1", "前往s1", "前往s2", "前往s3", "前往s4", "前往s5", "概率前往"]  
# state transition function
P = {
    "s1-保持s1-s1": 1.0,
    "s1-前往s2-s2": 1.0,
    "s2-前往s1-s1": 1.0,
    "s2-前往s3-s3": 1.0,
    "s3-前往s4-s4": 1.0,
    "s3-前往s5-s5": 1.0,
    "s4-前往s5-s5": 1.0,
    "s4-概率前往-s2": 0.2,
    "s4-概率前往-s3": 0.4,
    "s4-概率前往-s4": 0.4,
}
# reward function
R = {
    "s1-保持s1": -1,
    "s1-前往s2": 0,
    "s2-前往s1": -1,
    "s2-前往s3": -2,
    "s3-前往s4": -2,
    "s3-前往s5": 0,
    "s4-前往s5": 10,
    "s4-概率前往": 1,
}
# discount factor
gamma = 0.5  
MDP = (S, A, P, R, gamma)

# policy 1 --> random policy
Pi_1 = {
    "s1-保持s1": 0.5,
    "s1-前往s2": 0.5,
    "s2-前往s1": 0.5,
    "s2-前往s3": 0.5,
    "s3-前往s4": 0.5,
    "s3-前往s5": 0.5,
    "s4-前往s5": 0.5,
    "s4-概率前往": 0.5,
}
# policy 2 --> expert policy
Pi_2 = {
    "s1-保持s1": 0.6,
    "s1-前往s2": 0.4,
    "s2-前往s1": 0.3,
    "s2-前往s3": 0.7,
    "s3-前往s4": 0.5,
    "s3-前往s5": 0.5,
    "s4-前往s5": 0.1,
    "s4-概率前往": 0.9,
}


# helper
def join(str1, str2):
    return str1 + '-' + str2

# state transition matrix
P_from_mdp_to_mrp = [
    [0.5, 0.5, 0.0, 0.0, 0.0],
    [0.5, 0.0, 0.5, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.5, 0.5],
    [0.0, 0.1, 0.2, 0.2, 0.5],
    [0.0, 0.0, 0.0, 0.0, 1.0],
]

P_from_mdp_to_mrp = np.array(P_from_mdp_to_mrp)
R_from_mdp_to_mrp = [-0.5, -1.5, -1.0, 5.5, 0]

def compute(P, rewards, gamma, states_num):
    """using bellman equation"""
    rewards = np.array(rewards).reshape((-1, 1))  
    value = np.dot(np.linalg.inv(np.eye(states_num, states_num) - gamma * P), rewards)
    return value

V = compute(P_from_mdp_to_mrp, R_from_mdp_to_mrp, gamma, 5)

print("\nstate value in MDP\n", V)
