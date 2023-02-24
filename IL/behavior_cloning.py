"""behavior cloning"""
from expert_data_gen import PolicyNet
import torch.optim as optim
import torch
from expert_data_gen import device, ENV_NAME, generate_expert_data
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import gym


class BehaviorClone:
    def __init__(self,state_dim, action_dim, hidden_dim=128, lr=1e-3) -> None:
        self.policy = PolicyNet(state_dim, action_dim, hidden_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def learn(self, states, actions):
        states = torch.tensor(states, dtype=torch.float).to(device)
        actions = torch.tensor(actions).view(-1, 1).to(device)
        log_probs = torch.log(self.policy(states).gather(1, actions))
        # bc loss
        bc_loss = torch.mean(-log_probs)  

        self.optimizer.zero_grad()
        bc_loss.backward()
        self.optimizer.step()
    
    @torch.no_grad()
    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(device)
        probs = self.policy(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

def test_agent(agent, env, n_episode):
    return_list = []
    for episode in range(n_episode):
        episode_return = 0
        state = env.reset()
        done = False
        while not done:
            action = agent.take_action(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            episode_return += reward
        return_list.append(episode_return)
    return np.mean(return_list)

env = gym.make(ENV_NAME)
env.seed(0)
torch.manual_seed(0)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
env.seed(0)
torch.manual_seed(0)
np.random.seed(0)

lr = 1e-3
bc_agent = BehaviorClone(state_dim, action_dim)
n_iterations = 1000
batch_size = 64
test_returns = []

expert_s, expert_a, expert_reward = generate_expert_data()

with tqdm(total=n_iterations, desc="pbar") as pbar:
    for i in range(n_iterations):
        sample_indices = np.random.randint(low=0,
                                           high=expert_s.shape[0],
                                           size=batch_size)
        bc_agent.learn(expert_s[sample_indices], expert_a[sample_indices])
        current_return = test_agent(bc_agent, env, 5)
        test_returns.append(current_return)
        if (i + 1) % 10 == 0:
            pbar.set_postfix({'return': '%.3f' % np.mean(test_returns[-10:])})
        pbar.update(1)

iteration_list = list(range(len(test_returns)))
plt.plot(iteration_list, test_returns)
plt.xlabel('Iterations')
plt.ylabel('Returns')
plt.title('BC on {}'.format(ENV_NAME))
plt.show()