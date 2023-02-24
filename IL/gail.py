"""generative adversarial imitation learning, GAIL"""
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from expert_data_gen import device, ENV_NAME, generate_expert_data, PPO
import numpy as np
import matplotlib.pyplot as plt
import gym

class Discriminator(nn.Module):
    """Discriminator"""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim+action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1), nn.Sigmoid()
        )

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)
        return self.net(cat)

class GAIL:
    def __init__(self, agent, state_dim, action_dim, hidden_dim, lr_d) -> None:
        self.discriminator = Discriminator(state_dim,
                                           action_dim, 
                                           hidden_dim).to(device)
        self.discriminator_optimizer = optim.Adam(
            self.discriminator.parameters(), lr=lr_d
            )
        # binary_cross_entropy
        self.discriminator_loss = nn.BCELoss()
        self.agent = agent
    
    def learn(self, expert_s, expert_a, agent_s, agent_a, next_s, dones):
        # train the discriminator
        expert_states = torch.tensor(expert_s, dtype=torch.float).to(device)
        expert_actions = torch.tensor(expert_a).to(device)
        agent_states = torch.tensor(agent_s, dtype=torch.float).to(device)
        agent_actions = torch.tensor(agent_a).to(device)

        expert_actions = F.one_hot(expert_actions, num_classes=2).float()
        agent_actions = F.one_hot(agent_actions, num_classes=2).float()

        expert_prob = self.discriminator(expert_states, expert_actions)
        agent_prob = self.discriminator(agent_states, agent_actions)

        discriminator_loss = self.discriminator_loss(agent_prob, torch.ones_like(agent_prob)) \
                           + self.discriminator_loss(expert_prob, torch.zeros_like(expert_prob))
                         
        self.discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        self.discriminator_optimizer.step()

        rewards = -torch.log(agent_prob).detach().cpu().numpy()
        transition_dict = {
            'states': agent_s,
            'actions': agent_a,
            'rewards': rewards, # instead of using the rewards designed from the environment
            'next_states': next_s,
            'dones': dones
        }
        # train agent
        self.agent.update(transition_dict)

env = gym.make(ENV_NAME)
env.seed(0)
torch.manual_seed(0)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
env.seed(0)
torch.manual_seed(0)
np.random.seed(0)
lr_d = 1e-3
actor_lr = 1e-3
critic_lr = 1e-2
num_episodes = 250
hidden_dim = 128
gamma = 0.98
lmbda = 0.95
epochs = 10
eps = 0.2

ppo_agent = PPO(state_dim, 
                hidden_dim, 
                action_dim, 
                actor_lr, 
                critic_lr, 
                lmbda,
                epochs, 
                eps, 
                gamma, 
                device)
gail = GAIL(ppo_agent, state_dim, action_dim, hidden_dim, lr_d)
n_episode = 500
return_list = []

print(id(ppo_agent) == id(gail.agent))

expert_s, expert_a, expert_reward = generate_expert_data()


for i in range(n_episode):
    episode_return = 0
    state = env.reset()
    done = False
    state_list = []
    action_list = []
    next_state_list = []
    done_list = []
    
    while not done:
        action = gail.agent.take_action(state)
        next_state, reward, done, _ = env.step(action)
        state_list.append(state)
        action_list.append(action)
        next_state_list.append(next_state)
        done_list.append(done)
        state = next_state
        episode_return += reward
    
    return_list.append(episode_return)

    gail.learn(expert_s, expert_a, state_list, action_list,
                next_state_list, done_list)
    
    if (i + 1) % 50 == 0:
        print(f'==== Episode #: {i}   Return: {np.mean(return_list[-10:])} ====')
    

iteration_list = list(range(len(return_list)))
plt.plot(iteration_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('GAIL on {}'.format(ENV_NAME))
plt.show()