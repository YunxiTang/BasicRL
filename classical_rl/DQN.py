"""Deep Q Net Learning"""
import random, os, sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import deque
import rl_utils


if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')


# utils 
def to_numpy(tensor):
    return tensor.to('cpu').detach().numpy()

def from_numpy(x):
    """put a variable to a device tensor"""
    return torch.from_numpy(x).float().to(device)



class ReplayBuffer:
    """Replay Buffer for DQN"""
    def __init__(self, capacity) -> None:
        self._capacity = capacity
        self._buffer = deque(maxlen=self._capacity)

    def add(self, state:np.ndarray, action:np.ndarray, reward:float, next_state:np.ndarray, done:bool)  -> None:
        """add a data tuple"""
        self._buffer.append(
            (state, action, reward, next_state, done)
        )
        return None

    def sample(self, batch_size):
        """sample a batch size of trainning data from replay buffer"""
        transitions = random.sample(self._buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)
        return states, actions, rewards, next_states, dones

    @property
    def size(self):
        """data size in replay buffer"""
        return len(self._buffer)


class QCritic(nn.Module):
    """Q network"""
    def __init__(self, state_dim:int, action_dim:int, hidden_dim=64):
        super(QCritic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, state: torch.Tensor):
        return self.net(state)


class Agent_DQN:
    """DQN Algorithm"""
    def __init__(self, state_dim, action_dim, learning_rate, gamma, epsilon, target_update, device):
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._learning_rate = learning_rate
        self._gamma = gamma
        self._epsilon = epsilon
        self._target_update = target_update
        self._device = device
        self._TAU = 0.005

        self.q_net = QCritic(
            self._state_dim, 
            self._action_dim
        ).to(device=self._device)

        # target Q network
        self.target_q_net = QCritic(
            self._state_dim, 
            self._action_dim
        ).to(device=self._device)

        self.q_optimizer = optim.Adam(
            self.q_net.parameters(), 
            lr=self._learning_rate
        )

        self._count = 0

    @torch.no_grad()
    def select_action(self, state: np.array):
        """select action via epsilon-greedy policy"""
        if np.random.random() < self._epsilon:
            action = np.random.randint(self._action_dim)
        else:
            if len(state.shape) == 1:
                state = state[None]
            state_tensor = from_numpy(state)
            q_sa = self.q_net(state_tensor)
            action = q_sa.argmax(dim=1).item()
        return action
    
    def update(self, transition_dict):
        """update the Q network"""
        states = from_numpy( np.array(transition_dict['states']) )
        actions = torch.tensor( np.array(transition_dict['actions'], dtype=np.int64) ).view(-1, 1).to(self._device)
        rewards = from_numpy( np.array(transition_dict['rewards']) ).view(-1, 1) 
        next_states = from_numpy( np.array(transition_dict['next_states']) )
        dones = from_numpy( np.array(transition_dict['dones']) ).view(-1, 1)
        # shape checking
        assert states.shape[0] == actions.shape[0] == rewards.shape[0] == next_states.shape[0] == dones.shape[0], "Wrong dimensions"

        # Q(s,a)
        q_values = self.q_net(states).gather(1, actions)  

        # max Q on next state
        with torch.no_grad():
            max_next_q_values = self.target_q_net(next_states).max(dim=1)[0].view(-1, 1)

        # TD target
        q_targets = rewards + self._gamma * max_next_q_values * (1 - dones) 

        # loss
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  
        self.q_optimizer.zero_grad() 
        dqn_loss.backward()
        self.q_optimizer.step()

        if self._count % self._target_update == 0:
            # soft update of the target Q network
            target_q_net_state_dict = self.target_q_net.state_dict()
            q_net_state_dict = self.q_net.state_dict()
            for key in q_net_state_dict:
                target_q_net_state_dict[key] = q_net_state_dict[key] * self._TAU + target_q_net_state_dict[key] * (1-self._TAU)
                self.target_q_net.load_state_dict(target_q_net_state_dict)
            
            # hard update of the target Q network
            # self.target_q_net.load_state_dict(self.q_net.state_dict())  

        self._count += 1 

if __name__ == '__main__':

    import gym

    lr = 2e-3
    num_episodes = 500
    hidden_dim = 128
    gamma = 0.98
    epsilon = 0.01
    target_update = 10
    buffer_size = 10000
    minimal_size = 500
    batch_size = 64
    

    env_name = 'CartPole-v1'
    env = gym.make(env_name)
    random.seed(0)
    np.random.seed(0)
    env.seed(0)
    torch.manual_seed(0)
    replay_buffer = ReplayBuffer(buffer_size)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = Agent_DQN(
        state_dim,
        action_dim,
        lr,
        gamma,
        epsilon,
        target_update,
        device
    )

    return_list = []
    for i in range(50):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state = env.reset()
                done = False
                while not done:
                    action = agent.select_action(state)
                    next_state, reward, done, _ = env.step(action)
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    # start trainning when data is sufficient
                    if replay_buffer.size > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {
                            'states': b_s,
                            'actions': b_a,
                            'next_states': b_ns,
                            'rewards': b_r,
                            'dones': b_d
                        }
                        agent.update(transition_dict)

                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix(
                    {
                        'episode':
                        '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return':
                        '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)
    # plotting
    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DQN on {}'.format(env_name))
    plt.show()

    mv_return = rl_utils.moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DQN on {}'.format(env_name))
    plt.show()