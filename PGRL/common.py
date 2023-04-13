import numpy as np

class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, capacity):
        """
            Replay buffer
        """
        self.obs_buf = np.zeros([capacity, obs_dim], dtype=np.float32)
        self.act_buf = np.zeros([capacity, act_dim], dtype=np.float32)
        self.rew_buf = np.zeros([capacity, 1], dtype=np.float32)
        self.next_obs_buf = np.zeros([capacity, obs_dim], dtype=np.float32)
        self.done_buf = np.zeros([capacity, 1], dtype=bool)
        self.capacity = capacity
        self.pointer = 0
        self.size = 0

    def add(self, obs, act, reward, next_obs, done):
        """
            add a tuple
        """
        self.obs_buf[self.pointer, :] = obs
        self.act_buf[self.pointer, :] = act
        self.rew_buf[self.pointer, :] = reward
        self.next_obs_buf[self.pointer, :] = next_obs
        self.done_buf[self.pointer, :] = done

        self.pointer = (self.pointer + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """
            sample a batch size
        """
        indices = np.random.choice(self.size, size=batch_size, replace=False)
