import random
import numpy as np
import torch
from collections import namedtuple, deque

EpisodicBatch = namedtuple('EpisodicBatch', 'o a r d')

class EpisodicReplayBuffer(object):

    def __init__(self, capacity: int, episode_len: int, obs_dim: int, action_dim: int):

        self.capacity = capacity  # number of episode to store
        self.episode_len = episode_len
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.o_entire = np.zeros((capacity, episode_len + 1, obs_dim))
        self.a_entire = np.zeros((capacity, episode_len, action_dim))
        self.r = np.zeros((capacity, episode_len, 1))
        self.d = np.zeros((capacity, episode_len, 1))

        self.episode_ptr = 0
        self.time_ptr = 0

        self.num_episodes = 0

    def push(self, o, a, r, no, d) -> None:

        self.o_entire[self.episode_ptr, self.time_ptr] = o
        self.a_entire[self.episode_ptr, self.time_ptr] = a
        self.r[self.episode_ptr, self.time_ptr] = r
        self.d[self.episode_ptr, self.time_ptr] = d

        if d:
            self.o_entire[self.episode_ptr, self.time_ptr+1] = no
            self.episode_ptr = (self.episode_ptr + 1) % self.capacity
            self.time_ptr = 0
            if self.num_episodes < self.capacity:
                self.num_episodes += 1
        else:
            self.time_ptr += 1

    def ready_for(self, batch_size: int) -> bool:
        return self.num_episodes >= batch_size

    def sample(self, batch_size: int) -> EpisodicBatch:

        indices = np.random.randint(self.num_episodes, size=batch_size)

        o = torch.tensor(self.o_entire[indices]).view(batch_size, self.episode_len+1, self.obs_dim).float()
        a = torch.tensor(self.a_entire[indices]).view(batch_size, self.episode_len, self.action_dim).float()
        r = torch.tensor(self.r[indices]).view(batch_size, self.episode_len, 1).float()
        d = torch.tensor(self.d[indices]).view(batch_size, self.episode_len, 1).float()

        return EpisodicBatch(o, a, r, d)