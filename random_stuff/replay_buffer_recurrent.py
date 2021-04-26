from collections import namedtuple
import numpy as np
import torch

RecurrentTransition = namedtuple('RecurrentTransition', 'o a r n mask')
RecurrentBatch = namedtuple('RecurrentBatch', 'o a r mask')

class RecurrentReplayBuffer(object):

    def __init__(self, max_episodes, episode_len, o_dim, a_dim):

        self.max_episodes = max_episodes
        self.episode_len = episode_len

        self.o_dim = o_dim
        self.a_dim = a_dim

        self.num_episodes = 0

        self.o = np.zeros((max_episodes, episode_len+1, o_dim))
        self.a = np.zeros((max_episodes, episode_len, a_dim))
        self.r = np.zeros((max_episodes, episode_len, 1))
        self.m = np.zeros((max_episodes, episode_len, 1))

        self.e_pointer = 0  # episode pointer
        self.t_pointer = 0  # time-step pointer

    def push(self, transition: RecurrentTransition) -> None:

        self.o[self.e_pointer, self.t_pointer] = transition.o
        self.a[self.e_pointer, self.t_pointer] = transition.a
        self.r[self.e_pointer, self.t_pointer] = transition.r
        self.m[self.e_pointer, self.t_pointer] = transition.mask

        if transition.mask == 0: # done
            self.o[self.e_pointer, self.t_pointer+1] = transition.n  # add the final transition to complete history
            self.e_pointer = (self.e_pointer + 1) % self.max_episodes
            self.t_pointer = 0
            if self.num_episodes < self.max_episodes:
                self.num_episodes += 1
        else:
            self.t_pointer += 1

        # sanity checks
        assert self.e_pointer < self.max_episodes
        assert self.t_pointer < self.episode_len

    def ready_for(self, batch_size: int) -> bool:
        assert batch_size <= self.max_episodes
        return batch_size <= self.num_episodes

    def sample(self, batch_size: int) -> RecurrentBatch:

        indices = np.random.randint(self.num_episodes, size=batch_size)

        o_batch = torch.tensor(self.o[indices], dtype=torch.float).view(batch_size, self.episode_len+1, self.o_dim)
        a_batch = torch.tensor(self.a[indices], dtype=torch.float).view(batch_size, self.episode_len, self.a_dim)
        r_batch = torch.tensor(self.r[indices], dtype=torch.float).view(batch_size, self.episode_len, 1)
        m_batch = torch.tensor(self.m[indices], dtype=torch.float).view(batch_size, self.episode_len, 1)

        return RecurrentBatch(o_batch, a_batch, r_batch, m_batch)
