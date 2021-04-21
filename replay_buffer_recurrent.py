import random
import numpy as np
import torch
from collections import namedtuple, deque

RecurrentTransition = namedtuple('RecurrentTransition', 'o a r o_prime mask h c')
RecurrentWindow     = namedtuple('RecurrentTransitions', 'o a r o_prime mask h c')
RecurrentBatch      = namedtuple('RecurrentTransitions', 'o a r o_prime mask h c')

RecurrentBatchForTraining = namedtuple('RecurrentBatch', 'o a r o_prime mask h0 c0 h1 c1')

class RecurrentReplayBuffer(object):

    """
    How this works?
    - max_episodes_to_store
    - ready_for
    - sample
    """

    def __init__(self, capacity: int, estimated_episode_len: int, bptt_len: int):
        max_episodes_to_store = capacity // estimated_episode_len
        self.episode  = []
        self.episodes = deque(maxlen=max_episodes_to_store)
        self.bptt_len = bptt_len

    def push(self, transition: RecurrentTransition) -> None:
        # all entries of transition should be in numpy format
        self.episode.append(transition)  # h and c has shape: (1, 1, hidden_size)
        if transition.mask == 0:  # done!
            self.episodes.append(self.episode)
            self.episode = []

    def ready_for(self, batch_size: int) -> bool:
        if len(self.episodes) >= batch_size:
            return True
        return False

    def sample(self, batch_size: int) -> RecurrentBatchForTraining:

        episode_lengths = [len(episode) for episode in self.episodes]
        sampled_episodes = random.sample(self.episodes, k=batch_size) #counts=episode_lengths) # TODO

        # counts change the sampling probabilities to account for the fact that episodes might be of different lengths

        windows = []
        for episode in sampled_episodes:
            index = np.random.randint(len(episode) - self.bptt_len)
            transitions = episode[index:index+self.bptt_len]
            window = RecurrentWindow(*zip(*transitions))  # each field corresponds to a list of items
            windows.append(window)

        batch = RecurrentBatch(*zip(*windows))  # each field contains a list of lists (each correspond to a window) of items

        o       = torch.tensor(np.array(batch.o      )).view(batch_size, self.bptt_len, -1).float()
        a       = torch.tensor(np.array(batch.a      )).view(batch_size, self.bptt_len, -1).float()
        r       = torch.tensor(np.array(batch.r      )).view(batch_size, self.bptt_len,  1).float()
        o_prime = torch.tensor(np.array(batch.o_prime)).view(batch_size, self.bptt_len, -1).float()
        masks   = torch.tensor(np.array(batch.mask   )).view(batch_size, self.bptt_len,  1).float()  # long does not work with some pytorch versions / cuda
        h       = torch.tensor(np.array(batch.h      )).view(batch_size, self.bptt_len, -1).float()
        c       = torch.tensor(np.array(batch.c      )).view(batch_size, self.bptt_len, -1).float()

        # np.array(batch.s) has shape (batch_size, bptt_len, obs_dim)

        # np.array(batch.h) has shape (batch_size, bptt_len, 1, 1, hidden_size)
        # this get converted to shape (batch_size, bptt_len, hidden_size)

        h0 = h[:, 0, :].unsqueeze(0)  # target shape: (1, batch_size, hidden_size)
        c0 = c[:, 0, :].unsqueeze(0)
        h1 = h[:, 1, :].unsqueeze(0)
        c1 = c[:, 1, :].unsqueeze(0)

        return RecurrentBatchForTraining(o, a, r, o_prime, masks, h0, c0, h1, c1)