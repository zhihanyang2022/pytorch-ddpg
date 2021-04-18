import numpy as np
import random
import torch
from collections import namedtuple, deque
from operator import itemgetter

Transition = namedtuple('Transition', 's a r s_prime mask')
Batch = namedtuple('Batch', 's a r s_prime mask')

class ReplayBuffer(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, transition):
        self.memory.appendleft(transition)

    def ready_for(self, batch_size):
        if len(self.memory) >= batch_size:
            return True
        return False

    @staticmethod
    def check_equal(batch1, batch2):
        if torch.equal(batch1.s, batch2.s):
            return True
        return False

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        batch = Batch(*zip(*batch))
        self.state = torch.tensor(batch.s, dtype = torch.float).view(batch_size, -1)
        self.action = torch.tensor(batch.a, dtype = torch.float).view(batch_size, -1)
        self.next_state = torch.tensor(batch.s_prime, dtype = torch.float).view(batch_size, -1)
        self.reward = torch.tensor(batch.r, dtype = torch.float).view(batch_size, 1)
        self.mask = torch.tensor(batch.mask, dtype = torch.long).view(batch_size, 1)
        return Batch(self.state, self.action, self.reward, self.next_state, self.mask)

    def __len__(self):
        return len(self.memory)

# class ReplayBuffer:
#
#     def __init__(self, capacity):
#         self.transitions = []
#         self.capacity = capacity
#
#     def ready_for(self, batch_size):
#         if len(self.transitions) >= batch_size:
#             return True
#         return False
#
#     def push(self, transition: Transition) -> None:
#         self.transitions.append(transition)
#         if len(self.transitions) > self.capacity:
#             self.transitions = self.transitions[1:]
#
#     def sample(self, batch_size:int) -> Batch:
#
#         transitions = random.sample(self.transitions, batch_size)
#         batch_np = Batch(*zip(*transitions))
#
#         s = torch.tensor(batch_np.s, dtype=torch.float)
#         a = torch.tensor(batch_np.a, dtype=torch.float)
#         r = torch.tensor(batch_np.r, dtype=torch.float)
#         s_prime = torch.tensor(batch_np.s_prime, dtype=torch.float)
#         mask = torch.tensor(batch_np.mask, dtype=torch.long)
#
#         batch_torch = Batch(s, a, r, s_prime, mask)
#
#         return batch_torch
