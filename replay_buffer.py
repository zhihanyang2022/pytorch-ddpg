import random
import torch
from collections import namedtuple, deque

Transition = namedtuple('Transition', 's a r s_prime mask')
Batch = namedtuple('Batch', 's a r s_prime mask')

class ReplayBuffer(object):

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, transition: Transition) -> None:
        self.memory.appendleft(transition)

    def ready_for(self, batch_size: int) -> bool:
        if len(self.memory) >= batch_size:
            return True
        return False

    def sample(self, batch_size: int) -> Batch:
        batch = random.sample(self.memory, batch_size)
        batch = Batch(*zip(*batch))
        self.state = torch.tensor(batch.s, dtype = torch.float).view(batch_size, -1)
        self.action = torch.tensor(batch.a, dtype = torch.float).view(batch_size, -1)
        self.next_state = torch.tensor(batch.s_prime, dtype = torch.float).view(batch_size, -1)
        self.reward = torch.tensor(batch.r, dtype = torch.float).view(batch_size, 1)
        self.mask = torch.tensor(batch.mask, dtype = torch.float).view(batch_size, 1)
        return Batch(self.state, self.action, self.reward, self.next_state, self.mask)