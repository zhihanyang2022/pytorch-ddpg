import random
import numpy as np
import torch
from collections import namedtuple, deque
from typing import Callable

Transition = namedtuple('Transition', 's a r s_prime done')
Batch = namedtuple('Batch', 's a r s_prime done')

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
        s = torch.tensor(batch.s, dtype = torch.float).view(batch_size, -1)
        a = torch.tensor(batch.a, dtype = torch.float).view(batch_size, -1)
        r = torch.tensor(batch.r, dtype=torch.float).view(batch_size, 1)
        s_prime = torch.tensor(batch.s_prime, dtype = torch.float).view(batch_size, -1)
        done = torch.tensor(batch.done, dtype = torch.long).view(batch_size, 1)
        return Batch(s, a, r, s_prime, done)

class EpisodeBuffer:

    def __init__(self, goal: np.array, proj_fn: Callable, goal_achieved_fn: Callable):
        self.memory = []
        self.goal = goal
        self.proj_fn = proj_fn
        self.goal_achieved_fn = goal_achieved_fn  # element-wise threshold checking
        self.expired = False

    def push(self, transition: Transition) -> None:
        assert self.expired is False, "The EpisodeBuffer has expired; please call its reset() method."
        self.memory.append(transition)

    def push_standard_and_her_transitions_to(self, replay_buffer: ReplayBuffer) -> None:

        hindsight_goal = self.proj_fn(self.memory[-1].s)  # not s_prime!

        for transition in self.memory:

            replay_buffer.push(Transition(
                np.append(transition.s, self.goal),
                transition.a,
                transition.r,
                np.append(transition.s_prime, self.goal),
                transition.done
            ))  # standard UVFA transition

            # print(self.goal_achieved_fn(transition.s, hindsight_goal))

            replay_buffer.push(Transition(
                np.append(transition.s, hindsight_goal),
                transition.a,
                0 if self.goal_achieved_fn(transition.s, hindsight_goal) else -1,
                np.append(transition.s_prime, hindsight_goal),
                1 if self.goal_achieved_fn(transition.s, hindsight_goal) else 0
            ))  # HER transition

        self.expired = True

    def reset(self):
        self.memory = []
        self.expired = False