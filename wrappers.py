from collections import deque
import numpy as np
import gym

class ActionScalingWrapper(gym.ActionWrapper):

    def __init__(self, env, scaling_factor: float):
        super(ActionScalingWrapper, self).__init__(env)
        self.scaling_factor = scaling_factor

    def action(self, action):
        return self.scaling_factor * action

class PartialObsWrapper(gym.ObservationWrapper):

    """Only present angle information to the agent, without angular velocity."""

    def __init__(self, env):
        super(PartialObsWrapper, self).__init__(env)

    def observation(self, observation):
        return observation[:2]  # only retain the first two floats (cos(theta), sin(theta))

class PartialObsConcatWrapper(gym.ObservationWrapper):

    def __init__(self, env, window_size: int):
        super(PartialObsConcatWrapper, self).__init__(env)
        self.window_size = window_size
        self.window = deque(maxlen=window_size)
        for i in range(self.window_size - 1):
            self.window.append(np.zeros((2, )))  # append some dummy observations first

    def observation(self, observation: np.array) -> np.array:
        self.window.append(observation[:2])
        return np.concatenate(self.window)
