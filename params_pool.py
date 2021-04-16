import numpy as np
import torch
import torch.nn as nn
from torch import optim

from replay_buffer import Batch

import time

class OUNoise(object):
    def __init__(self, low, high, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.01, decay_period=10000):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = 1
        self.low = low
        self.high = high
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self, action, t):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)

def get_net(
        num_in:int,
        num_out:int,
        final_activation,  # e.g. nn.Tanh
        num_hidden_layers:int=1,
        num_neurons_per_hidden_layer:int=64
    ) -> nn.Sequential:

    layers = []

    layers.extend([
        nn.Linear(num_in, num_neurons_per_hidden_layer),
        nn.ReLU(),
    ])

    for _ in range(num_hidden_layers):
        layers.extend([
            nn.Linear(num_neurons_per_hidden_layer, num_neurons_per_hidden_layer),
            nn.ReLU(),
        ])

    layers.append(nn.Linear(num_neurons_per_hidden_layer, num_out))

    if final_activation is not None:
        layers.append(final_activation)

    return nn.Sequential(*layers)

class ParamsPool:

    """
    ParamPool stands for parameter pool. This is inspired by the fact that everything
    in this class, including the behavior and target policies + the prediction and target
    Q networks all depend heavily on lots of parameters.
    Of course, it also involves methods to update the parameters in the face of new data.
    Exposed arguments:
        input_dim (int): dimension of input of the two q networks
        action_dim (int): dimension of output of the two q networks
        epsilon_multiplier (float): epsilon is multiplier by this constant after each episode
        epsilon_min (float): epsilon will be decayed until it reaches this threshold
        gamma (float): discount factor
    Un-exposed arguments (that you might want to play with):
        number of layers and neurons in each layer
        learning rate
        epsilon decay schedule

    a_ stands for normalized action
    """

    def __init__(self,
            input_dim:int,
            action_dim:int,
            action_lower_bounds:np.array,
            action_upper_bounds:np.array,
            gamma:float=0.95,
            noise_var:float=0,
            noise_var_multiplier:float=0.93,
            noise_var_min:float=0.5,
            polyak:float=0.995
        ):

        # ===== networks =====

        # q_prediction_net: (s, a_) --- network --> scalar
        # q_target_net    : (s, a_) --- network --> scalar
        # q_maximizing_net: s --- network --> a_ (in (0, 1) and hence need to undo normalization)

        self.q_prediction_net = get_net(num_in=input_dim + action_dim, num_out=1,          final_activation=None)
        self.q_target_net =     get_net(num_in=input_dim + action_dim, num_out=1,          final_activation=None)
        self.q_maximizing_net = get_net(num_in=input_dim,              num_out=action_dim, final_activation=nn.Tanh())

        self.q_target_net.eval()  # we won't be passing gradients to this network

        self.update_q_target_net()  # probably won't matter much but I think it's good practice

        # ===== optimizers =====

        # ref: https://pytorch.org/docs/stable/optim.html
        self.q_prediction_net_optimizer = optim.Adam(self.q_prediction_net.parameters())
        self.q_maximizing_net_optimizer = optim.Adam(self.q_maximizing_net.parameters())

        # ===== hyper-parameters =====

        # for doing / un-doing normalization
        self.action_range_center = (action_lower_bounds + action_upper_bounds) / 2
        self.action_range_radius = (action_upper_bounds - action_lower_bounds) / 2  # half of length

        # for discounting
        self.gamma = gamma

        # for exploration during training
        self.action_dim = action_dim
        self.noise_var = noise_var
        self.noise_var_multiplier = noise_var_multiplier
        self.noise_var_min = noise_var_min

        # for updating the q target network
        self.polyak =polyak

    def update_q_prediction_net(self, batch:Batch) -> None:
        """Pseudo-code step 12 and 13"""

        # start = time.perf_counter()

        a_ = self._normalize_action(batch.a)
        predictions = self.q_prediction_net(torch.cat([batch.s, a_], dim=1))

        q_maximizing_a_ = self.q_maximizing_net(batch.s_prime).detach()
        targets = batch.r + self.gamma * self.q_target_net(torch.cat([batch.s_prime, q_maximizing_a_], dim=1)) * batch.mask

        loss1 = torch.mean((targets - predictions) ** 2)

        # q_maximizing_a_ =
        q_values = self.q_prediction_net(torch.cat([batch.s, self.q_maximizing_net(batch.s)], dim=1))

        loss2 = - torch.mean(q_values)  # minimizing this loss is maximizing the q values

        self.q_prediction_net_optimizer.zero_grad()
        self.q_maximizing_net_optimizer.zero_grad()
        loss1.backward()
        loss2.backward()
        self._clip_gradient(self.q_prediction_net)
        self._clip_gradient(self.q_maximizing_net)
        self.q_prediction_net_optimizer.step()
        self.q_maximizing_net_optimizer.step()

        # print(time.perf_counter() - start)

    def update_q_maximizing_net(self, batch:Batch) -> None:
        """Pseudo-code step 14"""

        q_maximizing_a_ = self.q_maximizing_net(batch.s)
        q_values = self.q_prediction_net(torch.cat([batch.s, q_maximizing_a_], dim=1))

        loss = - torch.mean(q_values)  # minimizing this loss is maximizing the q values

        self.q_maximizing_net_optimizer.zero_grad()
        loss.backward()
        self._clip_gradient(self.q_maximizing_net)
        self.q_maximizing_net_optimizer.step()

    def _clip_gradient(self, net):
        """Equivalent to Huber loss. For improving stability."""
        for param in net.parameters():
            param.grad.data.clamp_(-1, 1)

    def update_q_target_net(self) -> None:

        # TODO: change to polyak averaging
        self.q_target_net.load_state_dict(self.q_prediction_net.state_dict())

    def act(self, state:np.array, noisy:bool) -> np.array:
        """
        For training: turn on deterministic
        For testing: turn off deterministic
        """

        state = torch.tensor(state).unsqueeze(0).float()

        with torch.no_grad():
            a_ = self.q_maximizing_net(state).numpy()[0]
            # use [0] instead of un-squeeze because un-squeeze gets rid of all extra brackets but we need one

        if noisy:
            a_ += np.random.normal(loc=0, scale=self.noise_var, size=self.action_dim)  # add noise
            a_ = np.clip(a_, -1, 1)  # clip into [-1, 1] for de-normalization

        return self._denormalize_action(a_)

    def act_randomly(self):
        return self._denormalize_action(np.random.uniform(low=-1, high=1, size=self.action_dim))

    def _normalize_action(self, action:np.array) -> np.array:
        return (action - self.action_range_center) / self.action_range_radius

    def _denormalize_action(self, action_:np.array) -> np.array:
        return action_ * self.action_range_radius + self.action_range_center

    def decay_noise_var(self) -> None:
        if self.noise_var > self.noise_var_min:
            self.noise_var *= self.noise_var_multiplier