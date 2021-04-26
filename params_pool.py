import numpy as np
import torch
import torch.nn as nn
from torch import optim
from replay_buffer import Batch

def get_net(
        num_in:int,
        num_out:int,
        final_activation,  # e.g. nn.Tanh
        num_hidden_layers:int=6,
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
    Description: ParamPool stands for parameter pool. This is inspired by the fact that everything
    in this class all depend heavily on lots of parameters. Of course, it also involves methods
    to update the parameters in the face of new data.

    Caution: This class can deal with multi-dimensional actions, but the range of actions has
    to be within [-1, 1]. To ensure it in gym, you can use wrappers. A list of official wrappers
    are available at https://github.com/openai/gym/tree/master/gym/wrappers; you should pay
    attention to the rescale_action.py. If you are coming from the future, then note that this
    link might be deprecated.

    Exposed arguments:
        input_dim (int): dimension of input of the two q networks
        action_dim (int): dimension of output of the two q networks
        noise_var (float): variance of exploration noise added to greedy actions
        noise_var_multiplier (float)
        polyak (float): polyak-averaging coefficient; interpreted as the proportion of target parameters to keep

    Un-exposed arguments (that you might want to play with):
        learning rates
        number of layers and neurons in each layer
        exploration noise decay schedule
    """

    def __init__(self,
            input_dim:int,
            action_dim:int,
            gamma:float=0.95,
            noise_var:float=0.1,
            noise_var_multiplier:float=0.95,
            noise_var_min:float=0,
            polyak:float=0.90
        ):

        # ===== networks =====

        # q_prediction_net: (s, a_) --- network --> scalar
        # q_target_net    : (s, a_) --- network --> scalar
        # q_maximizing_net: s --- network --> a_ (in (0, 1) and hence need to undo normalization)

        self.q_prediction_net = get_net(num_in=input_dim + action_dim, num_out=1,          final_activation=None)
        self.q_target_net =     get_net(num_in=input_dim + action_dim, num_out=1,          final_activation=None)
        self.q_maximizing_net = get_net(num_in=input_dim,              num_out=action_dim, final_activation=nn.Tanh())

        self.q_target_net.eval()  # we won't be passing gradients to this network
        self.q_target_net.load_state_dict(self.q_prediction_net.state_dict())

        # ===== optimizers =====

        # ref: https://pytorch.org/docs/stable/optim.html
        self.q_prediction_net_optimizer = optim.Adam(self.q_prediction_net.parameters(), lr=1e-3)
        self.q_maximizing_net_optimizer = optim.Adam(self.q_maximizing_net.parameters(), lr=1e-3)

        # ===== hyper-parameters =====

        # for discounting
        self.gamma = gamma

        # for exploration during training
        self.action_dim = action_dim
        self.noise_var = noise_var
        self.noise_var_multiplier = noise_var_multiplier
        self.noise_var_min = noise_var_min

        # for updating the q target network
        self.polyak = polyak

    def update_q_prediction_net_and_q_maximizing_net(self, batch: Batch) -> tuple:

        # ==================================================
        # bellman equation loss (just like Q-learning)
        # ==================================================

        PREDICTIONS = self.q_prediction_net(torch.cat([batch.s, batch.a], dim=1))

        q_maximizing_a_prime = self.q_maximizing_net(batch.s_prime)
        # oh my, this bug in the following line took me 2 days or so to find it
        # basically, if batch.mask has shape (64, ) and its multiplier has shape (64, 1)
        # the result is a (64, 64) tensor, but this does not even cause an error!!!
        TARGETS = batch.r + \
                  self.gamma * self.q_target_net(torch.cat([batch.s_prime, q_maximizing_a_prime], dim=1)) * batch.mask
        Q_LEARNING_LOSS = torch.mean((PREDICTIONS - TARGETS.detach()) ** 2)

        # ==================================================
        # policy loss (not present in Q-learning)
        # ==================================================

        q_maximizing_a = self.q_maximizing_net(batch.s)
        Q_VALUES = self.q_prediction_net(torch.cat([batch.s, q_maximizing_a], dim=1))

        ACTOR_LOSS = - torch.mean(Q_VALUES)  # minimizing this loss is maximizing the q values

        # ==================================================
        # backpropagation and gradient descent
        # ==================================================

        self.q_maximizing_net_optimizer.zero_grad()
        ACTOR_LOSS.backward()  # inconveniently this back-props into prediction net as well, but (see following line)
        self.q_prediction_net_optimizer.zero_grad()  # clear the gradient of the prediction net accumulated by ACTOR_LOSS.backward()
        Q_LEARNING_LOSS.backward()

        # doing a gradient clipping between -1 and 1 is equivalent to using Huber loss
        # guaranteed to improve stability so no harm in using at all
        for param in self.q_prediction_net.parameters():
            param.grad.data.clamp_(-1, 1)
        for param in self.q_maximizing_net.parameters():
            param.grad.data.clamp_(-1, 1)

        self.q_prediction_net_optimizer.step()
        self.q_maximizing_net_optimizer.step()

        return float(Q_LEARNING_LOSS), float(ACTOR_LOSS)

    def update_q_target_net(self) -> None:

        for target_param, param in zip(self.q_target_net.parameters(), self.q_prediction_net.parameters()):
            target_param.data.copy_(target_param.data * self.polyak + param.data * (1 - self.polyak))

    def act(self, state: np.array, noisy: bool) -> np.array:
        """
        For training: turn on deterministic
        For testing: turn off deterministic
        """
        state = torch.tensor(state).unsqueeze(0).float()

        greedy_action = self.q_maximizing_net(state).detach().numpy()[0]
        # use [0] instead of un-squeeze because un-squeeze gets rid of all extra brackets but we need one

        if noisy:
            return np.clip(greedy_action + self.noise_var * np.random.randn(self.action_dim), -1.0, 1.0)
        else:
            return greedy_action

    def decay_noise_var(self) -> None:
        if self.noise_var > self.noise_var_min:
            self.noise_var *= self.noise_var_multiplier