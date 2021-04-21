import numpy as np
import torch
import torch.nn as nn
from torch import optim
from replay_buffer_recurrent import RecurrentBatchForTraining

def get_net(
        num_in:int,
        num_out:int,
        final_activation,  # e.g. nn.Tanh
        num_hidden_layers:int=3,
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

class ObsBasedRecurrentNet(nn.Module):

    def __init__(self, num_in, num_out):
        super(ObsBasedRecurrentNet, self).__init__()
        self.preprocessing_net = get_net(num_in=num_in, num_out=64, final_activation=nn.ReLU())
        self.lstm = nn.LSTM(input_size=64, hidden_size=num_out, batch_first=True)

    def forward(self, obs, hidden):
        out = self.preprocessing_net(obs)
        out, new_hidden = self.lstm(out, hx=hidden)
        return out, new_hidden

class RecurrentParamsPool:

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

        self.obs_based_recurrent_net = ObsBasedRecurrentNet(num_in=input_dim, num_out=64)
        self.q_prediction_net = get_net(num_in=64 + action_dim, num_out=1,          final_activation=None)
        self.q_target_net =     get_net(num_in=64 + action_dim, num_out=1,          final_activation=None)
        self.q_maximizing_net = get_net(num_in=64,              num_out=action_dim, final_activation=nn.Tanh())

        self.q_target_net.eval()  # we won't be passing gradients to this network
        self.q_target_net.load_state_dict(self.q_prediction_net.state_dict())

        # ===== optimizers =====

        # ref: https://pytorch.org/docs/stable/optim.html
        self.obs_based_recurrent_net_optimizer = optim.Adam(self.obs_based_recurrent_net.parameters(), lr=1e-3)
        self.q_prediction_net_optimizer = optim.Adam(self.q_prediction_net.parameters(), lr=5e-4)
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

    def clip_gradient_like_huber(self, net):
        for param in net.parameters():
            param.grad.data.clamp_(-1, 1)

    def update_networks(self, batch: RecurrentBatchForTraining) -> None:

        # ==================================================
        # bellman equation loss (just like Q-learning)
        # ==================================================

        # ** Shapes of tensors in batch **
        # - batch.o: (bs, seq_len, obs_dim)
        # - batch.a: (bs, seq_len, action_dim)
        # - batch.r: (bs, seq_len, 1)
        # - batch.o_prime: (bs, seq_len, obs_dim)
        # - batch.mask: (bs, seq_len, 1)
        # - batch.h0: (1, bs, hidden_size)
        # - batch.c0: (1, bs, hidden_size)
        # - batch.c1: (1, bs, hidden_size)
        # - batch.c2: (1, bs, hidden_Size)
        # * more details on preprocessing are available in replay_buffer_recurrent.py
        # * utilize R2D2's store state strategy but not the burn-in strategy

        # PyTorch made its linear layers really convenient for dealing with time-series data.
        # - In standard use cases, if you instantiate a nn.Linear(5, 6) and pass to it a tensor of
        # shape (64, 5), then you get a tensor of shape (64, 6).
        # - In the recurrent case, if you instantiate the same nn.Linear(5, 6) and pass to it a tensor
        # of shape (64, 100, 5) where 100 is the seq_len, the you get a tensor of shape (64, 100, 6).

        s_proxy, _ = self.obs_based_recurrent_net(batch.o, (batch.h0, batch.c0))  # s_proxy's shape: (bs, seq_len, hidden_dim)
        PREDICTIONS = self.q_prediction_net(torch.cat([s_proxy, batch.a], dim=2))  # PREDICTION's shape: (bs, seq_len, 1)

        s_prime_proxy, _ = self.obs_based_recurrent_net(batch.o_prime, (batch.h1, batch.c1))  # s_prime_proxy's shape: (bs, seq_len, hidden_dim)
        q_maximizing_a_prime = self.q_maximizing_net(s_prime_proxy)  # q_maximizing_a_prime's shape: (bs, seq_len, action_dim)
        TARGETS = batch.r + \
                  self.gamma * batch.mask * self.q_target_net(torch.cat([s_prime_proxy, q_maximizing_a_prime], dim=2))  # TARGET's shape: (bs, seq_len, 1)

        Q_LEARNING_LOSS = torch.mean((PREDICTIONS - TARGETS.detach()) ** 2)

        # ==================================================
        # policy loss (not present in Q-learning)
        # ==================================================

        q_maximizing_a = self.q_maximizing_net(s_proxy)  # s_proxy need to be back-propagated through
        Q_VALUES = self.q_prediction_net(torch.cat([s_proxy, q_maximizing_a], dim=2))  # again, s_proxy need to be back-propagated through

        ACTOR_LOSS = - torch.mean(Q_VALUES)  # minimizing this loss is maximizing the q values

        # ==================================================
        # backpropagation and gradient descent
        # ==================================================

        self.obs_based_recurrent_net_optimizer.zero_grad()  # need gradients from both losses

        self.q_maximizing_net_optimizer.zero_grad()
        ACTOR_LOSS.backward(retain_graph=True)  # inconveniently this back-props into prediction net as well, but (see following line)

        self.q_prediction_net_optimizer.zero_grad()  # clear the gradient of the prediction net accumulated by ACTOR_LOSS.backward()
        Q_LEARNING_LOSS.backward()

        self.clip_gradient_like_huber(self.q_prediction_net)
        self.clip_gradient_like_huber(self.q_maximizing_net)
        self.clip_gradient_like_huber(self.obs_based_recurrent_net)

        self.q_prediction_net_optimizer.step()
        self.q_maximizing_net_optimizer.step()
        self.obs_based_recurrent_net_optimizer.step()

        # ==================================================
        # update the target network
        # ==================================================

        for target_param, param in zip(self.q_target_net.parameters(), self.q_prediction_net.parameters()):
            target_param.data.copy_(target_param.data * self.polyak + param.data * (1 - self.polyak))

    def act(self, obs: np.array, hidden: torch.tensor) -> np.array:

        obs = torch.tensor(obs).unsqueeze(0).unsqueeze(0).float()
        state_proxy, new_hidden = self.obs_based_recurrent_net(obs, hidden)

        greedy_action = self.q_maximizing_net(state_proxy).detach().numpy().reshape(-1)

        return np.clip(greedy_action + self.noise_var * np.random.randn(self.action_dim), -1.0, 1.0), new_hidden

    def decay_noise_var(self) -> None:
        if self.noise_var > self.noise_var_min:
            self.noise_var *= self.noise_var_multiplier