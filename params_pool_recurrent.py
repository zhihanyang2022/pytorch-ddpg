import numpy as np
import torch
import torch.nn as nn
from torch import optim
from replay_buffer_recurrent import RecurrentBatchForTraining

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

class RecurrentActor(nn.Module):

    def __init__(self, obs_dim, action_dim):
        super(RecurrentActor, self).__init__()
        self.net1 = get_net(num_in=obs_dim, num_out=64, final_activation=nn.ReLU())
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, batch_first=True)
        self.net2 = get_net(num_in=64, num_out=action_dim, final_activation=nn.Tanh())

    def forward(self, obs, hidden=None, output_new_hidden=False):
        out = self.net1(obs)
        out, new_hidden = self.lstm(out, hidden)
        out = self.net2(out)
        if output_new_hidden:
            return out, new_hidden
        else:
            return out

class RecurrentCritic(nn.Module):

    def __init__(self, obs_dim, action_dim):
        super(RecurrentCritic, self).__init__()
        self.net1 = get_net(num_in=obs_dim, num_out=64, final_activation=nn.ReLU())
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, batch_first=True)
        self.net2 = get_net(num_in=64+action_dim, num_out=1, final_activation=None)

    def forward(self, obs, action):
        out = self.net1(obs)
        out, new_hidden = self.lstm(out)
        out = self.net2(torch.cat([out, action], dim=2))
        return out

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

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.actor  = RecurrentActor(obs_dim=input_dim, action_dim=action_dim).to(self.device)
        self.critic = RecurrentCritic(obs_dim=input_dim, action_dim=action_dim).to(self.device)
        self.critic_target = RecurrentCritic(obs_dim=input_dim, action_dim=action_dim).to(self.device)

        self.critic_target.eval()  # we won't be passing gradients to this network
        self.critic_target.load_state_dict(self.critic.state_dict())

        # ===== optimizers =====

        # ref: https://pytorch.org/docs/stable/optim.html
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

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

        def slice_burn_in(item):
            burn_in_length = 3
            return item[:, burn_in_length:, :]

        entire_history = torch.cat([batch.o, batch.o_prime[:,-1,:].unsqueeze(1)], dim=1)

        PREDICTIONS = self.critic(batch.o, batch.a)  # (bs, seq_len, 1)
        PREDICTIONS = slice_burn_in(PREDICTIONS)

        TARGETS = batch.r + self.gamma * batch.mask * self.critic_target(entire_history, self.actor(entire_history))[:,1:,:]  # (bs, seq_len, 1)
        TARGETS = slice_burn_in(TARGETS)

        Q_LEARNING_LOSS = torch.mean((PREDICTIONS - TARGETS.detach()) ** 2)

        # ==================================================
        # policy loss (not present in Q-learning)
        # ==================================================

        Q_VALUES = self.critic(batch.o, self.actor(batch.o))
        Q_VALUES = slice_burn_in(Q_VALUES)
        ACTOR_LOSS = - torch.mean(Q_VALUES)

        # ==================================================
        # backpropagation and gradient descent
        # ==================================================

        self.actor_optimizer.zero_grad()
        ACTOR_LOSS.backward()  # gradient for actor
        # inconveniently this back-props into the critic as well, but (see following line)

        self.critic_optimizer.zero_grad()  # clear the gradient of the prediction net accumulated by ACTOR_LOSS.backward()
        Q_LEARNING_LOSS.backward()  # gradient for critic only

        self.clip_gradient_like_huber(self.actor)
        self.clip_gradient_like_huber(self.critic)

        self.actor_optimizer.step()
        self.critic_optimizer.step()

        # ==================================================
        # update the target network
        # ==================================================

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * self.polyak + param.data * (1 - self.polyak))

    def act(self, obs: np.array, hidden: torch.tensor) -> np.array:

        obs = torch.tensor(obs).unsqueeze(0).unsqueeze(0).float().to(self.device)

        greedy_action, new_hidden = self.actor(obs, hidden, output_new_hidden=True)
        greedy_action = greedy_action.detach().cpu().numpy().reshape(-1)

        return np.clip(greedy_action + self.noise_var * np.random.randn(self.action_dim), -1.0, 1.0), new_hidden

    def decay_noise_var(self) -> None:
        if self.noise_var > self.noise_var_min:
            self.noise_var *= self.noise_var_multiplier