import numpy as np
import torch
import torch.nn as nn
from torch import optim
from replay_buffer_episodic import EpisodicBatch

def get_net(
        num_in:int,
        num_out:int,
        final_activation,  # e.g. nn.Tanh
        num_hidden_layers: int=5,
        num_neurons_per_hidden_layer: int=64
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

def cat_features(a, b):
    return torch.cat([a, b], dim=2)

class EpisodicActor(nn.Module):

    def __init__(self, obs_dim, action_dim):
        super(EpisodicActor, self).__init__()
        self.net = get_net(num_in=obs_dim, num_out=action_dim, final_activation=nn.Tanh())

    def forward(self, o):
        return self.net(o)

class EpisodicCritic(nn.Module):

    def __init__(self, obs_dim, action_dim):
        super(EpisodicCritic, self).__init__()
        self.net = get_net(num_in=obs_dim+action_dim, num_out=action_dim, final_activation=None)

    def forward(self, o, a):
        return self.net(torch.cat([o, a], dim=2))

class EpisodicParamsPool:

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

        self.actor  = EpisodicActor(obs_dim=input_dim, action_dim=action_dim)
        self.critic = EpisodicCritic(obs_dim=input_dim, action_dim=action_dim)
        self.critic_target = EpisodicCritic(obs_dim=input_dim, action_dim=action_dim)

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

    def clip_gradient(self, net):
        for param in net.parameters():
           param.grad.data.clamp_(-1, 1)

    def update_networks(self, batch: EpisodicBatch) -> None:

        # ==================================================
        # bellman equation loss (just like Q-learning)
        # ==================================================

        PREDICTIONS = self.critic(batch.o[:,:-1,:], batch.a)

        with torch.no_grad():

            TARGETS = batch.r + \
                      self.gamma * (1 - batch.d) * \
                      self.critic_target(batch.o, self.actor(batch.o))[:,1:,:]

        Q_LEARNING_LOSS = torch.mean((PREDICTIONS - TARGETS.detach()) ** 2)

        # ==================================================
        # policy loss (not present in Q-learning)
        # ==================================================

        Q_VALUES = self.critic(batch.o[:,:-1,:], self.actor(batch.o[:,:-1,:]))
        ACTOR_LOSS = - torch.mean(Q_VALUES)

        # ==================================================
        # backpropagation and gradient descent
        # ==================================================

        self.actor_optimizer.zero_grad()
        ACTOR_LOSS.backward()  # gradient for actor
        # inconveniently this back-props into the critic as well, but (see following line)

        self.critic_optimizer.zero_grad()  # clear the gradient of the prediction net accumulated by ACTOR_LOSS.backward()
        Q_LEARNING_LOSS.backward()  # gradient for critic only

        self.clip_gradient(self.actor)
        self.clip_gradient(self.critic)

        self.actor_optimizer.step()
        self.critic_optimizer.step()

        # ==================================================
        # update the target network
        # ==================================================

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * self.polyak + param.data * (1 - self.polyak))

    def act(self, o) -> np.array:

        o = torch.tensor(o).unsqueeze(0).float()

        with torch.no_grad():
            greedy_action = self.actor(o)
        greedy_action = greedy_action.cpu().numpy().reshape(-1)

        return np.clip(greedy_action + self.noise_var * np.random.randn(self.action_dim), -1.0, 1.0)

    def decay_noise_var(self) -> None:
        if self.noise_var > self.noise_var_min:
            self.noise_var *= self.noise_var_multiplier