import torch
import gym
import numpy as np
from gym.wrappers import TimeLimit
from collections import deque
import time

from replay_buffer import ReplayBuffer, Transition
from replay_buffer_episodic import EpisodicReplayBuffer

from params_pool import ParamsPool
from params_pool_episodic import EpisodicParamsPool

from wrappers import ActionScalingWrapper, PartialObsWrapper, PartialObsConcatWrapper

import wandb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--version', type=str)  # can be mdp, concat-pomdp, pomdp
parser.add_argument('--run_id', type=int)
args = parser.parse_args()

# wandb.init(
#     project='recurrent-ddpg-sac',
#     entity='pomdpr',
#     group=f'ddpg-pendulum-{args.version}',
#     settings=wandb.Settings(_disable_stats=True),
#     name=f'run_id={args.run_id}'
# )

env = ActionScalingWrapper(env=gym.make('Pendulum-v0'), scaling_factor=2)
input_dim = env.observation_space.shape[0]

# ==================================================

buf = EpisodicReplayBuffer(capacity=1000, episode_len=200, obs_dim=3, action_dim=1)
param = EpisodicParamsPool(
    input_dim=input_dim,  # different for different versions of the environment
    action_dim=env.action_space.shape[0],
    noise_var=0.01,
    noise_var_multiplier=1,
    polyak=0.95
)


batch_size = 64
num_episodes = 1000

for e in range(num_episodes):

    obs = env.reset()

    total_reward = 0
    total_updates = 0

    num_steps = 0

    while True:

        # ==================================================
        # getting the tuple (s, a, r, s', done)
        # ==================================================

        action = param.act(obs)

        next_obs, reward, done, _ = env.step(action)
        num_steps += 1

        total_reward += reward

        # ==================================================
        # storing it to the buffer
        # ==================================================

        buf.push(obs, action, reward, next_obs, int(done))

        # ==================================================
        # update the parameters
        # ==================================================

        if buf.ready_for(batch_size) and (num_steps % 50 == 0):
            param.update_networks(buf.sample(batch_size))
            total_updates += 1

        # ==================================================
        # check done
        # ==================================================

        if done: break

        obs = next_obs

    # ==================================================
    # after each episode
    # ==================================================

    # wandb.log({'return': total_reward})

    if buf.ready_for(batch_size):
        param.decay_noise_var()

    print(f'Episode {e:4.0f} | Steps {e * 200:6.0f} | Return {total_reward:9.3f} | Noise var {param.noise_var:5.3f} | Updates {total_updates:4.0f}')

env.close()