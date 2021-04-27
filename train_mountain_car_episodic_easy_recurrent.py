import gym
from gym.wrappers import TimeLimit
from gym.wrappers import Monitor

from wrappers import ActionScalingWrapper
from replay_buffer_episodic import EpisodicReplayBuffer
from params_pool_recurrent import RecurrentParamsPool

import wandb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--run_id', type=int)
args = parser.parse_args()

wandb.init(
    project='recurrent-ddpg-sac',
    entity='pomdpr',
    group=f'ddpg-recurrent-mountain-car-pomdp-episodic-easy',
    settings=wandb.Settings(_disable_stats=True),
    name=f'run_id={args.run_id}'
)

# env = Monitor(
#     TimeLimit(
#         ActionScalingWrapper(gym.make('gym_custom:pomdp-mountain-car-episodic-v0'), scaling_factor=5),
#         max_episode_steps=200
#     ),
#     directory=f'results/pomdp-mountain-car-episodic-v0/{args.run_id}',
#     force=True
# )
env = TimeLimit(
    ActionScalingWrapper(gym.make('gym_custom:pomdp-mountain-car-episodic-easy-v0'), scaling_factor=15),
    max_episode_steps=15  # enough for this one
)
input_dim = 3
action_dim = 1

# env = ActionScalingWrapper(env=gym.make('Pendulum-v0'), scaling_factor=2)
# input_dim = env.observation_space.shape[0]

# ==================================================

buf = EpisodicReplayBuffer(capacity=5000, episode_len=15, obs_dim=input_dim, action_dim=action_dim)
param = RecurrentParamsPool(
    input_dim=input_dim,
    action_dim=action_dim
)

batch_size = 64
num_episodes = 5000 * 3  # 3 million transitions; 5000 * 3 * 5 updates

for e in range(num_episodes):

    obs = env.reset()
    param.reset_hidden()

    total_reward = 0
    total_updates = 0

    while True:

        # ==================================================
        # getting the tuple (s, a, r, s', done)
        # ==================================================

        action = param.act(obs)

        next_obs, reward, done, _ = env.step(action)

        total_reward += reward

        # ==================================================
        # storing it to the buffer
        # ==================================================

        buf.push(obs, action, reward, next_obs, int(done))

        # ==================================================
        # update the parameters
        # ==================================================

        # ==================================================
        # check done
        # ==================================================

        if done: break

        obs = next_obs

    # ==================================================
    # after each episode
    # ==================================================

    wandb.log({'return': total_reward})

    if buf.ready_for(batch_size):
        for i in range(5):
            param.update_networks(buf.sample(batch_size))
            total_updates += 1

    if buf.ready_for(batch_size):
        param.decay_noise_var()

    print(f'Episode {e:4.0f} | Steps {e * 200:6.0f} | Return {total_reward:9.3f} | Noise var {param.noise_var:5.3f} | Updates {total_updates:4.0f}')

env.close()