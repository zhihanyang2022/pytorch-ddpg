import torch
import gym

from replay_buffer import ReplayBuffer, Transition
from replay_buffer_recurrent import RecurrentReplayBuffer, RecurrentTransition

from params_pool import ParamsPool
from params_pool_recurrent import RecurrentParamsPool

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

if args.version == 'mdp':

    env = ActionScalingWrapper(env=gym.make('Pendulum-v0'), scaling_factor=2)
    input_dim = env.observation_space.shape[0]

elif args.version == 'concat-pomdp':

    window_size = 2
    env = PartialObsConcatWrapper(
        ActionScalingWrapper(
            env=gym.make('Pendulum-v0'),
            scaling_factor=2
        ),
        window_size=window_size
    )
    input_dim = (env.observation_space.shape[0] - 1) * window_size

elif args.version == 'pomdp':

    env = PartialObsWrapper(ActionScalingWrapper(env=gym.make('Pendulum-v0'), scaling_factor=2))
    input_dim = env.observation_space.shape[0] - 1

else:

    raise NotImplementedError

# ==================================================

if args.version in ['mdp', 'concat-pomdp']:

    buf = ReplayBuffer(capacity=60000)
    param = ParamsPool(
        input_dim=input_dim,  # different for different versions of the environment
        action_dim=env.action_space.shape[0],
        noise_var=0.01,
        noise_var_multiplier=1,
        polyak=0.95
    )

elif args.version  == 'pomdp':

    buf = RecurrentReplayBuffer(capacity=60000, estimated_episode_len=200, bptt_len=5)
    param = RecurrentParamsPool(
        input_dim=input_dim,
        action_dim=env.action_space.shape[0],
        noise_var=0.01,
        noise_var_multiplier=1,
        polyak=0.95
    )

batch_size = 10
num_episodes = 1000  # enough for convergence

for e in range(num_episodes):

    obs = env.reset()

    total_reward = 0
    total_updates = 0

    if args.version == 'pomdp':
        h = torch.zeros((1, 1, 64))
        c = torch.zeros((1, 1, 64))

    while True:

        # ==================================================
        # getting the tuple (s, a, r, s', done)
        # ==================================================

        if args.version in ['mdp', 'concat-pomdp']:
            action = param.act(obs, noisy=True)
        elif args.version == 'pomdp':
            action, (new_h, new_c) = param.act(obs, (h, c))

        next_obs, reward, done, _ = env.step(action)
        # no need to keep track of max time-steps, because the environment
        # is wrapped with TimeLimit automatically (timeout after 1000 steps)

        total_reward += reward

        mask = 0 if done else 1

        # ==================================================
        # storing it to the buffer
        # ==================================================

        #reward += 13 * abs(next_obs[1])
        if args.version in ['mdp', 'concat-pomdp']:
            buf.push(Transition(obs, action, reward, next_obs, mask))
        elif args.version == 'pomdp':
            buf.push(RecurrentTransition(obs, action, reward, next_obs, mask, h.detach().numpy(), c.detach().numpy()))

        # ==================================================
        # update the parameters
        # ==================================================

        if args.version in ['mdp', 'concat-pomdp']:
            if buf.ready_for(batch_size):
                param.update_q_prediction_net_and_q_maximizing_net(buf.sample(batch_size))
                param.update_q_target_net()
                total_updates += 1
        elif args.version == 'pomdp':
            if buf.ready_for(batch_size):
                param.update_networks(buf.sample(batch_size))
                total_updates += 1

        # ==================================================
        # check done
        # ==================================================

        if done: break

        obs = next_obs
        if args.version == 'pomdp':
            h, c = new_h, new_c

    # ==================================================
    # after each episode
    # ==================================================

    # wandb.log({'return': total_reward})

    if buf.ready_for(batch_size):
        param.decay_noise_var()

    print(f'Episode {e:4.0f} | Return {total_reward:9.3f} | Noise var {param.noise_var:5.3f} | Updates {total_updates:4.0f}')

env.close()