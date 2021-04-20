import argparse
import numpy as np
import gym
from gym.wrappers import TimeLimit

from replay_buffer import ReplayBuffer, Transition, EpisodeBuffer
from params_pool import ParamsPool

parser = argparse.ArgumentParser()
parser.add_argument('--use_her', type=int)
args = parser.parse_args()

goal = 0.45  # aligned with the environment's force-done mechanism
goal_dim = 1
proj = lambda state: state[0]  # useful for creating hindsight goals from final states
goal_achieved = lambda state, goal : proj(state) == goal  # maps from state to bool

env = TimeLimit(gym.make('gym_foo:continuous-mountain-car-her-v0'), max_episode_steps=1000)
buf = ReplayBuffer(capacity=60000)
if args.use_her:
    episode_buf = EpisodeBuffer(
        goal=goal,
        proj_fn=proj,
        goal_achieved_fn=goal_achieved
    )
param = ParamsPool(
    input_dim=env.observation_space.shape[0] + goal_dim,
    action_dim=env.action_space.shape[0],
    noise_var=0.1,
    noise_var_multiplier=1,
    polyak=0.5
)

batch_size = 64
num_episodes = 1000  # enough for convergence

for e in range(num_episodes):

    obs = env.reset()

    total_reward = 0
    total_updates = 0

    while True:

        # ==================================================
        # getting the tuple (s, a, r, s', done)
        # ==================================================

        if args.use_her:
            action = param.act(np.append(obs, goal), noisy=True)
        else:
            action = param.act(obs, noisy=True)

        next_obs, _, done, _ = env.step(action)  # done means goal is achieved or timeout
        reward = 0 if goal_achieved(obs, goal) else -1

        total_reward += reward  # just for logging purposes

        # ==================================================
        # storing it to the buffer
        # ==================================================

        if args.use_her:
            episode_buf.push(Transition(obs, action, reward, next_obs, done))
        else:
            buf.push(Transition(obs, action, reward, next_obs, done))

        # ==================================================
        # update the parameters
        # ==================================================

        if buf.ready_for(batch_size):
            param.update_q_prediction_net_and_q_maximizing_net(buf.sample(batch_size))
            param.update_q_target_net()
            total_updates += 1

        # ==================================================
        # check done
        # ==================================================

        if done: break

        obs = next_obs

    # ==================================================
    # after each episode
    # ==================================================

    if args.use_her:
        episode_buf.push_standard_and_her_transitions_to(buf)
        episode_buf.reset()

    if buf.ready_for(batch_size):
        param.decay_noise_var()

    print(f'Episode {e:4.0f} | Return {total_reward:15.10f} | Noise var {param.noise_var:5.3f} | Updates {total_updates:4.0f}')

env.close()