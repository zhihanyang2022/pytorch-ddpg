import gym
env = gym.make('MountainCarContinuous-v0')
import time

from replay_buffer import ReplayBuffer, Transition
from params_pool import ParamsPool

buf = ReplayBuffer(capacity=50000)
param = ParamsPool(
    input_dim=env.observation_space.shape[0],
    action_dim=env.action_space.shape[0],
    action_lower_bounds=env.action_space.low,
    action_upper_bounds=env.action_space.high,
    noise_var=0.1,
    noise_var_multiplier=0.95
)
target_network_update_duration = 10
batch_size = 64

num_episodes = 1000  # enough for convergence

obs = env.reset()

action = param.act(obs, noisy=True)
next_obs, reward, done, _ = env.step(action)

mask = 0 if done else 1

reward += 13 * abs(next_obs[1])
buf.push(Transition(obs, action, reward, next_obs, mask))

batch = buf.sample(batch_size=1)

loss1_array = []
loss2_array = []
for i in range(1000):
    loss1, loss2 = param.update_q_prediction_net_and_q_maximizing_net()
    loss1_array.append(loss1)
    loss2_array.append(loss2)

import matplotlib.pyplot as plt

plt.plot(loss1_array, label='q-learning loss')
plt.plot(loss2_array, label='actor loss')
plt.legend()
plt.show()