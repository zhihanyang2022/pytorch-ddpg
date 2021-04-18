import gym
from replay_buffer import ReplayBuffer, Transition
from params_pool import ParamsPool
import matplotlib.pyplot as plt

env = gym.make('MountainCarContinuous-v0')
buf = ReplayBuffer(capacity=50000)
param = ParamsPool(
    input_dim=env.observation_space.shape[0],
    action_dim=env.action_space.shape[0],
    noise_var=0.1,
    noise_var_multiplier=1
)

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
    loss1, loss2 = param.update_q_prediction_net_and_q_maximizing_net(batch)
    loss1_array.append(loss1)
    loss2_array.append(loss2)

fig = plt.figure(figsize=(12, 6))

fig.add_subplot(121)
plt.plot(loss1_array)
plt.title('q-learning loss')

fig.add_subplot(122)
plt.plot(loss2_array)
plt.title('actor loss')

plt.show()