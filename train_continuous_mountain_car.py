import gym
env = gym.make('MountainCarContinuous-v0')
import time

from replay_buffer import ReplayBuffer
from params_pool import ParamsPool

buf = ReplayBuffer(size=10000)
param = ParamsPool(
    input_dim=env.observation_space.shape[0],
    action_dim=env.action_space.shape[0],
    action_lower_bounds=env.action_space.low,
    action_upper_bounds=env.action_space.high,
    noise_var=1.0,
    noise_var_multiplier=0.95
)
target_network_update_duration = 10
max_steps = 200
batch_size = 32

num_episodes = 1000  # enough for convergence

for e in range(num_episodes):

    obs = env.reset()

    total_reward = 0
    total_steps = 0

    buf_sampled = False

    while True:

        # ===== getting the tuple (s, a, r, s', done) =====

        action = param.act(obs, noisy=True)
        next_obs, reward, done, _ = env.step(action)
        # no need to keep track of max time-steps, because the environment
        # is wrapped with TimeLimit

        # logistics
        total_reward += reward

        mask = 0 if done else 1

        # ===== storing it to the buffer =====

        buf.push(obs, action, reward, next_obs, mask)

        # ===== update the parameters =====

        # start = time.perf_counter()
        if buf.filled is False and buf_sampled is False:
            batch = buf.sample(batch_size=batch_size)
            buf_sampled = True
        if buf.filled:
            param.update_q_prediction_net(batch)

        # ===== check done =====

        if done: break

        obs = next_obs

    # ===== after an episode =====

    if buf.filled:
        if e % target_network_update_duration == 0:
            param.update_q_target_net()
        param.decay_noise_var()

    print(f'Episode {e:3.0f} | Return {total_reward:5.3f} | Noise var {param.noise_var:5.3f}')

env.close()