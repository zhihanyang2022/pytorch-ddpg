import gym
from replay_buffer import ReplayBuffer, Transition
from params_pool import ParamsPool

env = gym.make('MountainCarContinuous-v0')
buf = ReplayBuffer(capacity=60000)
param = ParamsPool(
    input_dim=env.observation_space.shape[0],
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

        action = param.act(obs, noisy=True)
        next_obs, reward, done, _ = env.step(action)
        # no need to keep track of max time-steps, because the environment
        # is wrapped with TimeLimit automatically (timeout after 1000 steps)

        total_reward += reward

        mask = 0 if done else 1

        # ==================================================
        # storing it to the buffer
        # ==================================================

        reward += 13 * abs(next_obs[1])
        buf.push(Transition(obs, action, reward, next_obs, mask))

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

    if buf.ready_for(batch_size):
        param.decay_noise_var()

    print(f'Episode {e:4.0f} | Return {total_reward:7.3f} | Noise var {param.noise_var:5.3f} | Updates {total_updates:4.0f}')

env.close()