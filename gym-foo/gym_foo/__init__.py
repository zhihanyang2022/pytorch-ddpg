from gym.envs.registration import register

register(
    id='continuous-mountain-car-her-v0',
    entry_point='gym_foo.envs:Continuous_MountainCarEnv',
)