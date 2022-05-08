from gym.envs.registration import register

register(
    id='LegoMNIST-v0',
    entry_point='gym_lego.envs:LegoEnvMNIST',
)
