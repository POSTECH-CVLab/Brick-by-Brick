from gym.envs.registration import register

register(
    id='LegoMnist-v0',
    entry_point='gym_lego.envs:LegoEnv_Mnist',
)

register(
    id='LegoMnistNoMask-v0',
    entry_point='gym_lego.envs:LegoEnv_Mnist_No_Mask',
)
